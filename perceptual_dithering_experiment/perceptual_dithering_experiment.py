# README
# Phillip Long
# 12/17/2025

# Perceptual Dithering Experiment

# This script experiments with perceptual dithering using a trained 16-bit GPT-2 model.
# The process:
# 1. Load a trained model (trained with max_bit_depth=16)
# 2. Load an audio file
# 3. Extract MSB tokens from the audio
# 4. For each MSB token, predict the corresponding LSB token using the model
# 5. Reconstruct the full 16-bit audio from MSB + predicted LSB pairs
# 6. Save the result

# The model is fed MSB tokens (ground truth) and predicts LSB tokens, alternating:
# MSB₁ (ground truth) -> LSB₁ (predicted) -> MSB₂ (ground truth) -> LSB₂ (predicted) -> ...

# For stereo audio, channels are interleaved: left channel samples, then right channel samples.


# IMPORTS
##################################################

import torch
import torchaudio
import argparse
from tqdm import tqdm
from train_gpt2 import (
    GPTAudioLightningModule,
    quantize_unsigned_pcm_torch,
    msb_torch,
    lsb_torch,
)

##################################################


# FUNCTIONS
##################################################

def load_model(checkpoint_path, **model_kwargs):
    """
    Load a trained GPT-2 model from checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt or .pt file
        **model_kwargs: Additional arguments for .pt files (not needed for .ckpt)
    
    Returns:
        Loaded model in eval mode
    """
    if checkpoint_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint - preserves hyperparameters
        model = GPTAudioLightningModule.load_from_checkpoint(checkpoint_path)
    elif checkpoint_path.endswith('.pt'):
        # Raw state dict - need to reconstruct model
        model = GPTAudioLightningModule(**model_kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")
    
    model.eval()
    return model


def prepare_audio_tokens(audio_path, max_bit_depth=16, msb_n_bits=8, sample_rate=44100, stereo_interleave=False):
    """
    Load audio file and prepare MSB/LSB tokens.
    
    Args:
        audio_path: Path to audio file
        max_bit_depth: Bit depth (16 for this experiment)
        msb_n_bits: Number of bits for MSB (typically 8)
        sample_rate: Target sample rate
        stereo_interleave: If True, interleave stereo channels (LLLRRR format).
                          Must match the training configuration.
    
    Returns:
        msb_tokens: MSB tokens as torch.Tensor
        lsb_tokens_ground_truth: Ground truth LSB tokens (for comparison)
        original_audio: Original audio tensor for reference
        actual_sample_rate: Actual sample rate of loaded audio
    """
    # Load audio
    wav, sr = torchaudio.load(audio_path, normalize=True, backend="soundfile")
    
    # Resample if needed
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate).clamp(-1.0, 1.0)
        sr = sample_rate
    
    # Handle mono/stereo
    if wav.shape[0] > 2:
        # Multi-channel: take first two
        wav = wav[:2, :]
    
    # Quantize to 16-bit unsigned PCM
    wav_quantized = quantize_unsigned_pcm_torch(wav, n_bits=max_bit_depth, kind='linear')
    
    # Interleave stereo channels if requested (must match training config)
    if stereo_interleave and wav_quantized.shape[0] == 2:
        # Interleave: left channel samples, then right channel samples (LLLRRR format)
        left_channel = wav_quantized[0]  # (num_samples,)
        right_channel = wav_quantized[1]  # (num_samples,)
        interleaved = torch.cat([left_channel, right_channel], dim=0)  # (2 * num_samples,)
        wav_quantized = interleaved
    elif wav_quantized.shape[0] == 2 and not stereo_interleave:
        # If stereo but not interleaving, take only one channel (matching training behavior)
        # Randomly pick left or right (for consistency with training, we'll use left)
        wav_quantized = wav_quantized[0]
    
    # Extract MSB and LSB tokens
    msb_tokens = msb_torch(wav_quantized, orig_n_bits=max_bit_depth, n_bits=msb_n_bits).squeeze(dim=0)
    lsb_tokens_ground_truth = lsb_torch(wav_quantized, n_bits=max_bit_depth - msb_n_bits).squeeze(dim=0)
    assert len(msb_tokens) == len(lsb_tokens_ground_truth)
    assert len(msb_tokens) == len(wav_quantized) and len(lsb_tokens_ground_truth) == len(wav_quantized)
    
    return msb_tokens, lsb_tokens_ground_truth, wav, sr


def predict_lsb_tokens(model, msb_tokens):
    """
    Predict LSB tokens from MSB tokens using the trained model.
    
    The model was trained on interleaved sequences: [MSB₁, LSB₁, MSB₂, LSB₂, ...]
    So we feed MSB tokens (ground truth) and predict the corresponding LSB tokens.
    
    Process:
    1. Feed MSB₁ (ground truth) -> model predicts LSB₁ (greedy: highest probability)
    2. Feed MSB₂ (ground truth) -> model predicts LSB₂ (greedy: highest probability)
    3. Continue alternating...
    
    Note: Both MSB and LSB tokens are in range [0, 255] for 8-bit splits.
    The model learns to distinguish them based on position in the sequence.
    
    Args:
        model: Trained GPTAudioLightningModule
        msb_tokens: MSB tokens as torch.Tensor of shape (num_samples,). Assumed to be on the same device as the model.
    
    Returns:
        predicted_lsb_tokens: Predicted LSB tokens (greedy decoding)
    """
    model.eval()
    num_samples = len(msb_tokens)
    predicted_lsb_tokens = torch.zeros((num_samples,), dtype=torch.long, device=msb_tokens.device)
    
    # Build sequence incrementally: MSB₁ -> predict LSB₁ -> MSB₂ -> predict LSB₂ -> ...
    input_tokens = torch.zeros((1, 0), dtype=torch.long, device=msb_tokens.device)
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Predicting LSB tokens"):
            # Append the current MSB token (ground truth)
            msb_token = msb_tokens[i].unsqueeze(0).unsqueeze(0)  # (1, 1)
            input_tokens = torch.cat([input_tokens, msb_token], dim=1)  # (1, seq_len)
            
            # Get model prediction for next token (should be LSB)
            outputs = model(input_tokens)
            logits = outputs.logits[:, -1, :].detach()  # (1, vocab_size) - last position logits
            
            # Greedy decoding: pick the token with highest probability
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
            predicted_lsb_tokens[i] = next_token.item()
            
            # Append predicted LSB to sequence for next iteration
            input_tokens = torch.cat([input_tokens, next_token], dim=1)
    
    return predicted_lsb_tokens


def reconstruct_audio_from_tokens(msb_tokens, lsb_tokens, max_bit_depth=16, msb_n_bits=8):
    """
    Reconstruct 16-bit audio tokens from MSB and LSB tokens.
    
    Args:
        msb_tokens: MSB tokens
        lsb_tokens: LSB tokens
        max_bit_depth: Total bit depth (16)
        msb_n_bits: Number of bits in MSB (typically 8)
    
    Returns:
        reconstructed_tokens: Reconstructed 16-bit tokens
    """
    # Combine MSB and LSB: (MSB << (16 - msb_n_bits)) | LSB
    lsb_n_bits = max_bit_depth - msb_n_bits
    reconstructed = (msb_tokens << lsb_n_bits) | lsb_tokens
    return reconstructed


def tokens_to_audio(tokens, max_bit_depth=16):
    """
    Convert quantized tokens back to audio samples in range [-1, 1].
    
    Args:
        tokens: Quantized tokens (from quantize_unsigned_pcm_torch)
        max_bit_depth: Bit depth used for quantization
    
    Returns:
        audio: Audio tensor in range [-1, 1]
    """
    q_levels = 1 << max_bit_depth
    # Normalize to [0, 1]
    normalized = tokens.float() / (q_levels - 1)
    # Map to [-1, 1]
    audio = 2 * normalized - 1
    return audio.clamp(-1.0, 1.0)


def deinterleave_stereo(tokens, num_samples_per_channel):
    """
    Deinterleave stereo tokens back to left and right channels.
    
    Args:
        tokens: Interleaved tokens [L₁, L₂, ..., R₁, R₂, ...]
        num_samples_per_channel: Number of samples per channel
    
    Returns:
        left_channel: Left channel tokens
        right_channel: Right channel tokens
    """
    left_channel = tokens[:num_samples_per_channel]
    right_channel = tokens[num_samples_per_channel:]
    return left_channel, right_channel

##################################################


# MAIN METHOD
##################################################

def main():

    # defaults
    DEFAULT_CHECKPOINT = '/graft2/code/znovack/lnac/t5_lnac/qny3i4y9/checkpoints/gpt2audio-epoch=1379.ckpt'
    DEFAULT_AUDIO = '/mnt/arrakis_data/pnlong/lnac/youtube_mix/out000.wav'
    DEFAULT_OUTPUT = '/home/pnlong/zach_lnac/perceptual_dithering_experiment.wav'

    # parse arguments
    parser = argparse.ArgumentParser(description='Perceptual dithering experiment')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                       help='Path to trained model checkpoint (.ckpt or .pt)')
    parser.add_argument('--audio', type=str, default=DEFAULT_AUDIO,
                       help='Path to input audio file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                       help='Path to save output audio file')
    parser.add_argument('--msb_n_bits', type=int, default=8,
                       help='Number of bits for MSB (default: 8)')
    parser.add_argument('--max_bit_depth', type=int, default=16,
                       help='Total bit depth (default: 16)')
    parser.add_argument('--sample_rate', type=int, default=44100,
                       help='Target sample rate (default: 44100)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available (default: use CPU)')
    parser.add_argument('--save_ground_truth', action='store_true',
                       help='Also save quantized reconstruction (MSB + original LSB) for comparison')
    parser.add_argument('--stereo_interleave', action='store_true',
                       help='Use stereo interleaving (LLLRRR format). Must match training config. Default: False')
    parser.add_argument('--use_lora', action='store_true', # Model kwargs for .pt files
                       help='Whether model uses LoRA (only needed for .pt files)')
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    print("=" * 60)
    print("Perceptual Dithering Experiment")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model_kwargs = {}
    if args.checkpoint.endswith('.pt'):
        model_kwargs = {
            'model_name': 'gpt2',
            'use_lora': args.use_lora,
            'max_bit_depth': args.max_bit_depth,
        }
    model = load_model(args.checkpoint, **model_kwargs).to(device) # load model and move to device
    print(f"Model loaded on {device}")
    print(f"Model vocab size: {model.model.config.vocab_size}")
    
    # Prepare audio tokens
    print(f"\nLoading audio from {args.audio}...")
    print(f"Using stereo_interleave={args.stereo_interleave} (must match training config)")
    msb_tokens, lsb_tokens_gt, original_audio, actual_sr = prepare_audio_tokens(
        args.audio,
        max_bit_depth=args.max_bit_depth,
        msb_n_bits=args.msb_n_bits,
        sample_rate=args.sample_rate,
        stereo_interleave=args.stereo_interleave,
    )
    msb_tokens = msb_tokens.to(device)
    print(f"Audio loaded: {original_audio.shape[1]} samples at {actual_sr} Hz")
    print(f"MSB tokens: {msb_tokens.shape[-1]}")
    print(f"MSB token range: [{msb_tokens.min().item()}, {msb_tokens.max().item()}]")
    print(f"LSB token range (GT): [{lsb_tokens_gt.min().item()}, {lsb_tokens_gt.max().item()}]")
    
    # Predict LSB tokens
    print(f"\nPredicting LSB tokens (greedy decoding)...")
    predicted_lsb_tokens = predict_lsb_tokens(
        model=model,
        msb_tokens=msb_tokens,
    )
    print(f"Predicted LSB token range: [{predicted_lsb_tokens.min().item()}, {predicted_lsb_tokens.max().item()}]")
    
    # Reconstruct audio from predicted tokens
    print(f"\nReconstructing audio from predicted tokens...")
    reconstructed_tokens = reconstruct_audio_from_tokens(
        msb_tokens.cpu(),
        predicted_lsb_tokens.cpu(),
        max_bit_depth=args.max_bit_depth,
        msb_n_bits=args.msb_n_bits
    )
    predicted_audio = tokens_to_audio(reconstructed_tokens, max_bit_depth=args.max_bit_depth)
    
    # Handle stereo deinterleaving (only if we interleaved during preparation)
    if args.stereo_interleave and original_audio.shape[0] == 2:
        num_samples_per_channel = original_audio.shape[1]
        left_pred, right_pred = deinterleave_stereo(predicted_audio, num_samples_per_channel)
        predicted_audio = torch.stack([left_pred, right_pred], dim=0)
        print(f"Deinterleaved stereo: {predicted_audio.shape}")
    elif original_audio.shape[0] == 2 and not args.stereo_interleave:
        # Stero but not interleaved - we took one channel, so duplicate it back
        predicted_audio = predicted_audio.unsqueeze(0).repeat(2, 1)
    else:
        # Mono
        predicted_audio = predicted_audio.unsqueeze(0)  # Add channel dimension
    
    # Save predicted audio
    print(f"\nSaving predicted audio to {args.output}...")
    torchaudio.save(args.output, predicted_audio, args.sample_rate)
    print("Done!")
    
    # Optionally save quantized reconstruction for comparison
    # This is the audio reconstructed from MSB + original LSB (i.e., the quantized version of the original)
    # Note: Even if the original audio is already 16-bit PCM, this will differ slightly due to:
    # - Normalization (int16 → float32) when loading
    # - Floor() truncation during quantization
    # - Round-trip quantization error
    if args.save_ground_truth:
        quantized_tokens = reconstruct_audio_from_tokens(
            msb_tokens.cpu(),
            lsb_tokens_gt.cpu(),
            max_bit_depth=args.max_bit_depth,
            msb_n_bits=args.msb_n_bits
        )
        quantized_audio = tokens_to_audio(quantized_tokens, max_bit_depth=args.max_bit_depth)
        
        if args.stereo_interleave and original_audio.shape[0] == 2:
            left_quant, right_quant = deinterleave_stereo(quantized_audio, num_samples_per_channel)
            quantized_audio = torch.stack([left_quant, right_quant], dim=0)
        elif original_audio.shape[0] == 2 and not args.stereo_interleave:
            quantized_audio = quantized_audio.unsqueeze(0).repeat(2, 1)
        else:
            quantized_audio = quantized_audio.unsqueeze(0)
        
        quantized_output = f"{args.output.replace('.wav', '.quantized.wav')}"
        print(f"Saving quantized reconstruction (baseline) to {quantized_output}...")
        torchaudio.save(quantized_output, quantized_audio, args.sample_rate)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    mse = torch.mean((predicted_audio - original_audio) ** 2).item()
    print(f"Mean Squared Error (predicted vs original): {mse:.6f}")
    
    if args.save_ground_truth:
        mse_quantized = torch.mean((quantized_audio - original_audio) ** 2).item()
        print(f"Mean Squared Error (quantized reconstruction vs original): {mse_quantized:.6f}")
        print(f"  (This shows quantization error - predicted should be compared to this baseline)")
    
    print("\nExperiment complete!")

##################################################


# MAIN EXECUTION
##################################################

if __name__ == '__main__':
    main()

##################################################

