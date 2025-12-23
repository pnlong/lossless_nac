# README
# Phillip Long

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
from typing import Tuple
from os.path import exists, basename
from os import makedirs
from train_gpt2 import GPTAudioLightningModule, quantize_unsigned_pcm_torch, msb_torch, lsb_torch

##################################################

# CONSTANTS
##################################################

DEFAULT_CHECKPOINT = "/graft2/code/znovack/lnac/t5_lnac/qny3i4y9/checkpoints/gpt2audio-epoch=1379.ckpt"
DEFAULT_INPUT_AUDIO_PATH = "/mnt/arrakis_data/pnlong/lnac/youtube_mix/out000.wav"
DEFAULT_OUTPUT_DIR = "/home/pnlong/lnac/perceptual_dithering_experiment/results"
CHUNK_SIZE = 512 # number of samples to process at a time, since we can't process the entire sequence at once
CHUNK_OVERLAP = 64 # number of samples to overlap between chunks
MSB_N_BITS = 8 # number of bits for MSB
BIT_DEPTH = 16 # we assume inputs are 16-bit PCM
LSB_N_BITS = BIT_DEPTH - MSB_N_BITS

##################################################

# HELPER FUNCTIONS
##################################################

def load_model(checkpoint_path: str) -> GPTAudioLightningModule:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to trained model checkpoint (.ckpt or .pt)
    
    Returns:
        Trained GPTAudioLightningModule
    """

    # load model from checkpoint
    model = GPTAudioLightningModule.load_from_checkpoint(checkpoint_path)

    # return model
    return model

def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        waveform: Normalized audio tensor (n_channels, n_samples) where n_channels is 1 for mono and 2 for stereo
        sample_rate: Sample rate
    """

    # load audio file
    waveform, sample_rate = torchaudio.load(
        uri = audio_path,
        normalize = True, # returns waveform in the range [-1.0, 1.0]
        channels_first = True,
    )

    # return waveform and sample rate
    return waveform, sample_rate

def write_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    output_path: str,
) -> None:
    """
    Write an audio file. Assumes input waveform is 16-bit unsigned PCM.
    
    Args:
        waveform: 16-bit unsigned PCM audio tensor (n_channels, n_samples) where n_channels is 1 for mono and 2 for stereo
        sample_rate: Sample rate
        output_path: Path to output audio file
    """

    # ensure waveform is on cpu
    waveform = waveform.detach().cpu()

    # normalize waveform
    waveform = waveform.to(torch.float32) # convert waveform to floating point
    waveform = waveform / ((2 ** BIT_DEPTH) - 1) # waveform is now in the range [0.0, 1.0]
    waveform = (waveform * 2.0) - 1.0 # waveform is now in the range [-1.0, 1.0]

    # save waveform as audio file
    torchaudio.save(
        uri = output_path,
        src = waveform,
        sample_rate = sample_rate,
        channels_first = True,
    )

    # return nothing
    return

##################################################


# PARSE ARGUMENTS
##################################################

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description = "Perceptual dithering experiment")
    parser.add_argument("--checkpoint", type = str, default = DEFAULT_CHECKPOINT, help = "Path to trained model checkpoint (.ckpt or .pt)")
    parser.add_argument("--input_audio_path", type = str, default = DEFAULT_INPUT_AUDIO_PATH, help = "Path to input audio file")
    parser.add_argument("--output_dir", type = str, default = DEFAULT_OUTPUT_DIR, help = "Directory to save output audio files (original.wav and predicted.wav)")
    parser.add_argument("--gpu", action = "store_true", help = "Use GPU if available (default: use CPU)")
    parser.add_argument("--chunk_size", type = int, default = CHUNK_SIZE, help = "Number of samples to process at a time")
    parser.add_argument("--chunk_overlap", type = int, default = CHUNK_OVERLAP, help = "Number of samples to overlap between chunks")
    args = parser.parse_args()
    return args

##################################################


# MAIN FUNCTION
##################################################

def predict_chunk(
    model: GPTAudioLightningModule,
    chunk: torch.Tensor,
) -> torch.Tensor:
    """
    Predict a chunk of audio using perceptual dithering experiment framework.

    Args:
        model: Trained GPTAudioLightningModule
        chunk: Chunk of audio tensor (n_channels, chunk_size)

    Returns:
        Predicted chunk of audio tensor (n_channels, chunk_size)
    """

    # initialize
    model.eval()
    n_channels, n_samples = chunk.shape
    vocab_size = model.model.config.vocab_size

    # interleave stereo if necessary, ultimately yields a 1D tensor
    # if n_channels == 2: # stereo
    #     flattened_chunk = torch.cat((chunk[0, :], chunk[1, :]), dim = 0) # interleave channels, convert to 1D tensor
    # else: # mono
    #     flattened_chunk = chunk.squeeze(dim = 0) # convert to 1D tensor
    flattened_chunk = chunk.reshape(n_channels * n_samples)

    # get MSB tokens from chunk
    msb_tokens = msb_torch(
        x = flattened_chunk,
        orig_n_bits = BIT_DEPTH,
        n_bits = MSB_N_BITS,
    )
    lsb_tokens = lsb_torch(
        x = flattened_chunk,
        n_bits = LSB_N_BITS,
    )
    assert msb_tokens.dim() == 1, "MSB tokens must be a 1D tensor"
    assert lsb_tokens.dim() == 1, "LSB tokens must be a 1D tensor"
    assert msb_tokens.shape[0] == lsb_tokens.shape[0], "MSB and LSB tokens must have the same number of samples"
    assert msb_tokens.shape[0] == n_channels * n_samples, "MSB and LSB tokens must have the same number of samples as the chunk"
    assert msb_tokens.max() < vocab_size, f"MSB tokens must be within the vocabulary range, but got {msb_tokens.max()}"
    assert msb_tokens.min() >= 0, f"MSB tokens must be within the vocabulary range, but got {msb_tokens.min()}"
    assert lsb_tokens.max() < vocab_size, f"LSB tokens must be within the vocabulary range, but got {lsb_tokens.max()}"
    assert lsb_tokens.min() >= 0, f"LSB tokens must be within the vocabulary range, but got {lsb_tokens.min()}"

    # build sequence incrementally: MSB₁ -> predict LSB₁ -> MSB₂ -> predict LSB₂ -> ...
    sequence = torch.repeat_interleave(
        input = msb_tokens,
        repeats = 2,
        dim = -1,
    ).unsqueeze(dim = 0) # will fill in with LSB tokens as we predict them, currently has shape (1, len(msb_tokens) * 2)
    with torch.no_grad():
        for i in range(len(msb_tokens)):
            current_lsb_index = (i * 2) + 1 # index in sequence for the current LSB token
            outputs = model(sequence[:, :current_lsb_index]) # get model prediction for next token (should be LSB)
            logits = outputs.logits[:, -1, :] # (1, vocab_size) - last position logits
            predicted_lsb_token = torch.argmax(
                input = logits,
                dim = -1,
                keepdim = False,
            ) # greedy decoding: pick the token with highest probability, shape (1, 1)
            predicted_lsb_token = predicted_lsb_token.item() # get the token as a scalar
            assert predicted_lsb_token >= 0 and predicted_lsb_token < vocab_size, f"Predicted LSB token {predicted_lsb_token} is out of range [0, {vocab_size - 1}]"
            sequence[:, current_lsb_index] = predicted_lsb_token # set LSB token in sequence
    predicted_lsb_tokens = sequence.squeeze(dim = 0)[1::2] # get odd indices (LSB tokens), convert back to 1D tensor

    # convert MSB tokens and predicted LSB tokens back into waveform
    predicted_chunk = (msb_tokens << LSB_N_BITS) | predicted_lsb_tokens
    predicted_chunk = predicted_chunk.reshape(n_channels, n_samples)

    # final assertions
    assert predicted_chunk.shape[0] == chunk.shape[0], f"Predicted chunk (n_channels={predicted_chunk.shape[0]}) must have the same number of channels as the original chunk (n_channels={chunk.shape[0]})"
    assert predicted_chunk.shape[1] == chunk.shape[1], f"Predicted chunk (chunk_size={predicted_chunk.shape[1]}) must have the same number of samples as the actual chunk size (chunk_size={chunk.shape[1]})"

    # return predicted chunk
    return predicted_chunk

def perceptual_dithering_experiment(
    model: GPTAudioLightningModule, # device is inferred from the model (whatever device the model is on)
    waveform: torch.Tensor, # audio tensor
    sample_rate: int, # sample rate
    output_dir: str, # output directory
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    """
    Perform perceptual dithering experiment.
    
    Args:
        model: Trained GPTAudioLightningModule
        waveform: Normalized audio tensor (n_channels, n_samples) where n_channels is 1 for mono and 2 for stereo
        sample_rate: Sample rate, for writing audio files
        output_dir: Output directory, for saving original and predicted audio files
        chunk_size: Number of samples to process at a time
        chunk_overlap: Number of samples to overlap between chunks
    """

    # assertions
    assert isinstance(model, GPTAudioLightningModule), "Model must be a trained GPTAudioLightningModule"
    assert isinstance(waveform, torch.Tensor), "Waveform must be a tensor"
    assert waveform.dtype == torch.float32, "Waveform must be a float32 tensor"
    assert waveform.dim() == 2, "Waveform must be a 2D tensor"
    assert waveform.shape[0] == 1 or waveform.shape[0] == 2, "Waveform must be mono (n_channels=1) or stereo (n_channels=2)"
    assert waveform.shape[1] > 0, "Waveform must have at least one sample"
    assert waveform.min() >= -1.0 and waveform.max() <= 1.0, "Waveform must be normalized to the range [-1.0, 1.0]"
    assert exists(output_dir), "Output directory must exist"
    assert isinstance(sample_rate, int), "Sample rate must be an integer"
    assert sample_rate > 0, "Sample rate must be greater than 0"
    assert isinstance(chunk_size, int), "Chunk size must be an integer"
    assert chunk_size > 0, "Chunk size must be greater than 0"
    assert isinstance(chunk_overlap, int), "Chunk overlap must be an integer"
    assert chunk_overlap >= 0, "Chunk overlap must be greater than or equal to 0"
    assert chunk_overlap < chunk_size, "Chunk overlap must be less than chunk size"

    # get information about waveform
    n_channels, n_samples = waveform.shape
    effective_chunk_size = chunk_size - chunk_overlap # chunk size minus overlap, since the overlap is already covered in the previous chunk

    # ensure waveform and model are on the same device
    if waveform.device != model.device:
        waveform = waveform.to(model.device)

    # quantize waveform to 16-bit unsigned PCM
    waveform = quantize_unsigned_pcm_torch(
        x = waveform,
        n_bits = BIT_DEPTH,
        kind = "linear",
    )

    # write original waveform to output directory
    write_audio(
        waveform = waveform, # waveform is 16-bit unsigned PCM, as expected by write_audio
        sample_rate = sample_rate,
        output_path = f"{output_dir}/original.wav",
    )

    # add front padding to waveform to account for initial overlap in first chunk
    waveform = torch.cat(
        (
            torch.zeros((n_channels, chunk_overlap), dtype = waveform.dtype, device = waveform.device), # zero padding for initial overlap
            waveform,
        ),
        dim = -1,
    )
    n_samples_with_initial_overlap = n_samples + chunk_overlap
    assert waveform.shape[1] == n_samples_with_initial_overlap, "Waveform must have the same number of samples as the original waveform plus the chunk overlap"

    # initialize predicted waveform
    predicted_waveform = torch.zeros(
        (n_channels, n_samples),
        dtype = waveform.dtype,
        device = waveform.device,
    )

    # process waveform in chunks
    for i in tqdm(
        iterable = range(0, n_samples_with_initial_overlap, effective_chunk_size),
        desc = "Processing Waveform (in chunks)",
    ):

        # get start and end indices for current chunk
        start_index = i
        end_index = min(start_index + chunk_size, n_samples_with_initial_overlap)

        # get current chunk of waveform
        chunk = waveform[:, start_index:end_index]

        # get predicted chunk
        predicted_chunk = predict_chunk(
            model = model,
            chunk = chunk,
        )

        # update predicted waveform
        predicted_waveform_start_index = i
        predicted_waveform_end_index = min(predicted_waveform_start_index + effective_chunk_size, n_samples)
        predicted_waveform[:, predicted_waveform_start_index:predicted_waveform_end_index] = predicted_chunk[:, chunk_overlap:] # only keep the non-overlapping portion of the predicted chunk    

    # write predicted waveform to output directory
    write_audio(
        waveform = predicted_waveform, # predicted waveform is 16-bit unsigned PCM, as expected by write_audio
        sample_rate = sample_rate,
        output_path = f"{output_dir}/predicted.wav",
    )

    # compare original and predicted waveforms
    waveform = waveform.detach().cpu()
    predicted_waveform = predicted_waveform.detach().cpu()
    print(f"MSE: {torch.mean((waveform - predicted_waveform) ** 2)}")
    print(f"MAE: {torch.mean(torch.abs(waveform - predicted_waveform))}")

    # return nothing
    return

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    # load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(checkpoint_path = args.checkpoint) # load model from checkpoint
    model = model.to(device) # move model to device
    print(f"Model loaded and moved to device: {model.device}")

    # load audio, torchaudio loads waveform as (n_channels, n_samples) where n_channels is 1 for mono and 2 for stereo
    print(f"Loading audio from file: {args.input_audio_path}")
    waveform, sample_rate = load_audio(audio_path = args.input_audio_path) # load audio file
    waveform = waveform.to(device) # move waveform to device

    # determine output directory
    output_dir = f"{args.output_dir}/{basename(args.input_audio_path).split('.')[0]}"
    if not exists(output_dir):
        makedirs(output_dir, exist_ok = True)

    # run perceptual dithering experiment
    perceptual_dithering_experiment(
        model = model,
        waveform = waveform,
        sample_rate = sample_rate,
        output_dir = output_dir,
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
    )

##################################################