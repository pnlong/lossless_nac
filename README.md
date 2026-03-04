# Trilobyte Experiments

This repository is an **experiments and sandbox** for work on **Trilobyte** (lossless neural audio compression). It contains implementations of several lossless codecs, evaluation pipelines, dataset preprocessing, and analysis/plotting scripts.

- **Main repository (Trilobyte):** [https://github.com/ZacharyNovack/lnac](https://github.com/ZacharyNovack/lnac)

---

## Overview

| Area | Description |
|------|-------------|
| **Lossless codecs** | LDAC, LEC, LNAC (DAC/EnCodec-based), plus FLAC as baseline |
| **LMIC** | Language-Modeling-Is-Compression: compressors (FLAC, Llama, Trilobyte) and evaluation on audio chunks |
| **Eval & plots** | FLAC and LMIC evaluation scripts, result CSVs, and plotting/table generation |
| **Data** | MusDB18 preprocessing, Torrent dataset overview, Sashimi/T5 data prep |
| **Experiments** | Perceptual dithering (GPT-2 LSB prediction), LPC vs DAC residual comparison |

---

## Repository layout (brief)

### `lossless_compressors/`

- **`ldac.py`** — Lossless **Descript Audio Codec** (DAC): uses [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec), stores DAC codes + Rice-coded residuals for bit-exact decode.
- **`lec.py`** — Lossless **EnCodec**: uses [Meta EnCodec](https://github.com/facebookresearch/encodec), same idea (codes + residuals).
- **`lnac.py`** — Lossless **custom neural audio codec**: same pipeline with a custom NAC (e.g. custom DAC-style model), optional mid/side decorrelation for stereo.

Tests live in `test_lossless_compressors/` (`test_flac.py`, `test_ldac.py`, `test_lec.py`, `test_lnac.py`).

### `lmic/` — Language Modeling Is Compression (audio)

- **`language_modeling_is_compression/`**
  - **`compress_audio.py`** — Entry point to evaluate a compressor on audio: loads dataset, extracts byte chunks, runs compressor, writes CSV (and optional loss/bpb for arithmetic coders).
  - **`compressors_audio/`** — Compressor implementations: **`trilobyte.py`** (Trilobyte with arithmetic coding), **`llama.py`** (Llama-2), **`flac.py`**, **`png.py`**, **`language_model.py`**; **`compressor.py`** defines the interface and registry.
  - **`data_loaders_audio.py`** — Dataset iterators (MusDB18 mono/stereo, LibriSpeech, LJSpeech, Epidemic, VCTK, Torrent, Birdvox, Beethoven, YouTube Mix, SC09, etc.) and byte extraction for evaluation.
  - **`constants_audio.py`**, **`utils_audio.py`** — Sample rate, bit depth, paths, and audio helpers.
- **`lmic_eval.sh`** — Batch runner for `compress_audio.py` (compressor, chunk size, datasets, machine/batch).
- **`lmic_eval_plot.py`** — Plotting for LMIC evaluation results.
- **`dataset_channels.py`** — Channel/waveform iteration for datasets (e.g. for analysis).

### Root-level evaluation and plotting

- **`flac_eval.py`** — FLAC compression evaluation across many datasets (MusDB18 mono/stereo, LibriSpeech, LJSpeech, Epidemic, VCTK, Torrent 16/24-bit, Birdvox, Beethoven, YouTube Mix, SC09). Outputs CSV with sizes, compression rate, duration, etc.
- **`flac_eval.sh`** — Runs `flac_eval.py` for multiple FLAC levels and dataset/machine combinations.
- **`flac_eval_plot.py`** — Plots FLAC evaluation results (compression rate vs level, per dataset).
- **`flac_eval_results.csv`** — Example/output path for FLAC eval results.

### `figs/` — Tables and comparison plots

- **`trilobyte_table.py`** — Builds a LaTeX table from WandB runs (e.g. t5_lnac): Trilobyte val/bpb and compression rate, with FLAC and LMIC (Byte-to-ASCII) baselines from `flac_eval_results.csv` and `lmic_eval_results.csv`.
- **`nac_compression_rate_comparisons.py`** — Compression rate comparison plots across NAC-style compressors (FLAC, LDAC, LEC, etc.) using best ablation configs.
- **`sashimi_table.py`** — LaTeX table from WandB for Sashimi model configs (BPB, EMA smoothing, compression rate comparison).

### Data preprocessing and overview

- **`preprocess_musdb18.py`** — Converts MusDB18 to WAV and writes `mixes.csv` (target 44.1 kHz, 16-bit; stem IDs, train/val, mono/stereo track sets).
- **`process_musdb18_wav.py`** — Converts MusDB18 preprocessed NPY data to WAV (e.g. 60 s clips, train/val split).
- **`sashimi/`** — Scripts to produce WAV datasets for Sashimi: **`musdb18mono_wav.py`**, **`musdb18stereo_wav.py`** (from preprocessed data to mono/stereo WAVs).
- **`torrent_data_overview.py`** — Summary stats for Torrent subsets (amateur, freeload, pro) at 16- and 24-bit (sample rates, file count, duration), using `flac_eval` dataset classes.

### Experiments and analysis

- **`perceptual_dithering_experiment/`**
  - **`experiment.py`** / **`experiment_parallelized.py`** — Perceptual dithering: 16-bit GPT-2 predicts LSBs from MSBs; reconstructs audio from MSB + predicted LSB.
  - **`train_gpt2.py`** — Training for the GPT-2 audio model used in the dithering experiment.
- **`compare_lpc_dac_residuals_distribution.py`** — Compares residual distributions of LPC vs DAC (and related codecs: FLAC, LDAC, LEC, LNAC); can plot magnitude/log-density.

### `t5/` — T5/T5X

- **`t5/`**, **`t5/t5x/`** — T5X training and model code (e.g. for sequence/audio experiments that feed into Trilobyte or eval tables).

### Other

- **`utils.py`**, **`rice.py`**, **`logging_for_zach`** — Shared utilities, Rice coding, and logging used by the lossless codecs and eval scripts.
- **`dac/`**, **`encodec/`** — Referenced by LDAC/LEC/LNAC; typically external clones of Descript DAC and EnCodec.

---

## Quick start (conceptual)

- **FLAC baseline:** Run `flac_eval.py` (or `flac_eval.sh`) for your datasets → inspect `flac_eval_results.csv`; use `flac_eval_plot.py` for plots.
- **LMIC (Trilobyte / Llama / FLAC):** Run `lmic/language_modeling_is_compression/compress_audio.py` with `--compressor`, `--dataset`, etc.; use `lmic_eval.sh` for batch runs; plot with `lmic_eval_plot.py`.
- **Lossless neural codecs:** Use `lossless_compressors/ldac.py`, `lec.py`, or `lnac.py` (CLI or imports); tests in `test_lossless_compressors/`.
- **Tables for papers:** Use `figs/trilobyte_table.py` (Trilobyte vs FLAC vs LMIC) and `figs/sashimi_table.py` (Sashimi configs), with WandB and the FLAC/LMIC CSV results.

Paths in scripts often assume specific machines (e.g. `yggdrasil`, `pando`) and data roots; adjust constants or CLI args for your environment.

---

## Datasets (referenced in code)

MusDB18 (mono/stereo, mixes/stems), LibriSpeech, LJSpeech, Epidemic Sound, VCTK, Torrent (16/24-bit, pro/amateur/freeload), Birdvox, Beethoven piano sonatas, YouTube Mix, SC09. See `flac_eval.py` and `lmic/.../data_loaders_audio.py` for names and default paths.
