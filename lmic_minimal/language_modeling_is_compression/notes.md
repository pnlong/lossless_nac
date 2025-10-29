# Notes

## Language Modeling is Compression

Reading through the Language Modeling is Compression codebase, I make notes of what is important and what needs to be changed for our purposes.

### TODO

#### Adding New Data Sources

[x] Investigate `get_data_generator_fn` in `compress.py`, which can be traced to `GET_DATA_GENERATOR_FN_DICT` in `data_loaders.py`. For every new dataset, we likely need to add a data loader for it to `data_loaders.py`.
[x] Because we are working with audio, create a new file `data_loaders_audio.py`, and only make minimal edits to the original `data_loaders.py` file. `GET_DATA_GENERATOR_FN_DICT` in `data_loaders.py` is automatically joined with `GET_AUDIO_DATA_GENERATOR_FN_DICT` in `data_loaders_audio.py`, so we can access all data generators defined in either file.
[x] Implement `musdb18mono` data generator.
[x] Implement `musdb18stereo` data generator.

#### Variable Bit Depth

[x] Investigate how to make codebase work for 16- and 24-bit depth, and not just 8-bit.
[x] Implement variable bit depth.

#### Llama Integration

[ ] Investigate how `language_model.py` works. Does it work for variable bit depth?
[ ] Integrate Llama-2 models into `language_model.py`. Likely involves messing with `_retrieve_predict_fn`.

#### Stereo Support

[x] Investigate whether codebase just supports mono (I suspect it does).
[ ] Implement stereo handling -- need to ask Zach how to interleave stereo? For all the codebase cares, it just gets bytestreams, so how do we interleave stereo?

#### Helper Scripts

[x] Create `compress_audio.py`, which is the same as `compress.py`, except it supports variable bit depths and new audio datasets (so that we keep the original file intact).

--- 

### Implementation Details

- We aim to modify the original LMIC codebase files as little as possible. Most of our updated functionalities have been added in new files with the `_audio` suffix: `compress_audio.py`, `constants_audio.py`, `data_loaders_audio.py`, `compressors_audio`.