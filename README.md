# AMD Python NPU Whisper-Small Script
This directory contains a sample python script to run the  Whisper-Base-Small model on CPU and NPU. A single sample of the LibriSpeech dataset has been taken and converted from flac to wav (test.wav). Both encoder and decoder models have been converted to ONNX using static tensor shapes. The sequence length is 448 tokens.

The paths to the models, input file, and other needed constants are defined in upper case at the top of the file. The defaults are set to run the existing file `test.wav`

The sample text produced by the script should correspond to the first line of the transcribe.txt file

**Note**: The models need to be compiled and cached. This will happen the first time they are run. It may take several minutes for the compilation to complete. Compilation only needs to be done once.

## Instructions for running Whisper-Small

Open Developer Command Prompt for VS 2022
Activate the Ryzen AI 1.5 Conda env at the command prompt
```sh
conda activate ryzen-ai-1.5.0
```
If you don't have the requirements listed in requirements.txt , run
```sh
pip install -r requirements.txt
```
### To run on CPU
At the command prompt, enter
```sh
python -m amd_whisper 
```

### To run on NPU
At the command prompt, enter
```sh
python -m amd_whisper --npu --wav_path ./test.wav
```