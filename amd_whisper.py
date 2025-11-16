import os
import numpy as np
import torchaudio
import onnxruntime as ort
from transformers import WhisperProcessor
import argparse
from pathlib import Path
import sys

ENCODER_ONNX  = "onnx_static/whisper_encoder.onnx"
DECODER_ONNX  = "onnx_static/whisper_decoder.onnx"
SAMPLE_RATE   = 16000
N_MELS        = 80
ENC_FRAMES    = 3000
DEC_SEQ_LEN   = 448

def main(args):
    
    path = Path(args.wav_path)   # 假设 args.path 是你的参数
    if not path.exists():
        print(f"音频文件不存在: {path}")
        sys.exit(1)     # 退出程序（非 0 表示错误）

    waveform, sr = torchaudio.load(args.wav_path)
    # resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        # mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform[0].cpu().numpy()

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    feats = processor.feature_extractor(
        raw_speech=audio,
        sampling_rate=sr,
        return_tensors="np",
        padding=False          
    )

    input_features = feats["input_features"] 

    n_frames = input_features.shape[-1]
    if n_frames < 3000:
        pad_width = ((0,0),(0,0),(0,3000-n_frames))
        input_features = np.pad(input_features, pad_width, mode="constant", constant_values=0)
    else:
        input_features = input_features[..., :3000]

    cache_dir = os.path.abspath("cache")
    if args.npu:
        print(f'Running on NPU\n')
        providers=["VitisAIExecutionProvider"]
        enc_provider_options=[{
                "config_file": "vitisai_config.json",
                "cache_dir": cache_dir,
                "cache_key": "enc",
                "enable_cache_file_io_in_mem":0,
            }]
        dec_provider_options=[{
                "config_file": "vitisai_config.json",
                "cache_dir": cache_dir,
                "cache_key": "dec",
                "enable_cache_file_io_in_mem":0,
            }]
    else :
        print(f'Running on CPU\n')
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]

    print(ort.get_available_providers())

    enc_sess_opts = ort.SessionOptions()
    enc_sess = ort.InferenceSession(
        ENCODER_ONNX, 
        sess_options=enc_sess_opts,
        providers=providers,
        provider_options=enc_provider_options
    )

    dec_sess_opts = ort.SessionOptions()
    dec_sess = ort.InferenceSession(
            DECODER_ONNX,
            providers=providers,
            sess_options=dec_sess_opts,
            provider_options=dec_provider_options
        )

   
    encoder_output = enc_sess.run(None, {"input_features": input_features})[0]
    
    tokenizer   = processor.tokenizer
    start_token = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    eos_token   = tokenizer.eos_token_id
    pad_token   = tokenizer.pad_token_id


    input_ids   = [start_token]
    generated   = []

    
    for _ in range(DEC_SEQ_LEN):
        # pad input_ids to length DEC_SEQ_LEN
        padded = input_ids + [pad_token] * (DEC_SEQ_LEN - len(input_ids))
        ids_np = np.array([padded], dtype=np.int64)  # shape: (1, 448)

        # run ONNX decoder
        logits = dec_sess.run(
            None,
            {
                "input_ids": ids_np,
                "encoder_hidden_states": encoder_output
            }
        )[0]  # shape: (1, 448, vocab_size)

    
        pos = len(input_ids) - 1
        next_token = int(np.argmax(logits[0, pos]))

        
        if next_token == eos_token:
            break

        input_ids.append(next_token)
        generated.append(next_token)
    # print(f' TPS { 448/decoder_time} , with fixed length of 448 tokens per sequence')
    transcript = tokenizer.decode(generated, skip_special_tokens=True)
    print(" Transcription:\n", transcript)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whixper NPU")
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--wav_path",type=str,required=True, help="音频文件路径")
    args = parser.parse_args()
    main(args)