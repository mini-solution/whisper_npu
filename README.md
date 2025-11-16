#### 安装conda
#### 安装Ryzen AI
#### 切换环境
```sh
conda activate ryzen-ai-1.6.0
```

#### 安装依赖
```sh
pip install -r requirements.txt
```


#### 运行
At the command prompt, enter
```sh
python -m amd_whisper --npu --wav_path ./test.wav
```