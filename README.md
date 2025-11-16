#### 安装conda
#### 安装Ryzen AI
[ryzen ai安装](https://ryzenai.docs.amd.com/en/latest/inst.html)

#### 切换环境
```sh
conda activate ryzen-ai-1.6.0
```

#### 安装依赖
```sh
pip install -r requirements.txt
```


#### 运行
- npu 在npu上运行
- wav_path 音频文件

```sh
python -m amd_whisper --npu --wav_path ./test.wav
```
