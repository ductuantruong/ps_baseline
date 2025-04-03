## 1. Env setup
It is recommended that you install Python 3.8 or higher. We followed the installation setup in this project [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing), which is presented as follows:

```bash
conda create -n ps_baseline python=3.8.0 numpy=1.23.5
conda activate ps_baseline
pip install pip==22.3.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install soundfile==0.11.0 librosa==0.9.1 omegaconf==2.0.6
pip install conformer
pip install scikit-learn pandas==1.3.5
--------------install fairseq for XLSR--------------
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
cd SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
cd ../..
```


## Pretrained Model
The pretrained model XLSR can be found at [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

We have uploaded pretrained models of our experiments. You can download pretrained models from [link](https://entuedu-my.sharepoint.com/:u:/g/personal/zhlim_staff_main_ntu_edu_sg/Ectv8XEBK1BAgZJy1fXcziQBmMDLoML7UxYflIKQIKNJpA?e=EzkzhA). 

### 3. Run Inference
Inference the pretrained model with single wav file:
```
python inference.py --data_path path/to/wav_file  --ckpt_path path/to/model_ckpt
```

## Acknowledge
Our work is built upon the [CFPRF](https://github.com/ItzJuny/CFPRF) codebase.
