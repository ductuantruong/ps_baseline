## 1. Setup


It is recommended that you install Python 3.8 or higher. We followed the installation setup in this project [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing), which is presented as follows:


```bash
conda create -n SSL python=3.8.0 numpy=1.23.5
conda activate SSL
pip install pip==24.0
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install soundfile==0.11.0 librosa==0.9.1
pip install pandas scikit-learn
pip install conformer
--------------install fairseq for XLSR--------------
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
```


## 2. Change the data path of PartialSpoof (PS) corpus 
Change the PS corpus path in [this line](https://github.com/ductuantruong/CFPRF/blob/ac736a0d191d00afbec786daa402706363d7402a/libs/dataloader/data_io.py#L252) to the corresponding path on your local machine (e.g. /data/PartialSpoof/database). The PS corpus folder should have this items:
```
/data/PartialSpoof/database
├── train/
├── dev/
├── eval/
├── segment_labels/
├── protocols/
├── ...
```

### 3. Run 🚀
Training & evaluating checkpoints for PS the dataset:
```
bash run.sh
```


## Citation
Kindly cite our work if you find it useful.


```
@article{wu2024cfprf,
  title={Coarse-to-Fine Proposal Refinement Framework for Audio Temporal Forgery Detection and Localization},
  author={Wu, Junyan and Lu, Wei and Luo, Xiangyang and Yang, Rui and Wang, Qian and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2407.16554},
  year={2024},
  doi={10.1145/3664647.3680585},
}
```
