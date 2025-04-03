import torch,argparse
import numpy as np
from libs.dataloader.data_io import get_dataloader
from models.conformer import FrameConformer 
from libs.tool import *
import os
from libs.startup_config import set_random_seed
import warnings
warnings.filterwarnings("ignore")
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"
############EVAL CFPRF################
import soundfile as sf
    
def print_spoof_timestamps(pred, rso=20):
    start = None
    for i, value in enumerate(pred):
        if value == 0:
            if start is None:
                start = i * rso
        else:
            if start is not None:
                end = i * rso
                print(f"{start}ms - {end}ms")
                start = None

    # Check if the list ends with 0
    if start is not None:
        end = len(pred) * rso
        print(f"{start}ms - {end}ms")


def Inference(file_path, conformer_model,  device):
    print("++++++++++++++++++inference++++++++++++++++++")
    conformer_model.eval()
    with torch.no_grad():
        wave, sr = sf.read(file_path)
        batch_x = torch.Tensor(np.expand_dims(wave, axis=0))

        batch_x = batch_x.to(device)
        batch_size = batch_x.size(0)
        # inference FDN
        _, seg_score, _ = conformer_model(batch_x) 
        seg_score = torch.squeeze(seg_score)
        pred = (seg_score[:, 1] > args.threshold).long().tolist()
        print("Frame-level prediction output of {}:".format(args.data_path))
        print(pred)
        print("Duration of audio: {}s".format(round(wave.shape[-1]/sr, 2)))
        print("Percentage Spoof audio: {}%".format(round((1-sum(pred)/len(pred))*100, 2)))
        if sum(pred) != len(pred):
            print("Spoof audio segments:")
            print_spoof_timestamps(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python evaluate.py')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default="result/") # ['HAD','PS','LAVDF']
    parser.add_argument('--dn', type=str, default="PS") # ['HAD','PS','LAVDF']
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--seql', type=int, default=1070)
    parser.add_argument('--rso', type=int, default=20)
    parser.add_argument('--glayer', type=int, default=1) 
    parser.add_argument('--eval', action='store_true', default=True) 
    args = parser.parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """loading FDN model"""
    conformer_model = FrameConformer(seq_len=args.seql, gmlp_layers=args.glayer).to(device)
    """ 
    ./checkpoints
    ├── 1FDN_HAD.pth
    ├── 1FDN_LAVDF.pth
    ├── 1FDN_PS.pth
    ├── 2PRN_HAD.pth
    ├── 2PRN_LAVDF.pth
    ├── 2PRN_PS.pth
    """
    conformer_checkpoint="%s"%(args.ckpt_path)
    conformer_model.load_state_dict(torch.load(conformer_checkpoint, map_location=device))
    """makedir"""
    dict_save_path=os.path.join(args.save_path,'dict/%s_'%(args.dn))
    csv_save_path=os.path.join(args.save_path,'pd/%s_'%(args.dn))
    os.makedirs(os.path.dirname(dict_save_path),exist_ok=True)
    os.makedirs(os.path.dirname(csv_save_path),exist_ok=True)
    ###########INFERENCE#############
    print("Frame-level prediction output (1 means bonafide frame, 0 means spoof frame):")
    seg_score = Inference(args.data_path, conformer_model, device)
    ###########DECISION#############
