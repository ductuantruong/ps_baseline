import torch,argparse
import torch.nn.functional as F
import numpy as np
from libs.dataloader.data_io import get_dataloader
from models.conformer import FrameConformer 
from libs.tool import *
import os
from libs.startup_config import set_random_seed
import warnings
import soundfile as sf
import torch.nn.functional as F
import json
import math

warnings.filterwarnings("ignore")
__author__ = "Junyan Wu, Tuan, Zin"
__email__ = "wujy298@mail2.sysu.edu.cn"
############EVAL CFPRF################
import soundfile as sf
import torchaudio
    
def print_spoof_timestamps(pred, rso=20):
    start = None
    txt_output = []
    for i, value in enumerate(pred):
        if value == 0:
            if start is None:
                start = i * rso
        else:
            if start is not None:
                end = i * rso
                txt_output.append(f"{start}ms - {end}ms")
                start = None

    # Check if the list ends with 0
    if start is not None:
        end = len(pred) * rso
        txt_output.append(f"{start}ms - {end}ms")
    return txt_output

def Inference_mod(file_path, conformer_model, score_type_ouput, consolidate_output, device, base_rso, output_rso):
    print("++++++++++++++++++inference++++++++++++++++++")
    conformer_model.eval()
    text_list = []
    filename = os.path.basename(file_path)

    with torch.no_grad():
        file_type = file_path.split('.')[-1]
        if file_type != 'mp3':
            wave, sr = sf.read(file_path)
            wave = np.expand_dims(wave, axis=0)  # [1, T]
            wave = torch.Tensor(wave)
        else:
            wave, sr = torchaudio.load(file_path)

        if sr != 16000:
            wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wave)

        wave = wave.to(device)
        total_samples = wave.shape[1]
        sample_rate = 16000
        chunk_dur= 4
        #chunk_samples = sample_rate * 4  # 4s chunk
        chunk_samples = sample_rate * chunk_dur
        pred_all = []
        last_remain_chunk=0

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = wave[:, start:end]

            # Zero pad if less than 4s, only happened once in the last chunk remaining chunk
            if chunk.shape[1] < chunk_samples:
                last_remain_chunk=chunk.shape[1]
                padding = chunk_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))

            _, seg_score, _ = conformer_model(chunk)

            seg_score = torch.squeeze(seg_score)

            if score_type_ouput:
                pred = (seg_score[:, 1] > args.threshold).long().tolist()
            else:
                pred = F.softmax(seg_score, dim=-1)[:, 0]
                pred = F.avg_pool1d(
                    pred.unsqueeze(0).unsqueeze(0),
                    kernel_size=output_rso // base_rso,
                    stride=output_rso // base_rso,
                    ceil_mode=True
                )
                pred = pred.squeeze().tolist()

            pred_all.extend(pred)

        audio_duration = round(total_samples / sample_rate, 2)
        text_list.append("Duration of audio: {}s".format(audio_duration))
        
        # for the last non-4s segment, remove the padded prediction
        if (last_remain_chunk/16000) != 0:

            # calculate the negative index and remove padded pred
            # get the actual audio duration in the remaining chunk
            # if remaining duration is between 1.1 to 1.9s, get 2 prediction
            
            # - (4 - 2) = -2, 
            # chunk size =4s, remaining audio sec=2s, 
            # remove last 2 prob from pred_all
            pad_out=-(chunk_dur-math.ceil(last_remain_chunk/16000))
            # only for the negative index. audio of 3.1-3.4s means negative index=0. i.e. keep all 4 prod.
            if pad_out!=0: 
                pred_all=pred_all[0:pad_out]
        text_list.append("Percentage Spoof audio: {}%".format(round((1 - sum(pred_all) / len(pred_all)) * 100, 2)))

        if sum(pred_all) != len(pred_all):
            text_list.append("Spoof audio segments:")
            seg_text = print_spoof_timestamps(pred_all)
            text_list += seg_text

        text_list.append("Frame-level (1 frame is {} ms) prediction output (1 means bonafide frame, 0 means spoof frame):".format(args.output_rso))
        text_list.append(str(pred_all))

        # Save to JSON
        if consolidate_output:
            if os.path.exists(consolidate_output):
                with open(consolidate_output, 'r') as jf:
                    try:
                        output_dict = json.load(jf)
                    except json.JSONDecodeError:
                        output_dict = {}
            else:
                output_dict = {}

            output_dict[filename] = pred_all
            with open(consolidate_output, 'w') as aw:
                json.dump(output_dict, aw, indent=4)

    return text_list

def Inference(file_path, conformer_model, score_type_ouput, consolidate_output, device, base_rso, output_rso):
    print("++++++++++++++++++inference++++++++++++++++++")
    conformer_model.eval()
    text_list = []
    filename = os.path.basename(file_path)
    with torch.no_grad():
        file_type = file_path.split('.')[-1]
        if file_type != 'mp3':
            wave, sr = sf.read(file_path)
            batch_x = torch.Tensor(np.expand_dims(wave, axis=0))
        else:
            wave, sr = torchaudio.load(file_path)
            batch_x = wave
        
        if sr != 16000:
            batch_x = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=16000)(batch_x)

        batch_x = batch_x.to(device)
        batch_size = batch_x.size(0)
        # inference FDN
        _, seg_score, _ = conformer_model(batch_x) 
        seg_score = torch.squeeze(seg_score)
        if score_type_ouput:
            pred = (seg_score[:, 1] > args.threshold).long().tolist()
        else:
            pred = F.softmax(seg_score, dim=-1)[:, 0]
            pred = F.avg_pool1d(pred.unsqueeze(0).unsqueeze(0), kernel_size=output_rso//base_rso, stride=output_rso//base_rso, ceil_mode=True)
            pred = pred.squeeze().tolist()
        text_list.append("Duration of audio: {}s".format(round(wave.shape[-1]/sr, 2)))
        text_list.append("Percentage Spoof audio: {}%".format(round((1-sum(pred)/len(pred))*100, 2)))
        if sum(pred) != len(pred):
            text_list.append("Spoof audio segments:")
            seg_text = print_spoof_timestamps(pred)
            text_list += seg_text
        text_list.append("Frame-level (1 frame is {} ms) prediction output (1 means bonafide frame, 0 means spoof frame):".format(args.output_rso))
        text_list.append(str(pred))
        
        # saving to the JSON output
        if consolidate_output:
        # read existing json file
            if os.path.exists(consolidate_output):
                with open(consolidate_output, 'r') as jf:
                    try:
                        output_dict = json.load(jf)
                    except json.JSONDecodeError:
                        output_dict = {}
        		
        	# write a new json file
            else:
                output_dict={}
            output_dict[filename] = pred
            with open(consolidate_output, 'w') as aw:
	            json.dump(output_dict, aw, indent=4)
    return text_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python evaluate.py')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default="result/") # ['HAD','PS','LAVDF']
    parser.add_argument('--dn', type=str, default="PS") # ['HAD','PS','LAVDF']
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--consolidate_output', type=str, default="./result/consolidate_output.json")
    parser.add_argument('--score_type_ouput', action='store_true', default=False)
    parser.add_argument('--seql', type=int, default=1070)
    parser.add_argument('--output_rso', type=int, default=1000)
    parser.add_argument('--base_rso', type=int, default=20)
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
    #dict_save_path=os.path.join(args.save_path,'dict/%s_'%(args.dn))
    #csv_save_path=os.path.join(args.save_path,'pd/%s_'%(args.dn))
    #os.makedirs(os.path.dirname(dict_save_path),exist_ok=True)
    #os.makedirs(os.path.dirname(csv_save_path),exist_ok=True)

    os.makedirs(args.save_path, exist_ok=True)
    ###########INFERENCE#############
    print("Processing:", args.data_path)
    output_text = Inference_mod(args.data_path, conformer_model, args.score_type_ouput, args.consolidate_output, device, args.base_rso, args.output_rso)
    txt_file_name= args.data_path.split('/')[-1].split('.')[0] + '.txt'
    with open(os.path.join(args.save_path, txt_file_name), 'w+') as fh:
        fh.write('\n'.join(output_text) + '\n')
    print("Output file is saved at {}".format(os.path.join(args.save_path, txt_file_name)))
    fh.close()

