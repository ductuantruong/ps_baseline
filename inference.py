import torch,argparse
import torch.nn.functional as F
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
    output_text = Inference(args.data_path, conformer_model, args.score_type_ouput, args.consolidate_output, device, args.base_rso, args.output_rso)
    txt_file_name= args.data_path.split('/')[-1].split('.')[0] + '.txt'
    with open(os.path.join(args.save_path, txt_file_name), 'w+') as fh:
        fh.write('\n'.join(output_text) + '\n')
    print("Output file is saved at {}".format(os.path.join(args.save_path, txt_file_name)))
    fh.close()

