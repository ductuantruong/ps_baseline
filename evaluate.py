import torch,argparse
import numpy as np
from libs.dataloader.data_io import get_dataloader
from models.conformer import FrameConformer 
from libs.tool import *
import os
from libs.startup_config import set_random_seed
import glob
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"
############EVAL CFPRF################

    
def Inference(eval_dlr, conformer_model,  rso,  device):
    print("++++++++++++++++++inference++++++++++++++++++")
    conformer_model.eval()
    with torch.no_grad():
        cp_dict={}
        ver_dict={}
        regver_dict={}
        seg_score_dict={}
        seg_tar_dict={}
        for batch_x, filenames, batch_seg_label, batch_utt_label in tqdm(eval_dlr,ncols=50):
            for fn in filenames:
                cp_dict[fn]=np.array([])
                ver_dict[fn]=np.array([])
                regver_dict[fn]=np.array([])
            if batch_x.shape[1]>340000:
                continue
            batch_x, batch_seg_label = batch_x.to(device), batch_seg_label.to(device)
            batch_size = batch_x.size(0)
            # inference FDN
            utt_logit, seg_score, embs = conformer_model(batch_x) 
            seg_score, seg_target, seg_score_list,seg_target_list = prepare_segcon_target_ali(batch_seg_label, seg_score,rso)
            seg_score_np=np.array([ss.data.cpu().numpy() for ss in seg_score_list])
            # get coarse-gained proposal lists from FDN output scores with the initial confidence score set to 1
            FDN_cp_list=[segscore2proposal(seg_score_np[idx],cp_fun=proposal_func,rso=rso)[1] for idx in range(batch_size)]
            # save dict for FDN output 
            for idx,fn in enumerate(np.array(filenames)):
                cp_dict[fn]=frame2second_proposal(FDN_cp_list[idx], rso=rso)
                seg_score_dict[fn]=seg_score_np[idx][:,1].ravel()
                seg_tar_dict[fn]=seg_target_list[idx].data.cpu().numpy().ravel()
    return seg_score_dict, seg_tar_dict

def seg_performance_excel(seg_score_dict,seg_tar_dict):
    Seer_value,Sacc,f1,pre,rec, Sauc=eval_PFD(seg_score_dict,seg_tar_dict)
    savecontent=pd.DataFrame()
    savecontent['Seer']=["%.2f"%float(Seer_value)]
    savecontent['Sauc']=["%.2f"%float(Sauc)]
    savecontent['P']=["%.2f"%float(pre)]
    savecontent['R']=["%.2f"%float(rec)]
    savecontent['F1']=["%.2f"%float(f1)]
    savecontent['Sacc']=["%.2f"%float(Sacc)]
    return savecontent

if __name__ == '__main__':
    parser = argparse.ArgumentParser('python evaluate_CFPRF.py --dn HAD/PS/LAVDF')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default="./result/") # ['HAD','PS','LAVDF']
    parser.add_argument('--dn', type=str, default="HAD") # ['HAD','PS','LAVDF']
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--seql', type=int, default=1070)
    parser.add_argument('--rso', type=int, default=20)
    parser.add_argument('--glayer', type=int, default=1) 
    parser.add_argument('--eval', action='store_true', default=True) 
    args = parser.parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """loading dataset"""
    test_gt_dict, test_dlr=get_dataloader(batch_size=1,part="test",dn=args.dn,rso=args.rso)
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
    conformer_checkpoint="./checkpoints/%s/%s/FDN/best_model_stage1.pth"%(args.exp_name,args.dn)
    conformer_model.load_state_dict(torch.load(conformer_checkpoint))
    """loading PRN model"""
    """makedir"""
    args.save_path += args.exp_name
    dict_save_path=os.path.join(args.save_path,'dict/%s_'%(args.dn))
    csv_save_path=os.path.join(args.save_path,'pd/%s_'%(args.dn))
    os.makedirs(os.path.dirname(dict_save_path),exist_ok=True)
    os.makedirs(os.path.dirname(csv_save_path),exist_ok=True)
    ###########INFERENCE#############
    if args.eval:
        seg_score_dict, seg_tar_dict = Inference(test_dlr, conformer_model,  args.rso, device)
        writenpy(dict_save_path+'seg_tar_dict.npy',seg_tar_dict)
        writenpy(dict_save_path+'seg_score_dict.npy',seg_score_dict)
    else:
        seg_score_dict=readnpy(dict_save_path+'seg_score_dict.npy',seg_score_dict)
        seg_tar_dict=readnpy(dict_save_path+'seg_tar_dict.npy',seg_tar_dict)
    ###########PFD#############
    savecontent0=seg_performance_excel(seg_score_dict,seg_tar_dict)
    savecontent0.to_csv(csv_save_path+'PFD_results.csv')
    print("PFD",savecontent0)
