import torch

import numpy as np
import json
import os
import pyexr
import argparse

from omegaconf import OmegaConf
from dataset import MLTDataset
from models.MLTD import MLTD
from models.MLTDSimple import MLTDSimple
from models.MLTDBlend import MLTDBlend
from loss import MLTDLoss
from metrics import calc_psnr, calc_ssim, calc_fvvdp
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualization(output, tag):
    pyexr.write(f'{cfg.visualization_dir}/{cfg.task_name}_{tag}.exr', output)

def visualization_input(input, gt, test_name, i):
    pyexr.write(f'{cfg.visualization_dir}/{cfg.task_name}_test_input_{test_name}_{i}.exr', input)
    pyexr.write(f'{cfg.visualization_dir}/{cfg.task_name}_test_gt_{test_name}_{i}.exr', gt)

def make_dirs(mkdir_list):
    for dir in mkdir_list:
        os.makedirs(dir, exist_ok=True)

def load_loss(cfg):
    loss_dict = {"MLTD": MLTDLoss, "MLTDSimple": MLTDLoss, "MLTDBlend": MLTDLoss}
    return loss_dict[cfg.loss_name](cfg)

def load_model(cfg):
    model_dict = {"MLTD": MLTD, "MLTDSimple": MLTDSimple, "MLTDBlend": MLTDBlend}
    return model_dict[cfg.model_name](cfg)

def load_dataset(cfg):

    test_dir = cfg.test_dir
    test_names = sorted(os.listdir(test_dir))
    test_data_dirs = []

    for test_name in test_names:
        test_data_dirs.append(os.path.join(test_dir, test_name))

    test_dataset = MLTDataset(cfg=cfg, data_list=test_data_dirs, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    return test_names, test_dataloader

def test_metrics(cfg, test_name, preds, gt):
    
    psnr = []
    ssim = []
    
    preds_numpy = preds.cpu().detach().numpy().transpose(0, 2, 3, 1)
    gt_numpy = gt.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fvvdp = calc_fvvdp(preds_numpy, gt_numpy).item()
    
    for i in range(preds_numpy.shape[0]):
        psnr.append(calc_psnr(preds_numpy[i], gt_numpy[i]))
        ssim.append(calc_ssim(preds_numpy[i], gt_numpy[i]))
        if cfg.visualization:
            if cfg.test_input:
                visualization_input(preds_numpy[i], gt_numpy[i], test_name, i)
            else:
                visualization(preds_numpy[i], f'test_output_{cfg.checkpoint_name}_{test_name}_{i}')
    
    print(f'Test {test_name} PSNR: {np.mean(psnr)} SSIM: {np.mean(ssim)} FVVDP: {fvvdp}')
    return psnr, ssim, fvvdp

def test(cfg):

    print(f'[INFO] Test Device: {device}')
    test_names, test_dataloader = load_dataset(cfg)
    model = load_model(cfg).to(device)
    make_dirs([cfg.checkpoints_dir, cfg.statistics_dir, cfg.visualization_dir])

    model.eval()
    with torch.no_grad():
        if not cfg.test_input:
            if not os.path.exists(f'{cfg.checkpoints_dir}/{cfg.checkpoint_name}'):
                print(f'[ERROR] Checkpoint {cfg.checkpoint_name} does not exist.')
                return
            else:
                print(f'[INFO] Test Task {cfg.task_name}')
                checkpoint = torch.load(f'{cfg.checkpoints_dir}/{cfg.checkpoint_name}')
                model.load_state_dict(checkpoint['model'])
        
        preds_concat, gt_concat = [], []
        psnrs, ssims, fvvdps = [], [], []

        for iter, (inputs, gt) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            gt = gt.to(device)

            preds = model(inputs).clamp(min=0) if not cfg.test_input else inputs[:, :, :3]
            if cfg.split: # For CPU memory issue
                preds_concat.append(preds[0])
                gt_concat.append(gt[0])
            else:
                psnr, ssim, fvvdp = test_metrics(cfg, test_names[iter], preds[0], gt[0])
                psnrs += psnr
                ssims += ssim
                fvvdps.append(fvvdp)
        
        if cfg.split:
            psnrs, ssims, fvvdp = test_metrics(cfg, test_names[0], torch.cat(preds_concat, dim=0), torch.cat(gt_concat, dim=0))
            fvvdps.append(fvvdp)
        
        psnrs = np.mean(psnrs)
        ssims = np.mean(ssims)
        fvvdps = np.mean(fvvdps)
            
        metrics = {'psnr': psnrs, 'ssim': ssims, 'fvvdp': fvvdps}
        
        if cfg.test_input:
            with open(f"{cfg.statistics_dir}/{cfg.task_name}_{cfg.test_dir.split('/')[-1]}_input_metrics.json", "w") as f:
                json.dump(metrics, f)
        else:
            with open(f"{cfg.statistics_dir}/{cfg.task_name}_{cfg.test_dir.split('/')[-1]}_{cfg.checkpoint_name}_metrics.json", "w") as f:
                json.dump(metrics, f)
            
if __name__ == "__main__":

    # Fix random seed
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/MLTD.yaml', help="config path")
    args, extras = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    cfg.task_name = f'{cfg.task_id}_{cfg.model_name}_{cfg.rendering_type}'
    test(cfg)