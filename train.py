import torch

import numpy as np
import os
import pyexr
import argparse
import re

from omegaconf import OmegaConf
from dataset import MLTDataset
from models.MLTD import MLTD
from models.MLTDSimple import MLTDSimple
from models.MLTDBlend import MLTDBlend
from loss import MLTDLoss
from metrics import calc_psnr, calc_ssim, calc_fvvdp

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def visualization(output, tag):
    pyexr.write(f'{cfg.visualization_dir}/{cfg.task_name}_{tag}.exr', output)

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
    train_list = os.listdir(cfg.train_dir)
    train_data = [os.path.join(cfg.train_dir, dir) for dir in train_list]  
    valid_list = os.listdir(cfg.valid_dir)
    valid_data = [os.path.join(cfg.valid_dir, dir) for dir in valid_list]

    train_dataset = MLTDataset(cfg=cfg, data_list=train_data, mode='train')
    valid_dataset = MLTDataset(cfg=cfg, data_list=valid_data, mode='valid')
        
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=cfg.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.bs if cfg.num_workers_valid > 0 else 1, 
        shuffle=False, num_workers=cfg.num_workers_valid)
    
    return train_dataloader, valid_dataloader

def eval(cfg, model, loss_criterion, valid_dataloader, epoch):

    print(f"[INFO] Evaluate on Validation Set (Epoch {epoch})")
    torch.cuda.empty_cache()
    model.eval()
    valid_loss, psnr, ssim, fvvdp = [], [], [], []
    
    with torch.no_grad():
        for iter, (inputs, gt) in enumerate(valid_dataloader):
            inputs = inputs.to(cfg.device)
            gt = gt.to(cfg.device)

            preds = torch.clamp(model(inputs), min=0)
            valid_loss.append(loss_criterion(preds, gt).item())
            preds_numpy = preds[0].cpu().detach().numpy().transpose(0, 2, 3, 1)
            gt_numpy = gt[0].cpu().detach().numpy().transpose(0, 2, 3, 1)
            fvvdp.append(calc_fvvdp(preds_numpy, gt_numpy).item())
            
            for i in range(preds_numpy.shape[0]):
                visualization(preds_numpy[i], f'valid_output_{iter}_{i}')
                psnr.append(calc_psnr(preds_numpy[i], gt_numpy[i]))
                ssim.append(calc_ssim(preds_numpy[i], gt_numpy[i]))
                
    return np.mean(valid_loss), np.mean(psnr), np.mean(ssim), np.mean(fvvdp)

def train(cfg):
    
    print(f'[INFO] Train Device: {cfg.device}')
    train_dataloader, valid_dataloader = load_dataset()
    model = load_model(cfg).to(cfg.device)

    writer = SummaryWriter(log_dir=f'runs/{cfg.task_name}')
    make_dirs([cfg.checkpoints_dir, cfg.statistics_dir, cfg.visualization_dir])

    if cfg.use_checkpoint:
        filenames = os.listdir(cfg.checkpoints_dir)
        filenames = [filename for filename in filenames if cfg.task_name in filename]
        if filenames != []:
            filenames.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]), reverse=True)
            checkpoint_filename = filenames[0]
            checkpoint = torch.load(f'{cfg.checkpoints_dir}/{checkpoint_filename}')
            model.load_state_dict(checkpoint['model'])
            start_epoch = int(re.findall(r'\d+', checkpoint_filename)[-1]) + 1
            print(f'[INFO] Load Checkpoint: {checkpoint_filename}')
    else:
        start_epoch = 1
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * (0.8 ** (start_epoch // args.ss)))    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.ss, gamma=0.8)
    loss_criterion = load_loss(cfg)
    
    iter_checked = 0
    for epoch in range(start_epoch, cfg.total_epoch + 1):
        cfg.current_epoch = epoch
        total_loss = 0
        model.train()
        
        for iter, (inputs, gt) in enumerate(train_dataloader):
            inputs = inputs.to(cfg.device)
            gt = gt.to(cfg.device)
            
            preds = model(inputs)
            loss = loss_criterion(preds, gt)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            if norm > cfg.clip_grad:
                print(f'[WARNNING] Gradient clipped from {norm} to {cfg.clip_grad}')
            optimizer.step()
            
            # Log average loss
            if (iter + 1) % cfg.print_interval == 0:
                iter_checked += 1
                avg_loss = total_loss / cfg.print_interval
                total_loss = 0
                print_epoch = f'Epoch: [{epoch}/{cfg.end_epoch}] \t'
                print_iter = f'Iter: {iter + 1} \t'
                print_loss = f'Total Loss: {avg_loss} \t'
                print(f'[INFO] {print_epoch}{print_iter}{print_loss}')
                writer.add_scalar('Train Loss', avg_loss, global_step=iter_checked)
            
        # Evaluate the model on the valid set & Save the model
        if epoch % cfg.save_interval == 0:
            torch.save({'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(),},
                        f'{cfg.checkpoints_dir}/checkpoint_{cfg.task_name}_{epoch}.pth.tar')
        
            valid_loss, psnr, ssim, fvvdp = eval(model, loss_criterion, valid_dataloader, epoch)
            writer.add_scalar('Valid Loss', valid_loss, global_step=epoch)
            writer.add_scalar('PSNR', psnr, global_step=epoch)
            writer.add_scalar('SSIM', ssim, global_step=epoch)
            writer.add_scalar('FVVDP', fvvdp, global_step=epoch)
            
        scheduler.step()
    
    print("[INFO] Train Finished!")

if __name__ == "__main__":

    # Fix random seed
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/MLTD.yaml', help="config path")
    args, extras = parser.parse_known_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.task_name = f'{cfg.task_id}_{cfg.model_name}_{cfg.rendering_type}'
    train(cfg)