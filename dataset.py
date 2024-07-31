import torch
import numpy as np
import random
import cv2
import h5py

def luminance(img):
    lumi = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return lumi[..., np.newaxis]

def crop_data(data, crop_width, crop_height):
    _, height, width, _ = data['gt'].shape
    x_start = (width - crop_width) // 2
    y_start = (height - crop_height) // 2
    
    for key in data.keys():
        data[key] = data[key][:, y_start:y_start+crop_height, x_start:x_start+crop_width, :]
        
    return data

def augment_data(data, crop_width, crop_height, patch_point=None):

    flip_flag_x = np.random.rand() < 0.5
    flip_flag_y = np.random.rand() < 0.5
    k = random.randint(0, 3)

    _, height, width, _ = data['gt'].shape

    # Use patch
    if patch_point is not None:
        x_start = patch_point[1]
        y_start = patch_point[0]
    else:
        x_start = np.random.randint(0, width - crop_width + 1)
        y_start = np.random.randint(0, height - crop_height + 1)
        
    for key in data.keys():  
        # Crop
        data[key] = data[key][:, y_start:y_start+crop_width, x_start:x_start+crop_height, :]
    
        # Flip
        if flip_flag_x:
            data[key] = np.flip(data[key], axis=2)
            if key == 'aux':
                data[key][..., -3] *= -1        
        if flip_flag_y:
            data[key] = np.flip(data[key], axis=1)
            if key == 'aux':
                data[key][..., -2] *= -1
        
        # Rotate
        data[key] = np.rot90(data[key], k=k, axes=(1, 2))
        if key == 'aux':
            for _ in range(k):
                tmp_motion = data[key][..., -3].copy()
                data[key][..., -3] = data[key][..., -2]
                data[key][..., -2] = -tmp_motion
        
    return data

# Constant weight per-pixel blending, not used in our model
def temporal_accumulation(data):

    t, height, width, _ = data['input'].shape
    prev_sample_num = np.ones((height, width, 1))
    
    for i in range(1, t):
    
        motion_one = data['aux'][i, :, :, -3:]
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        src_x = x - motion_one[..., 0]
        src_y = y - motion_one[..., 1]
        prev_warp = cv2.remap(data['input'][i - 1], src_x.astype(np.float32), src_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
        prev_sample_num = cv2.remap(prev_sample_num, src_x.astype(np.float32), src_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]
        blend_alpha = np.clip(1 / (prev_sample_num + 1), a_min=0.2, a_max=1.0)
        data['input'][i] = np.where(motion_one[..., 2:] > 0.5, (1 - blend_alpha) * prev_warp + blend_alpha * data['input'][i], data['input'][i])
        prev_sample_num = np.where(motion_one[..., 2:] > 0.5, prev_sample_num + 1, 1)
    
    return data
            
def normalization_preprocess(inputs, input_dim):
    
    # Normalize depth to [-1, 1]
    inputs[:, input_dim] -= torch.min(inputs[:, input_dim])
    inputs[:, input_dim] = inputs[:, input_dim] / (torch.max(inputs[:, input_dim]) * 0.5) - 1.0
    
    # Normalize Albedo to [-1, 1]
    inputs[:, input_dim+1:input_dim+4] = inputs[:, input_dim+1:input_dim+4] * 2 - 1
    
    # Normalize position to [-1, 1]
    inputs[:, input_dim+7:input_dim+10] -= torch.min(inputs[:, input_dim+7:input_dim+10])
    inputs[:, input_dim+7:input_dim+10] = inputs[:, input_dim+7:input_dim+10] / (torch.max(inputs[:, input_dim+7:input_dim+10]) * 0.5) - 1.0
        
    # Log tonemapping input radiance
    inputs[:, :input_dim] = torch.clamp(inputs[:, :input_dim], min=0)
    inputs[:, :input_dim] = torch.log(1 + inputs[:, :input_dim])

    # Normalize input radiance to [-1, 1]
    inputs[:, :input_dim] /= torch.max(inputs[:, :input_dim]) * 0.5
    inputs[:, :input_dim] -= 1.0
    inputs[~torch.isfinite(inputs)] = 0.0
    
    return inputs
    
class MLTDataset(torch.utils.data.Dataset):
    
    def __init__(self, cfg, data_list, mode='train'):
        task_tokens = cfg.task_name.split('_')
        self.rendering_type = task_tokens[-1] 
        self.cfg = cfg
        self.mode = mode
        self.data_list = data_list if mode == 'train' else sorted(data_list)
        
        self.temporal_acc_model_list = [] 
        self.extra_input_model_list = ['MLTD']
        self.normalization_model_list = ['MLTD', 'MLTDSimple']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        input_path = self.data_list[idx]
        ref_path = '_'.join(input_path.split('_')[:-1]) + '_32spp.h5' if self.mode == 'train' else input_path
        patch_point = None
        data = {}
        
        with h5py.File(ref_path, 'r') as f:
            data['gt'] = np.array(f['gt'])
            data['aux'] = np.array(f['aux'])
            if self.mode == 'train' and self.cfg.use_patch:
                patches = np.array(f['patches'])
                patch_point = patches[self.cfg.current_epoch % patches.shape[0]]
        
        with h5py.File(input_path, 'r') as f:    
            if self.cfg.model_name in self.extra_input_model_list:
                data['input'] = np.array(f[f'{self.rendering_type}_extra'])
                input_dim = 12 if self.rendering_type == 'mlt' else 6
            else:
                data['input'] = np.array(f[self.rendering_type])
                input_dim = 3
        
        if self.mode == 'train':
            data = augment_data(data, self.cfg.width, self.cfg.height, patch_point)
        if self.cfg.model_name in self.temporal_acc_model_list:
            data = temporal_accumulation(data)

        input = np.concatenate([data['input'], data['aux'], data['input']], axis=-1)
        inputs = torch.from_numpy(input.copy()).permute(0, 3, 1, 2).float().clamp(max=1e12)
        gt = torch.from_numpy(data['gt'].copy()).permute(0, 3, 1, 2).float().clamp(max=1e12)
        
        if self.cfg.model_name in self.normalization_model_list:
            if self.mode != 'test':
                inputs = normalization_preprocess(inputs, input_dim)
            elif not self.cfg.test_input:
                inputs = normalization_preprocess(inputs, input_dim)
    
        return inputs, gt