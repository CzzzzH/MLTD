import torch
import torch.nn as nn
import numpy as np
import numbers
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from . import halide_ops as ops

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def warp_tensor(img, motion_vectors):
    
    actual_motion = motion_vectors[:, :2, :, :]
    
    # Dimensions of motion_vectors
    _, _, mv_height, mv_width = motion_vectors.shape

    # Generate grid based on motion_vectors dimensions
    y, x = torch.meshgrid([torch.linspace(-1, 1, mv_height, device=device), torch.linspace(-1, 1, mv_width, device=device)])
    grid = torch.stack((x, y), dim=-1).float().unsqueeze(0)

    width_factor = torch.tensor([[[[mv_width / 2]]]], device=device).expand_as(actual_motion[:, :1, :, :])
    height_factor = torch.tensor([[[[mv_height / 2]]]], device=device).expand_as(actual_motion[:, 1:2, :, :])
    motion_norm = torch.cat((actual_motion[:, :1, :, :] / width_factor, actual_motion[:, 1:2, :, :] / height_factor), 1).permute(0, 2, 3, 1)
    
    warped_grid = grid - motion_norm
    warped_img_resized = F.grid_sample(img, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Resize warped image back to original img dimensions
    _, _, img_height, img_width = img.shape
    warped_img = F.interpolate(warped_img_resized, size=(img_height, img_width), mode='bilinear', align_corners=True)

    # Compute the mask using motion_vectors' dimensions and then resize to match warped_img dimensions
    mask_resized = motion_vectors[:, 2:3, :, :]
    mask = F.interpolate(mask_resized, size=(img_height, img_width), mode='nearest')
    
    return warped_img * mask

class KernelWeighting(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, data, weights):
        
        bs, c, h, w = data.shape
        
        output = data.new()
        sum_w = data.new()
        output.resize_as_(data)
        sum_w.resize_(bs, h, w)
        
        if device == 'cpu': ops.kernel_weighting_cpu(data, weights, output, sum_w)
        else: ops.kernel_weighting_cuda(data, weights, output, sum_w)

        ctx.save_for_backward(data, weights, sum_w)
        return output, sum_w

    @staticmethod
    def backward(ctx, d_output, d_sum_w):
        
        data, weights, sum_w = ctx.saved_tensors
        
        d_data = data.new()
        d_weights = weights.new()
        d_data.resize_as_(data)
        d_weights.resize_as_(weights)
        
        if device == 'cpu': ops.kernel_weighting_grad_cpu(data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)
        else: ops.kernel_weighting_grad_cuda(data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)

        return d_data, d_weights

class KernelApply(nn.Module):
    
    def __init__(self, softmax=True):
        super(KernelApply, self).__init__()
        self.softmax = softmax

    def forward(self, data, kernels):
        
        bs, k2, h, w = kernels.shape
        k = int(np.sqrt(k2))
        kernels = kernels.view(bs, k, k, h, w)
        if self.softmax:
            kernels = kernels.view(bs, k * k, h, w)
            kernels = F.softmax(kernels, dim=1)
            kernels = kernels.view(bs, k, k, h, w)

        output, sum_w = KernelWeighting.apply(data, kernels)
        sum_w = sum_w.unsqueeze(1)
        return output

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class RecurrentAttention(nn.Module):
    
    def __init__(self, dim, num_heads, bias):
        
        super(RecurrentAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_mut = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.q_mut = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_mut = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.q_mut_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_mut_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, x_hidden):
        
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q_mut = self.q_mut_dwconv(self.q_mut(x))
        kv_mut = self.kv_mut_dwconv(self.kv_mut(x_hidden))
        k_mut, v_mut = kv_mut.chunk(2, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_mut = rearrange(q_mut, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_mut = rearrange(k_mut, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_mut = rearrange(v_mut, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        q_mut = torch.nn.functional.normalize(q_mut, dim=-1)
        k_mut = torch.nn.functional.normalize(k_mut, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        cross_attn = (q_mut @ k_mut.transpose(-2, -1)) * self.temperature_mut
        cross_attn = cross_attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_mut = (cross_attn @ v_mut)
        out_mut = rearrange(out_mut, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(torch.cat((out, out_mut), dim=1))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, block_type, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.block_type = block_type
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        
        if self.block_type == "down" or self.block_type == "bottleneck":
            self.attn = RecurrentAttention(dim, num_heads, bias)
        else:
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        
        if self.block_type == "down" or self.block_type == "bottleneck":
            self.hidden = None

    def forward(self, x, motion):
        
        if self.block_type == "down" or self.block_type == "bottleneck":
            hidden_warp = warp_tensor(self.hidden, motion)
            x = x + self.attn(self.norm1(x), self.norm1(hidden_warp))
            x = x + self.ffn(self.norm2(x))
            self.hidden = x
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        
        return x

class BasicLayer(nn.Module):

    def __init__(self, dim, depth, num_heads, sample_method=None, block_type="down"):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.block_type = block_type
        
        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, block_type=block_type, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="BiasFree")
            for i in range(depth)])

        # patch merging layer
        if sample_method is not None:
            self.sample_method = sample_method(n_feat=dim)
        else:
            self.sample_method = None

    def forward(self, x, motion):
        
        for i in range(self.depth):
            x = self.blocks[i](x, motion)
            
        if self.sample_method is not None:
            x_sample = self.sample_method(x)
            return x_sample, x
        else:
            return x, None
    
    def reset_hidden(self, x, dfac):
        
        bs, C, H, W = x.shape
        for i in range(self.depth):
            self.blocks[i].hidden = torch.zeros((bs, self.dim, H // dfac, W // dfac), device=device)
    
class RecurrentBlock(nn.Module):

    def __init__(self, dim, depth, num_heads, block_type, recon_dim):
        
        super(RecurrentBlock, self).__init__()

        if block_type == "down":
            sample_method = Downsample
        elif block_type == "bottleneck":
            sample_method = None
        elif block_type == "up":
            sample_method = Upsample
            self.project_conv = nn.Conv2d(2 * dim, dim, 1)
            self.recon_conv = nn.Conv2d(dim, recon_dim + 1, 1)
        elif block_type == "last":
            sample_method = None
            self.project_conv = nn.Conv2d(2 * dim, dim, 1)
            self.recon_conv = nn.Conv2d(dim, recon_dim, 1)
            
        self.block_type = block_type
        self.transformer_layer = BasicLayer(dim=dim, depth=depth, num_heads=num_heads, sample_method=sample_method, block_type=block_type)
        self.kernels = None

    def forward(self, x, motion):
        
        if self.block_type == "down" or self.block_type == "bottleneck":
            op2, _ = self.transformer_layer(x, motion)
            
        elif self.block_type == "up":
            op1 = self.project_conv(x)
            op2, op2_recon = self.transformer_layer(op1, motion)
            self.kernels = self.recon_conv(op2_recon)
        
        elif self.block_type == "last":
            op1 = self.project_conv(x)
            op2, _ = self.transformer_layer(op1, motion)
            op2 = self.recon_conv(op2)
        
        return op2
    
    def reset_hidden(self, x, dfac):
        
        if self.block_type == "down" or self.block_type == "bottleneck":
            self.transformer_layer.reset_hidden(x, dfac)

class RecurrentUNet(nn.Module):

    def __init__(self, input_channels, base_dim, depth, recon_dim):
        super(RecurrentUNet, self).__init__()
        
        self.input_channels = input_channels
        self.base_dim = base_dim
        self.depth = depth
        self.first_projection = nn.Conv2d(self.input_channels, base_dim, 3, padding=1)
        
        down_layers = []
        up_layers = []
        
        down_depth = [2, 2, 3]
        up_depth = [3, 2, 2]
        bottleneck_depth = 3
        last_depth = 2
        
        down_head = [1, 2, 4]
        up_head = [4, 2, 1]
        bottleneck_head = 8
        last_head = 1
        
        for i in range(depth):
            down_layers.append(RecurrentBlock(dim=base_dim * (2 ** i), 
                                              depth=down_depth[i], 
                                              num_heads=down_head[i],
                                              block_type="down",
                                              recon_dim=0))
        
        self.bottleneck = RecurrentBlock(dim=base_dim * (2 ** depth),
                                         depth=bottleneck_depth, 
                                         num_heads=bottleneck_head,
                                         block_type="bottleneck",
                                         recon_dim=0)

        for i in range(depth):
            up_layers.append(RecurrentBlock(dim=base_dim * (2 ** (depth - i)), 
                                            depth=up_depth[i],
                                            num_heads=up_head[i],
                                            block_type="up",
                                            recon_dim=recon_dim))
        
        self.last = RecurrentBlock(dim=base_dim,
                                   depth=last_depth,
                                   num_heads=last_head, 
                                   block_type="last",
                                   recon_dim=recon_dim)
        
        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, x, motion):
        
        downs = []
        input = x[:, :self.input_channels]
        down = self.first_projection(input)
        down0 = down
        
        for i in range(self.depth): 
            down = self.down_layers[i](down, motion)
            downs.append(down)
        
        up = self.bottleneck(down, motion)
        
        for i in range(self.depth):
            up = self.up_layers[i](torch.cat((up, downs[self.depth - i - 1]), dim=1), motion)

        output = self.last(torch.cat((up, down0), dim=1), motion)

        return output

    def reset_hidden(self, x):
        
        for i in range(self.depth):
            self.down_layers[i].reset_hidden(x, 2 ** i)        
        self.bottleneck.reset_hidden(x, 2 ** self.depth)

class MLTDSimple(nn.Module):
    
    def __init__(self, cfg):
        
        super(MLTDSimple, self).__init__()
        assert cfg is not None # Need to have config
        
        # Fixed parameters
        self.input_channels = 13
        self.base_dim = 48
        self.depth = 3

        self.cfg = cfg
        self.reset_flag = True
        
        self.kernel_size_temporal= cfg.kernel_size_temporal ** 2
        self.kernel_size = cfg.kernel_size ** 2
        
        self.recurrent_unet = RecurrentUNet(input_channels=self.input_channels, base_dim=self.base_dim,
            depth=self.depth, recon_dim=(self.kernel_size_temporal + self.kernel_size + 1) * 4)
        self.apply_kernels = KernelApply(softmax=True)
        
        if hasattr(self.cfg, 'split') and self.cfg.split:
            self.denoised_coarse = None
        
    def forward(self, x):
        
        bs, t, w, h = x.shape[0], x.shape[1], x.shape[3], x.shape[4]
        preds = torch.zeros(bs, t, 3, w, h, device=device)
        denoised_coarse = [[None] * 4 for i in range(t)]
        
        for i in range(t):

            input, motion, radiance = x[:, i, :13], x[:, i, 13:16], x[:, i, 16:19] 

            # If the animation is split into multiple clips, we should not reset the hidden states
            if i == 0 and self.reset_flag:
                self.recurrent_unet.reset_hidden(radiance)
                if hasattr(self.cfg, 'split') and self.cfg.split:
                    self.reset_flag = False

            result = self.recurrent_unet(input, motion)
            
            for j in range(4):
                recon_offset = 0
                
                if self.kernel_size_temporal > 0:
                    kernels_temporal = self.recurrent_unet.up_layers[j].kernels[:, recon_offset:recon_offset+self.kernel_size_temporal] if j < 3 \
                        else result[:, recon_offset:recon_offset+self.kernel_size_temporal]
                    recon_offset += self.kernel_size_temporal
                kernels = self.recurrent_unet.up_layers[j].kernels[:, recon_offset:recon_offset+self.kernel_size] if j < 3 \
                    else result[:, recon_offset:recon_offset+self.kernel_size]
                recon_offset += self.kernel_size
                
                if self.kernel_size == 3:
                    denoised_coarse[i][j] = kernels
                else:
                    radiance_down = F.interpolate(radiance, scale_factor=0.5 ** (3 - j)) if j < 3 else radiance
                    alpha_0 = self.recurrent_unet.up_layers[j].kernels[:, recon_offset:recon_offset+1] if j < 3 else result[:, recon_offset:recon_offset+1] 
                    alpha_0 = torch.sigmoid(alpha_0).expand_as(radiance_down)
                    
                    if i == 0 or self.kernel_size_temporal == 0:
                        if hasattr(self.cfg, 'split') and self.cfg.split and self.denoised_coarse is not None:
                            denoised_coarse_warp = warp_tensor(self.denoised_coarse[j], motion)
                            denoised_coarse[i][j] = (1 - alpha_0) * self.apply_kernels(radiance_down.contiguous(), kernels) + \
                                                         alpha_0 * self.apply_kernels(denoised_coarse_warp, kernels_temporal)
                        else:
                            denoised_coarse[i][j] = self.apply_kernels(radiance_down.contiguous(), kernels)
                    else:
                        denoised_coarse_warp = warp_tensor(denoised_coarse[i - 1][j], motion)
                        denoised_coarse[i][j] = (1 - alpha_0) * self.apply_kernels(radiance_down.contiguous(), kernels) + \
                                                 alpha_0 * self.apply_kernels(denoised_coarse_warp, kernels_temporal)
                        
            denoised_fine = denoised_coarse[i][0]
            
            for j in range(3):
                recon_offset = self.kernel_size_temporal + self.kernel_size + 1
                alpha_1 = self.recurrent_unet.up_layers[j].kernels[:, recon_offset:recon_offset+1]
                alpha_1 = torch.sigmoid(alpha_1).expand_as(denoised_fine)
                denoised_down = F.interpolate(denoised_coarse[i][j + 1], scale_factor=0.5)
                denoised_up = F.interpolate(alpha_1 * denoised_down, scale_factor=2)
                denoised_fine = denoised_coarse[i][j + 1] - denoised_up + F.interpolate(alpha_1 * denoised_fine, scale_factor=2)

            preds[:, i] = denoised_fine

        if hasattr(self.cfg, 'split') and self.cfg.split:
            self.denoised_coarse = denoised_coarse[i]

        return preds