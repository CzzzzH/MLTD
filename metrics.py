import numpy as np
import pyfvvdp
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def linear2srgb(img):
    img = np.clip(img, a_min=0, a_max=np.inf)
    img = np.where(img <= 0.0031308, img * 12.92, 1.055 * np.power(img, 1 / 2.4) - 0.055)
    img = np.clip(img, a_min=0, a_max=1)
    return (img * 255).astype(np.uint8)

def luminance(img):
    return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

def tonemap(img):
    img = np.clip(img, a_min=0, a_max=np.inf)
    return img / (1 + img)

def calc_tmse(img, ref):
    img = tonemap(img)
    ref = tonemap(ref)
    mse = (img - ref) ** 2
    loss = mse / ((ref ** 2) + 1e-2)
    return np.mean(loss)

def calc_rmse(img, ref):
    img = tonemap(img)
    ref = tonemap(ref)
    mse = (img - ref) ** 2
    return np.sqrt(np.mean(mse))

def calc_ssim(img, ref):
    img = linear2srgb(img)
    ref = linear2srgb(ref)
    return ssim(ref, img, channel_axis=2)

def calc_psnr(img, ref):
    img = linear2srgb(img)
    ref = linear2srgb(ref)
    return psnr(img, ref)

def calc_rmae(img, ref):
    img = linear2srgb(img)
    ref = linear2srgb(ref)
    rmae = np.abs(img - ref) / (ref + 1e-2)
    return np.mean(rmae)

def calc_trmae(imgs, refs):
    trmae = []
    for i in range(1, len(imgs)):
        temporal_img = imgs[i] - imgs[i - 1]
        temporal_gt = refs[i] - refs[i - 1] 
        trmae_value = np.abs(temporal_img - temporal_gt) / (np.abs(temporal_gt) + 1e-2)
        trmae.append(trmae_value)
    return np.mean(trmae)

def calc_tpsnr(imgs, refs):
    tpsnr = []
    for i in range(1, len(imgs)):
        temporal_img = linear2srgb(imgs[i] - imgs[i - 1])
        temporal_ref = linear2srgb(refs[i] - refs[i - 1])
        tpsnr.append(psnr(temporal_img, temporal_ref))
    return np.mean(tpsnr)

def calc_fvvdp(imgs, refs, dim_order="FHWC"):
    fv = pyfvvdp.fvvdp(display_name='standard_hdr_linear', heatmap=None)
    res, _ = fv.predict(imgs, refs, dim_order=dim_order, frames_per_second=30)
    return res