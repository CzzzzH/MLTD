import numpy as np
import random
import os
import argparse
from random import randint
from scipy import ndimage
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', type=str, default='dataset/train_indoors')
parser.add_argument('--rendering-type', type=str, default='mlt')
parser.add_argument('--start-idx', type=int, default=0)
parser.add_argument('--end-idx', type=int, default=1000)
args = parser.parse_args()

def get_variance_map(buffer, patch_size, relative=False):

    # compute variance
    mean = ndimage.uniform_filter(buffer, size=(patch_size, patch_size, 1))
    square_mean = ndimage.uniform_filter(buffer ** 2, size=(patch_size, patch_size, 1))
    variance = np.maximum(square_mean - mean ** 2, 0)

    # convert to relative variance if requested
    if relative:
        variance = variance / np.maximum(mean ** 2, 1e-4)

    # take the max variance along the three channels, gamma correct it to get a
    # less peaky map, and normalize it to the range [0,1]
    variance = variance.max(axis=2)
    variance = np.minimum(variance ** (1.0 / 2.2), 1.0)

    return variance / np.maximum(variance.max(), 1e-4)

def get_importance_map(buffers, metrics, weights, patch_size):
    
    if len(metrics) != len(buffers):
        metrics = [metrics[0]] * len(buffers)
    if len(weights) != len(buffers):
        weights = [weights[0]] * len(buffers)

    importance_map = None
    for buffer, metric, weight in zip(buffers, metrics, weights):
        if metric == 'variance':
            temp = get_variance_map(buffer, patch_size, relative=False)
        elif metric == 'relative':
            temp = get_variance_map(buffer, patch_size, relative=True)
        else:
            raise ValueError('Unknown metric: %s' % metric)

        if importance_map is None:
            importance_map = temp * weight
        else:
            importance_map += temp * weight

    return importance_map / np.max(importance_map)

def get_square_distance(x, y, patches):
    
    if len(patches) == 0:
        return np.infty
    dist = patches - [x, y]
    return np.sum(dist**2, axis=1).min()

def sample_patches_dart_throwing(exr_shapes, patch_size, num_patches, max_iter=5000):
    
    full_area = float(exr_shapes[0] * exr_shapes[1])
    sample_area = full_area / num_patches

    # get corresponding dart throwing radius
    radius = np.sqrt(sample_area/np.pi)
    min_square_distance = (2 * radius) ** 2

    # perform dart throwing, progressively reducing the radius
    rate = 0.96
    patches = np.zeros((num_patches, 2), dtype=int)
    x_min, x_max = 0, exr_shapes[1] - patch_size - 1
    y_min, y_max = 0, exr_shapes[0] - patch_size - 1
    for patch_index in range(num_patches):
        done = False
        while not done:
            for i in range(max_iter):
                x = randint(x_min, x_max)
                y = randint(y_min, y_max)
                square_distance = get_square_distance(x, y, patches[:patch_index, :])
                if square_distance > min_square_distance:
                    patches[patch_index, :] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                min_square_distance = (2 * radius) ** 2
    return patches

def get_region_list(exr_shapes, step):
    
    regions = []
    for y in range(0, exr_shapes[0], step):
        if y // step % 2 == 0:
            xrange = range(0, exr_shapes[1], step)
        else:
            xrange = reversed(range(0, exr_shapes[1], step))
        for x in xrange:
            regions.append((x, x+step, y, y+step))
    return regions

def split_patches(patches, region):

    current = np.empty_like(patches)
    remain = np.empty_like(patches)
    current_count, remain_count = 0, 0
    for i in range(patches.shape[0]):
        x, y = patches[i, 0], patches[i, 1]
        if region[0] <= x <= region[1] and region[2] <= y <= region[3]:
            current[current_count, :] = [x, y]
            current_count += 1
        else:
            remain[remain_count, :] = [x, y]
            remain_count += 1
    return current[:current_count, :], remain[:remain_count, :]

def prune_patches(exr_shapes, patches, patch_size, importance_map):
    
    pruned = np.empty_like(patches)
    remain = np.copy(patches)
    count, error = 0, 0
    for region in get_region_list(exr_shapes, 4*patch_size):
        current, remain = split_patches(remain, region)
        for i in range(current.shape[0]):
            x, y = current[i, 0], current[i, 1]
            if importance_map[y, x] - error > random.random():
                pruned[count, :] = [x, y]
                count += 1
                error += 1 - importance_map[y, x]
            else:
                error += 0 - importance_map[y, x]
    return pruned[:count, :]
    
def importance_sampling(buffers, patch_size, num_patches):
    
    # build the importance map
    metrics = ['relative', 'variance']
    weights = [1.0, 1.0]
    importance_map = get_importance_map(buffers, metrics, weights, patch_size)

    # get patches
    patches = sample_patches_dart_throwing(buffers[0].shape[:2], patch_size, num_patches)

    # prune patches
    pad = 0
    pruned = np.maximum(0, prune_patches(buffers[0].shape[:2], patches + pad, patch_size, importance_map) - pad)
    return pruned + pad

def generate_patches(dataset_path):

    data_list = os.listdir(dataset_path)
    data_list.sort()
    for data in data_list:
        tokens = data.split('_')
        scene_idx = int(tokens[1])
        if args.start_idx <= scene_idx and scene_idx < args.end_idx:
            print(f"Generated patches for {data}")
            input_path = os.path.join(dataset_path, data)
            ref_path = '_'.join(input_path.split('_')[:-1]) + '_32spp.h5'
            with h5py.File(input_path, 'r') as f:
                noisy = f[f'{args.rendering_type}'][0].clip(0, 1e3)
            with h5py.File(ref_path, 'r') as f:
                normal = f['aux'][0, :, :, 4:7]
            patches = importance_sampling([noisy, normal], 128, 300)
            patches[:, [0, 1]] = patches[:, [1, 0]]
            with h5py.File(input_path, 'a') as f:
                if 'patches' not in f:
                    f.create_dataset('patches', data=patches, compression='gzip', compression_opts=9)

if __name__ == "__main__":
    generate_patches(args.base_dir)