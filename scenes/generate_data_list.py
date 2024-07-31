import os
import json
import pyexr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start-idx', type=int, default=0)
parser.add_argument('--end-idx', type=int, default=1000)
parser.add_argument('--train-length', type=int, default=30000)
parser.add_argument('--valid-length', type=int, default=100)
parser.add_argument('--scene-name', type=str, default='indoors')
parser.add_argument('--option', type=str, default='raw')
args = parser.parse_args()

def merge_list(prefix, surfix):
    
    target_lists = [f'{prefix}_renderings_{args.scene_name}_{i}.json' for i in range(args.start_idx, args.end_idx)]
    output = []
    for target_list in target_lists:
        json_path = os.path.join(f'scenes/{args.scene_name}/{prefix}_lists/', target_list)
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as f:
            output += json.load(f)
    
    print(f"Output JSON Length: {len(output)}")
    np.random.shuffle(output)
    output_path = f'scenes/{args.scene_name}/{prefix}_renderings_{args.scene_name}_{surfix}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f)

def generate_raw_list():
    
    for i in range(args.start_idx, args.end_idx):
        source_dir = f'scenes/{args.scene_name}/output_{args.scene_name}/{i}'
        output_dir = f'scenes/{args.scene_name}/valid_lists/'
        if not os.path.exists(source_dir):
            continue

        valid_scenes = []
        filenames = sorted(os.listdir(source_dir))
        for filename in filenames:
            try:
                tokens = filename.split('_')
                data_name = '_'.join(tokens[:3])
                if '0_mlt+32.exr' not in filename:
                    continue
                is_valid = True 
                for frame in range(7):
                    preview_path = os.path.join(source_dir, f'{data_name}_{frame}_mlt+32.exr')
                    preview_img = pyexr.read(preview_path)
                    if np.isnan(preview_img).any():
                        is_valid = False
                        break
                    preview_img = np.clip(preview_img, 0, 1e3)
                    mean_lumiance = np.mean(0.2126 * preview_img[:, :, 0] + 0.7152 * preview_img[:, :, 1] + 0.0722 * preview_img[:, :, 2])
                    if mean_lumiance < 1e-4:
                        is_valid = False
                        break
                if is_valid:
                    valid_scenes.append(data_name)
            except Exception as e:
                print(e)
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'valid_renderings_{args.scene_name}_{i}.json'), 'w') as f:
            json.dump(valid_scenes, f)
            
def generate_data_list():
    
    idx_for_validation = [65, 212, 277, 379, 413, 457, 490, 493, 577, 690] # Use test scenes for validation
    train_list = []
    valid_list = []
    
    with open(f'scenes/{args.scene_name}/valid_renderings_{args.scene_name}_raw.json', 'r') as f:
        raw_list = json.load(f)
    with open(f'scenes/{args.scene_name}/invalid_renderings_{args.scene_name}_raw.json', 'r') as f:
        invalid_list = json.load(f)
        
    for data_idx in raw_list:
        if data_idx in invalid_list:
            continue
        scene_idx = data_idx.split('_')[-2]
        if int(scene_idx) in idx_for_validation:
            valid_list.append(data_idx)
        else:
            train_list.append(data_idx)
    
    with open(f'scenes/{args.scene_name}/valid_renderings_{args.scene_name}_train.json', 'w') as f:
        json.dump(train_list[:args.train_length], f)
    with open(f'scenes/{args.scene_name}/valid_renderings_{args.scene_name}_valid.json', 'w') as f:
        json.dump(valid_list[:args.valid_length], f)
        
if __name__ == "__main__":

    option = args.option
    if option == "raw":
        generate_raw_list()
    elif option == "data":
        generate_data_list()
    elif option == "merge_valid":
        merge_list('valid', 'raw')
    elif option == "merge_invalid":
        merge_list('invalid', 'raw')
    else:
        raise ValueError(f"Invalid option")