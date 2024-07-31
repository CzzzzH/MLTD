import os
import numpy as np
import pyexr
import re
import argparse
import json
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', type=str, default='scenes/indoors')
parser.add_argument('--scene-name', type=str, default="indoors")
parser.add_argument('--data-list', type=str, default='valid_renderings_indoors_train.json')
parser.add_argument('--dataset-name', type=str, default='train')
parser.add_argument('--spp-list', nargs='*', type=str, default=['32', '64', '128'])
parser.add_argument('--rendering-list', nargs='*', type=str, default=['mlt', 'mlt_extra', 'aux', 'gt'])
parser.add_argument('--frame-num', type=int, default=7)
parser.add_argument('--start-idx', type=int, default=0)
parser.add_argument('--end-idx', type=int, default=1000)
args = parser.parse_args()

def generate_h5py(input_path_prefix, output_path, spp):
    
    try:
        rendering_buffer = {}
        for rendering_type in args.rendering_list:
            rendering_buffer[rendering_type] = []

        for i in range(args.frame_num):
            
            # Generate GT (shared to save space)
            if 'gt' in args.rendering_list and spp == '32':
                rendering_buffer['gt'].append(pyexr.read(f'{input_path_prefix}_{i}_gt.exr'))
            
            # Generate Auxiliary Buffer (shared to save space)
            if 'aux' in args.rendering_list and spp == '32':
                aux_distance = pyexr.read(f'{input_path_prefix}_{i}_aux_distance.exr')
                aux_albedo = pyexr.read(f'{input_path_prefix}_{i}_aux_albedo.exr')
                aux_normal = pyexr.read(f'{input_path_prefix}_{i}_aux_normal.exr')
                aux_position = pyexr.read(f'{input_path_prefix}_{i}_aux_position.exr')
                aux_motion_one = pyexr.read(f'{input_path_prefix}_{i}_aux_motion_one.exr')
                rendering_buffer['aux'].append(np.concatenate([aux_distance, aux_albedo, aux_normal, aux_position, aux_motion_one], axis=-1))
            
            # Generate Input
            if 'mlt' in args.rendering_list:
                rendering_buffer['mlt'].append(pyexr.read(f'{input_path_prefix}_{i}_mlt+{spp}.exr'))
            if 'pssmlt' in args.rendering_list:
                rendering_buffer['pssmlt'].append(pyexr.read(f'{input_path_prefix}_{i}_pssmlt+{spp}.exr'))
            if 'pt' in args.rendering_list:
                rendering_buffer['pt'].append(pyexr.read(f'{input_path_prefix}_{i}_pt+{spp}.exr'))
            if 'bdpt' in args.rendering_list:
                rendering_buffer['bdpt'].append(pyexr.read(f'{input_path_prefix}_{i}_bdpt+{spp}.exr'))

            if 'mlt_extra' in args.rendering_list:
                extra_0 = pyexr.read(f'{input_path_prefix}_{i}_mlt+{spp}_extra_0.exr')
                extra_1 = pyexr.read(f'{input_path_prefix}_{i}_mlt+{spp}_extra_1.exr')
                extra_2 = pyexr.read(f'{input_path_prefix}_{i}_mlt+{spp}_extra_2.exr')
                extra_3 = pyexr.read(f'{input_path_prefix}_{i}_mlt+{spp}_extra_3.exr')
                rendering_buffer['mlt_extra'].append(np.concatenate([extra_0, extra_1, extra_2, extra_3], axis=-1))

            if 'pssmlt_extra' in args.rendering_list:
                extra_0 = pyexr.read(f'{input_path_prefix}_{i}_pssmlt+{spp}_extra_0.exr')
                extra_1 = pyexr.read(f'{input_path_prefix}_{i}_pssmlt+{spp}_extra_1.exr')
                rendering_buffer['pssmlt_extra'].append(np.concatenate([extra_0, extra_1], axis=-1))
        
        # Check NaN
        for value in rendering_buffer.values():
            if True in np.isnan(value):
                print('NaN detected')
                return False
        
        with h5py.File(output_path, 'w') as f:
            for key, value in rendering_buffer.items():
                if len(value) > 0:
                    f.create_dataset(key, data=np.stack(value, axis=0), compression='gzip', compression_opts=9)
        return True
    
    except Exception as e:
        print(e)
        return False
        
if __name__ == '__main__':
    
    for i in range(args.start_idx, args.end_idx):
        with open(os.path.join(args.base_dir, f'invalid_lists/invalid_renderings_{args.scene_name}_{i}.json'), 'r') as f:
            invalid_list = json.load(f)
            
        input_dir_path = os.path.join(args.base_dir, f'output_{args.scene_name}/{i}')
        output_base_dir = f'dataset/{args.dataset_name}_{args.scene_name}'
        filenames = os.listdir(input_dir_path)
        filenames.sort(key=lambda x:int(re.findall(r"\d+", x)[0]))
        
        data_list_path = os.path.join(args.base_dir, args.data_list)
        with open(data_list_path, 'r') as f:
            data_list = json.load(f)
                    
        for filename in filenames:
            if '0_gt' not in filename:
                continue
            tokens = filename.split('_')
            data_id = '_'.join(tokens[:-2])
            if data_id not in data_list:
                continue
            
            for spp in args.spp_list:
                if 'test' in args.dataset_name:
                    output_base_dir = f'{output_base_dir}_{spp}spp'
                os.makedirs(output_base_dir, exist_ok=True)
                output_path = os.path.join(output_base_dir, f'{data_id}_{spp}spp.h5')
                input_path_prefix = os.path.join(input_dir_path, data_id)                
                if not os.path.exists(output_path):
                    print(f'Generating Data: {data_id}')
                    if not generate_h5py(input_path_prefix, output_path, spp):
                        invalid_list.append(data_id)
                    
        with open(os.path.join(args.base_dir, f'invalid_lists/invalid_renderings_{args.scene_name}_{i}.json'), 'w') as f:
            json.dump(list(set(invalid_list)), f)