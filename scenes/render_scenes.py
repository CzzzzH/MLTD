import os
import re
import argparse
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', type=str, default='scenes/indoors')
parser.add_argument('--scene-name', type=str, default='indoors')
parser.add_argument('--valid-name', type=str, default='valid_renderings_indoors.json')
parser.add_argument('--mode', type=str, default='mlt+32')
parser.add_argument('--start-idx', type=int, default=0)
parser.add_argument('--end-idx', type=int, default=1000)
parser.add_argument('--start-cam', type=int, default=0)
parser.add_argument('--end-cam', type=int, default=100)
parser.add_argument('--start-frame', type=int, default=0)
parser.add_argument('--end-frame', type=int, default=7)
parser.add_argument('--shuffle', action='store_true')
args = parser.parse_args()
    
if __name__ == '__main__':

    valid_renderings = None
    valid_path = os.path.join(args.base_dir, args.valid_name)
    if os.path.exists(valid_path):
        with open(valid_path, 'r') as f:
            valid_renderings = json.load(f)
            
    for scene_idx in range(args.start_idx, args.end_idx):
        xml_dir = os.path.join(args.base_dir, f'xml_{args.scene_name}/{scene_idx}')
        output_dir = os.path.join(args.base_dir, f'output_{args.scene_name}/{scene_idx}')
        if not os.path.exists(xml_dir):
            continue
        
        filenames = os.listdir(xml_dir)
        filenames_filtered = []
        
        for filename in filenames:
            numbers = re.findall(r'\d+', filename)
            cam_flag = args.start_cam <= int(numbers[1]) < args.end_cam
            frame_flag = args.start_frame <= int(numbers[2]) < args.end_frame
            if cam_flag and frame_flag:
                filenames_filtered.append(filename)
            
        filenames_filtered.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

        if args.shuffle:
            random.shuffle(filenames_filtered)
            
        for filename in filenames_filtered:
            if f'_{args.mode}.xml' in filename:
                tokens = filename.split('_')
                input_path = os.path.join(xml_dir, filename)
                output_path = os.path.join(output_dir, f'{filename[:-4]}.exr')
                if (valid_renderings and '_'.join(filename.split('_')[:-2]) not in valid_renderings) or os.path.exists(output_path):
                    continue
                print(f'Now Rendering: {output_path}')
                os.makedirs(output_dir, exist_ok=True)
                os.system(f'mitsuba {input_path} -o {output_path} -p 8')