import os
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='dataset/test_veach_ajar_original_32spp')
parser.add_argument('--output-dir', type=str, default='dataset/test_veach_ajar_original_32spp_split')
parser.add_argument('--split-size', type=str, default=20)
args = parser.parse_args()

def split_data(input_data_dir, output_data_dir, split_size=20):
    
    filenames = os.listdir(input_data_dir)
    rendering_list = ['mlt', 'mlt_extra', 'aux', 'gt']
    
    for filename in filenames:
        
        input_path = os.path.join(input_data_dir, filename)
        data = {}
        with h5py.File(input_path, 'r') as f:
            for rendering in rendering_list:
                data[rendering] = np.array(f[rendering])
                length = data[rendering].shape[0]
        
        os.makedirs(output_data_dir, exist_ok=True)
        print(f'Splitting {filename} into {length // split_size} files')
        for i in range(length // split_size):
            with h5py.File(f'{output_data_dir}/{filename[:-3]}_{i:03d}.h5', 'w') as f:
                for rendering in rendering_list:
                    f.create_dataset(rendering, data=data[rendering][i*split_size:(i+1)*split_size], compression='gzip', compression_opts=9)
                
if __name__ == "__main__":
    
    split_data(args.input_dir, args.output_dir, args.split_size)