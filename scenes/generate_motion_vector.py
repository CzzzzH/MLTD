import os 
import json
import numpy as np
import pyexr
import cv2
import argparse
import xml.etree.ElementTree as etree

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', type=str, default='scenes/indoors')
parser.add_argument('--scene-name', type=str, default="indoors")
parser.add_argument('--start-idx', type=int, default=0)
parser.add_argument('--end-idx', type=int, default=1000)
args = parser.parse_args()

def warp(img, motion_vectors):
    
    # Generate the coordinates grid
    height, width, _ = img.shape
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Apply motion vectors to get the source coordinates
    src_x = x - motion_vectors[..., 0]
    src_y = y - motion_vectors[..., 1]
    
    img = cv2.remap(img, src_x.astype(np.float32), src_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return img

def compute_motion_vectors(distance_t1, camera_params_t0, camera_params_t1, image_width, image_height, fov_x):
    
    origin_t0, target_t0, up_t0 = camera_params_t0
    origin_t1, target_t1, up_t1 = camera_params_t1
    
    # Create lookAt matrix
    def lookAt(origin, target, up):
        f = (target - origin) / np.linalg.norm(target - origin)
        r = np.cross(up, f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        u /= np.linalg.norm(u)
        
        return np.array([
            [r[0], r[1], r[2], -np.dot(r, origin)],
            [u[0], u[1], u[2], -np.dot(u, origin)],
            [-f[0], -f[1], -f[2], np.dot(f, origin)],
            [0, 0, 0, 1]
        ])

    view_t0 = lookAt(np.array(origin_t0), np.array(target_t0), np.array(up_t0))
    view_t1 = lookAt(np.array(origin_t1), np.array(target_t1), np.array(up_t1))
    
    # Create projection matrix with FOV axis being x-axis
    aspect_ratio = float(image_width) / image_height
    n = 1e-2
    f = 1e4
    distance_t1 = np.where(distance_t1 > 0, distance_t1, 1e-4)[..., 0]
    
    r = np.tan(fov_x * np.pi / 360.) * n
    t = r / aspect_ratio
    proj = np.array([
        [n/r, 0, 0, 0],
        [0, n/t, 0, 0],
        [0, 0, -(f+n)/(f-n), -2*f*n/(f-n)],
        [0, 0, -1, 0]
    ])
    
    proj_inv = np.linalg.inv(proj)
    view_t1_inv = np.linalg.inv(view_t1)
    
    # Compute screen space coordinates
    x_interval = 2 / image_width
    y_interval = 2 / image_height
    x = np.linspace(1 - x_interval / 2, -1 + x_interval / 2, image_width)
    y = np.linspace(1 - y_interval / 2, -1 + y_interval / 2, image_height)
    xv, yv = np.meshgrid(x, y)
    
    focal = image_width / (2 * np.tan(fov_x * np.pi / 360.))
    depth_t1 = distance_t1 * focal / np.sqrt(focal ** 2 + (image_width * xv / 2) ** 2 + (image_height * yv / 2) ** 2) 
    clip_z = (f + n)/(f - n) + (2 * f * n / -depth_t1.flatten()) / (f - n)
    
    # Unproject from camera at t
    clip_coords = np.vstack((xv.flatten(), yv.flatten(), clip_z.flatten(), np.ones(xv.size)))
    camera_coords = proj_inv @ clip_coords
    world_coords = view_t1_inv @ camera_coords
    world_coords /= world_coords[3, :] # Homogeneous coordinatesdepth_t1_warp
    
    # Project to screen at time t-1
    new_coords = proj @ view_t0 @ world_coords
    new_coords /= new_coords[3, :]
    motion_vectors = np.column_stack(((new_coords[0, :] - xv.flatten()) * image_width / 2, (new_coords[1, :] - yv.flatten()) * image_height / 2))
    return motion_vectors.reshape((image_height, image_width, 2))

def detach_ref(source_dir, filename):

    invalid_list = []

    try:
        color = pyexr.read(os.path.join(source_dir, filename), "color")
        distance = pyexr.read(os.path.join(source_dir, filename), "distance")
        albedo = pyexr.read(os.path.join(source_dir, filename), "albedo")
        normal = pyexr.read(os.path.join(source_dir, filename), "normal")
        position = pyexr.read(os.path.join(source_dir, filename), "position")
        
        for img in [distance, albedo, normal, position, color]:
            if True in np.isnan(img) or True in np.isinf(img):
                raise ValueError("Invalid data")
                
        pyexr.write(os.path.join(source_dir, filename.replace('ref', 'aux_distance')), distance)
        pyexr.write(os.path.join(source_dir, filename.replace('ref', 'aux_albedo')), albedo)
        pyexr.write(os.path.join(source_dir, filename.replace('ref', 'aux_normal')), normal)
        pyexr.write(os.path.join(source_dir, filename.replace('ref', 'aux_position')), position)

    except Exception as e:
        print(e)
        scene_idx = '_'.join(filename.split('_')[:-2])
        invalid_list.append(scene_idx)

    return invalid_list
        
def generate_motion_vector(scene_idx, invalid_list):
    
    xml_dir = os.path.join(args.base_dir, f'xml_{args.scene_name}/{scene_idx}')
    output_dir = os.path.join(args.base_dir, f'output_{args.scene_name}/{scene_idx}')
    
    # Detach auxiliary buffer at first
    filenames = os.listdir(output_dir)
    for filename in filenames:
        if 'ref' not in filename:
            continue
        data_id = '_'.join(filename.split('_')[:-2])
        if data_id in invalid_list:
            continue
        if not os.path.exists(os.path.join(output_dir, filename.replace('ref', 'aux_albedo'))):
            invalid_list += detach_ref(output_dir, filename)
     
    # Generate motion vectors
    filenames = os.listdir(output_dir)
    for filename in filenames:
        
        if 'aux_distance' not in filename:
            continue
        if os.path.exists(os.path.join(output_dir, filename.replace('aux_distance', 'aux_motion_one'))):
            continue
        tokens = filename.split('_')
        data_id = '_'.join(tokens[:-3])
        if data_id in invalid_list:
            continue

        try:
            print(f"Generate Motion Vector for {filename} ")
            frame_idx = int(tokens[-3])
            distance_map = pyexr.read(os.path.join(output_dir, filename))
            motion_vectors = np.zeros((distance_map.shape[0], distance_map.shape[1], 3))

            if frame_idx >= 1:
                xml_path_prev = os.path.join(xml_dir, f'{data_id}_{frame_idx - 1}_gt.xml')
                xml_path_current = os.path.join(xml_dir, f'{data_id}_{frame_idx}_gt.xml')
                root_prev = etree.parse(xml_path_prev).getroot()
                root_current = etree.parse(xml_path_current).getroot()
                look_at_prev = root_prev.find('sensor').find('transform').find('lookAt')
                look_at_current = root_current.find('sensor').find('transform').find('lookAt')
                origin_prev = [float(value) for value in look_at_prev.attrib['origin'].split(' ')]
                origin_current = [float(value) for value in look_at_current.attrib['origin'].split(' ')]
                target_prev = [float(value) for value in look_at_prev.attrib['target'].split(' ')]
                target_current = [float(value) for value in look_at_current.attrib['target'].split(' ')]
                up_prev = [float(value) for value in look_at_prev.attrib['up'].split(' ')]
                up_current = [float(value) for value in look_at_current.attrib['up'].split(' ')]
                
                camera_params_prev = [origin_prev, target_prev, up_prev]
                camera_params_current = [origin_current, target_current, up_current]
                img_width = distance_map.shape[1]
                img_height = distance_map.shape[0]
                fov_x = float(root_prev.find('sensor').find('float').attrib['value'])
                motion_vectors = compute_motion_vectors(distance_map, camera_params_prev, camera_params_current, img_width, img_height, fov_x)
            
                position_prev = pyexr.read(os.path.join(output_dir, f'{data_id}_{frame_idx - 1}_aux_position.exr'))
                position_current = pyexr.read(os.path.join(output_dir, f'{data_id}_{frame_idx}_aux_position.exr'))
                position_current_warp = warp(position_prev, motion_vectors)
                position_error = np.sqrt(np.sum((position_current - position_current_warp) ** 2, axis=2))
                mask = np.where(position_error > 0.1, 0.0, 1.0)[..., np.newaxis]
                motion_vectors = np.concatenate([motion_vectors, mask.astype(motion_vectors.dtype)], axis=2)
                position_current_warp = np.where(mask > 0.5, position_current_warp, 0.0)

            pyexr.write(os.path.join(output_dir, filename.replace('aux_distance', f'aux_motion_one')), motion_vectors)

        except Exception as e:
            print(e)
            invalid_list.append(data_id)
        
    return invalid_list
            
if __name__ == '__main__':
    
    os.makedirs(os.path.join(args.base_dir, f'invalid_lists'), exist_ok=True)
    for i in range(args.start_idx, args.end_idx):
        if os.path.exists(os.path.join(args.base_dir, f'invalid_lists/invalid_renderings_{args.scene_name}_{i}.json')):
            with open(os.path.join(args.base_dir, f'invalid_lists/invalid_renderings_{args.scene_name}_{i}.json'), 'r') as f:
                invalid_list = json.load(f)
        else:
            invalid_list = []
        invalid_list = generate_motion_vector(i, invalid_list)
        with open(os.path.join(args.base_dir, f'invalid_lists/invalid_renderings_{args.scene_name}_{i}.json'), 'w') as f:
            json.dump(list(set(invalid_list)), f)