import xml.etree.ElementTree as etree
import numpy as np
import os
import copy
import argparse 
import json
import re
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', type=str, default='scenes/indoors')
parser.add_argument('--source-dir', type=str, default='scenes/OpenRooms/scenes')
parser.add_argument('--valid-path', type=str, default='scenes/indoors/valid_scenes_indoors.json')
parser.add_argument('--scene-name', type=str, default='indoors')
parser.add_argument('--start-idx', type=int, default=0)
parser.add_argument('--end-idx', type=int, default=1000)
parser.add_argument('--test-idx', type=int, default=0)
parser.add_argument('--test-cam', type=int, default=-1)
parser.add_argument('--frame-num', type=int, default=60)
parser.add_argument('--gt-spp', type=int, default=1024)
parser.add_argument('--height', type=int, default=-1)
parser.add_argument('--width', type=int, default=-1)
parser.add_argument('--modes', nargs='*', type=str, default=['mlt+32', 'mlt+64', 'mlt+128', 'ref', 'gt'])
args = parser.parse_args()

with open(args.valid_path, 'r') as f:
    valid_scenes = json.load(f)

def set_camera(scene, current_camera, lam):

    origins = current_camera['origin']
    targets = current_camera['target']
    ups = current_camera['up']

    origin_str = '_'.join([str(value) for value in origins[0]])
    rng = np.random.default_rng(seed=int(hashlib.md5(origin_str.encode()).hexdigest(), 16))
    shift = rng.normal(-1, 1, 3)
    shift = shift / np.linalg.norm(shift)

    for element in scene.iter():
        
        if element.tag == 'sensor':
            transform_element = etree.Element('transform')
            transform_element.attrib['name'] = 'toWorld'
            look_at = etree.Element('lookAt')
            
            origin = origins[0] * (1 - lam) + origins[-1] * lam
            target = targets[0] * (1 - lam) + targets[-1] * lam
            up = ups[0] * (1 - lam) + ups[-1] * lam
            
            look_at.set('origin', ' '.join([str(value) for value in origin]))
            look_at.set('target', ' '.join([str(value) for value in target]))
            look_at.set('up', ' '.join([str(value) for value in up]   ))
            transform_element.append(look_at)
            element.append(transform_element)      
 
        if element.tag == 'emitter' and element.attrib['type'] == 'spot':
            look_at = element.find('transform').find('lookAt')
            light_origin = np.array([float(value) for value in look_at.attrib['origin'].split(' ')])
            light_origin += shift * lam
            look_at.set('origin', ' '.join([str(value) for value in light_origin]))
 
    return scene

def set_basic(scene, mode, current_camera, current_frame, end_frame):

    scene = set_camera(scene, current_camera, current_frame / end_frame)
    integrator_flag = False
    spp = 128 # Default
    
    if '+' in mode:
        tokens = mode.split('+')
        mode, spp = tokens[0], tokens[1]
    
    for element in scene.iter():
        if element.tag == 'scene':
            del element.attrib['verion']
            element.set('version', '0.6.0')
        
        if element.tag == 'sampler':
            element.attrib['type'] = 'independent'
            spp_element = element.find('integer')
            
            if mode == 'ref': # Auxiliary buffer
                spp_element.set('value', '128')
                element.append(etree.Element('integer', {'name': 'seed', 'value': '0'}))
            elif mode == 'gt':
                spp_element.set('value', str(args.gt_spp))
                element.append(etree.Element('integer', {'name': 'seed', 'value': f'{args.gt_spp}'}))
            else:
                spp_element.set('value', spp)
                element.append(etree.Element('integer', {'name': 'seed', 'value': f'{spp}'}))
        
        if element.tag == 'film':
            element.append(etree.Element('boolean', {'name': 'banner', 'value': 'false'}))
            if mode == 'ref':
                element.append(etree.Element('string', {'name': 'pixelFormat', 'value': 'rgb, luminance, rgb, rgb, rgb'}))
                element.append(etree.Element('string', {'name': 'channelNames', 'value': 'color, distance, albedo, normal, position'}))
                
        if element.tag == 'integrator' and not integrator_flag:
            integrator_flag = True
            if mode == 'mlt' or mode == 'pssmlt':
                element.set('type', mode)
                element.append(etree.Element('integer', {'name': 'directSamples', 'value': '-1'}))
            elif mode == 'gt':
                element.set('type', 'pssmlt')
                element.append(etree.Element('integer', {'name': 'directSamples', 'value': '-1'}))
            elif mode == 'ref':
                element.set('type', 'multichannel')
                element.append(etree.Element('integrator', {'type': 'path'}))
                for field_name in ('distance', 'albedo', 'shNormal', 'position'):
                    string_element = etree.Element('string', {'name': 'field', 'value': field_name})
                    field_element = etree.Element('integrator', {'type': 'field'})
                    field_element.append(string_element)
                    element.append(field_element)
            else:
                element.set('type', mode)
        
        if element.tag == 'string':
            if 'name' in element.attrib and element.attrib['name'] == 'filename':
                tokens = element.attrib['value'].split('/')
                element.attrib['value'] = f'../../../OpenRooms/{"/".join(tokens[5:])}'
                
        if element.tag == 'integer':
            if 'name' in element.attrib and element.attrib['name'] == 'width' and args.width != -1:
                element.attrib['value'] = str(args.width)
            if 'name' in element.attrib and element.attrib['name'] == 'height' and args.height != -1:
                element.attrib['value'] = str(args.height)
        
    return scene

def set_bsdf(scene, scene_id, cam_id, indoor_flag, test_flag):
    
    root = scene.getroot()
    bsdf_elements = root.findall('bsdf')
    
    for bsdf_element in bsdf_elements:
        
        scales = bsdf_element.findall('float')
        rgbs = bsdf_element.findall('rgb')
        textures = bsdf_element.findall('texture')
        
        # Define material flags
        texture_flag = len(textures) > 0
        
        object_id = bsdf_element.attrib['id'].split('_')
        object_flag = bool(re.match("^\d+$", object_id[0]))
        if object_flag:
            obj_string = '_'.join([object_id[0], object_id[1], str(scene_id), str(cam_id)])
            rng = np.random.default_rng(seed=int(hashlib.md5(obj_string.encode()).hexdigest(), 16))
            rnd = rng.random()
            glass_flag = rnd > 0.5 if indoor_flag else rnd > 0.2
            conductor_flag = (not glass_flag and rnd > 0.3) if indoor_flag else False
        else:
            glass_flag = False
            conductor_flag = False
        
        # New element defination
        u_scale = etree.Element('float', {'name': 'uscale', 'value': '1'})
        v_scale = etree.Element('float', {'name': 'vscale', 'value': '1'})
        roughness_scale = etree.Element('float', {'name': 'scale', 'value': '1.0'})
        albedo_scale = etree.Element('rgb', {'name': 'scale', 'value': '1.0 1.0 1.0'})    
        normal_bsdf_element = etree.Element('bsdf', {'type': 'normalmap'})
        
        if conductor_flag:
            texture_flag = False
            conductor_list = ['Ag', 'Al', 'Au', 'Cr', 'Cu']
            microfacet_bsdf_element = etree.Element('bsdf', {'type': 'roughconductor'})
            microfacet_bsdf_element.append(etree.Element('string', {'name': 'distribution', 'value': 'ggx'}))
            if test_flag: # Test scenes in old configuration use fixed conductor
                microfacet_bsdf_element.append(etree.Element('float', {'name': 'alpha', 'value': '0.15'}))
                microfacet_bsdf_element.append(etree.Element('rgb', {'name': 'eta', 'value': '1.65746, 0.880369, 0.521229'}))
                microfacet_bsdf_element.append(etree.Element('rgb', {'name': 'k', 'value': '9.22387, 6.26952, 4.837'}))
            else:
                rnd_alpha = rng.uniform(0.05, 0.2)
                rnd_conductor = rng.integers(0, len(conductor_list))
                microfacet_bsdf_element.append(etree.Element('float', {'name': 'alpha', 'value': f'{rnd_alpha}'}))
                microfacet_bsdf_element.append(etree.Element('string', {'name': 'material', 'value': f'{conductor_list[rnd_conductor]}'}))
        else:
            microfacet_bsdf_element = etree.Element('bsdf', {'type': 'roughplastic'})
            microfacet_bsdf_element.append(etree.Element('string', {'name': 'distribution', 'value': 'ggx'}))
            microfacet_bsdf_element.append(etree.Element('rgb', {'name': 'specularReflectance', 'value': '0.5 0.5 0.5'}))
        
        # Extract the bsdf element
        if glass_flag:
            bsdf_element.attrib['type'] = 'dielectric'
        else:
            bsdf_element.attrib['type'] = 'twosided'
            if texture_flag:
                bsdf_element.append(normal_bsdf_element)
                normal_bsdf_element.append(microfacet_bsdf_element)
            else:
                bsdf_element.append(microfacet_bsdf_element)
        
        for scale_element in scales:
            if scale_element.attrib['name'] == 'uvScale':
                u_scale.attrib['value'] = scale_element.attrib['value']
                v_scale.attrib['value'] = scale_element.attrib['value']
            elif scale_element.attrib['name'] == 'roughnessScale':
                roughness_scale.attrib['value'] = scale_element.attrib['value']
            elif scale_element.attrib['name'] == 'roughness':
                microfacet_bsdf_element.append(etree.Element('float', {'name': 'alpha', 'value': scale_element.attrib['value']}))
            bsdf_element.remove(scale_element)
        
        for rgb_element in rgbs:
            if rgb_element.attrib['name'] == 'albedoScale':
                albedo_scale.attrib['value'] = rgb_element.attrib['value']
            if rgb_element.attrib['name'] == 'albedo':
                microfacet_bsdf_element.append(etree.Element('rgb', {'name': 'diffuseReflectance', 'value': rgb_element.attrib['value']}))
            bsdf_element.remove(rgb_element)
            
        for texture_element in textures:
            texture_name = texture_element.attrib['name']
            del texture_element.attrib['name']
            texture_element.append(u_scale)
            texture_element.append(v_scale)
            
            if texture_flag:
                if texture_name == 'albedo':
                    new_texture = etree.Element('texture', {'name': 'diffuseReflectance', 'type': 'scale'})
                    new_texture.append(albedo_scale)
                    new_texture.append(texture_element)
                elif texture_name == 'roughness':
                    new_texture = etree.Element('texture', {'name': 'alpha', 'type': 'scale'})
                    new_texture.append(roughness_scale)
                    new_texture.append(texture_element)
                elif texture_name == 'normal':
                    new_texture = texture_element
                    new_texture.append(etree.Element('float', {'name': 'gamma', 'value': '1.0'}))
                
                if texture_name == 'normal':
                    normal_bsdf_element.append(new_texture)
                else:
                    microfacet_bsdf_element.append(new_texture)
                
            bsdf_element.remove(texture_element)
        
    return scene

def set_shape(scene, scene_id, cam_id, test_flag):
    
    root = scene.getroot()
    shape_elements = scene.findall('shape')
    container_elements = []
    
    for i in range(len(shape_elements)):
        element = shape_elements[i]
        element_id = element.attrib['id']
        
        if "scene" in element_id and "object" in element_id:
            container_height = float(element.find('transform').find('translate').attrib['y'])
            container_id = element_id
            container_elements.append(element)
            
    # Replace the container with a plane for 30% of the scenes (generalize to outdoor scenes)
    container_string = '_'.join([container_id, str(scene_id), str(cam_id)])
    rng = np.random.default_rng(seed=int(hashlib.md5(container_string.encode()).hexdigest(), 16))
    rnd = rng.random()
    indoor_flag = True if test_flag else rnd > 0.3 # Test scenes in old configuration are all indoors
    
    if not indoor_flag:
        for container_element in container_elements:
            root.remove(container_element)        
        rnd_color = rng.uniform(0.2, 1, 3)
        new_plane_shape = etree.Element('shape', {'type': 'rectangle'})
        new_plane_shape_transform = etree.Element('transform', {'name': 'toWorld'})
        scale_element = etree.Element('scale', {'x': '100', 'y': '100', 'z': '100'})
        new_plane_shape_transform.append(scale_element)
        new_plane_shape_transform.append(etree.Element('lookAt', {'origin': f'0 {container_height} 0', 
                                                                  'target': f'0 {container_height + 1} 0', 
                                                                  'up': '0 0 1'}))
        new_plane_shape.append(etree.Element('boolean', {'name': 'flipNormals', 'value': 'false'}))
        new_plane_bsdf = etree.Element('bsdf', {'type': 'diffuse'})
        new_plane_bsdf.append(etree.Element('rgb', {'name': 'reflectance', 'value': f'{rnd_color[0]} {rnd_color[1]} {rnd_color[2]}'}))
        new_plane_shape.append(new_plane_shape_transform)
        new_plane_shape.append(new_plane_bsdf)
        root.append(new_plane_shape)
    
    for i in range(len(shape_elements)):
        element = shape_elements[i]
        element_id = element.attrib['id']
        
        if "scene" in element_id and "object" in element_id:
            container_height = float(element.find('transform').find('translate').attrib['y'])
            container_id = element_id
            container_elements.append(element)
        if "curtain" in element_id:
            root.remove(element)
            continue
        if "window" in element_id:
            element_transform = element.find('transform')
            transform_matrix = np.identity(4)
            offset_matrix_t = np.identity(4)
            offset_matrix_o = np.identity(4)
            offset_matrix_a = np.identity(4)
            
            if "window_4" in element_id:
                offset_matrix_t[:3, 3] = np.array([0.7, 0.6, 0.0])
            elif "window_3" in element_id:
                offset_matrix_t[:3, 3] = np.array([4.2, 1.1, -6.2])
            elif "window_2" in element_id:
                offset_matrix_t[:3, 3] = np.array([0.25, 1.25, 0.0])
            elif "window_1" in element_id:
                offset_matrix_t[:3, 3] = np.array([0.5, 0.0, -0.4])
            
            if "window_1" in element_id:
                offset_matrix_o[:3, 3] = offset_matrix_t[:3, 3] + np.array([0, 2, 0.3])
                offset_matrix_a[:3, 3] = offset_matrix_t[:3, 3] + np.array([0, 4, 0])
            elif "window_3" in element_id:
                offset_matrix_o[:3, 3] = offset_matrix_t[:3, 3] + np.array([0, 0.5, 2])
                offset_matrix_a[:3, 3] = offset_matrix_t[:3, 3] + np.array([0, 0, 4])
            else:
                offset_matrix_o[:3, 3] = offset_matrix_t[:3, 3] + np.array([0, 0.5, -2])
                offset_matrix_a[:3, 3] = offset_matrix_t[:3, 3] + np.array([0, 0, -4])
            
            for elem in reversed(list(element_transform)):
                if elem.tag == 'scale':
                    scale = np.identity(4)
                    scale[0, 0] = float(elem.attrib['x'])
                    scale[1, 1] = float(elem.attrib['y'])
                    scale[2, 2] = float(elem.attrib['z'])
                    transform_matrix = np.dot(transform_matrix, scale)
                elif elem.tag == 'rotate':
                    angle, x, y, z = float(elem.attrib['angle']), float(elem.attrib['x']), float(elem.attrib['y']), float(elem.attrib['z'])
                    angle = np.radians(angle)
                    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    x, y, z = x / norm, y / norm, z / norm
                    c, s = np.cos(angle), np.sin(angle)
                    r = np.array([
                        [x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0],
                        [y*x*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0],
                        [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c, 0],
                        [0, 0, 0, 1]])
                    transform_matrix = np.dot(transform_matrix, r)
                elif elem.tag == 'translate':
                    translate = np.identity(4)
                    translate[0, 3] = float(elem.attrib['x'])
                    translate[1, 3] = float(elem.attrib['y'])
                    translate[2, 3] = float(elem.attrib['z'])
                    transform_matrix = np.dot(transform_matrix, translate)
            
            target = np.dot(transform_matrix, offset_matrix_t)[:3, 3]
            origin = np.dot(transform_matrix, offset_matrix_o)[:3, 3]
            area_light_origin = np.dot(transform_matrix, offset_matrix_a)[:3, 3]
            
            # Add spot light (Test scenes in old configuration use fixed spot light attributes)
            rnd_rgb = rng.uniform(0.7, 1, 3) * 200 if not test_flag else np.array([200, 200, 200])
            if not indoor_flag:
                rnd_rgb *= 2 # Increase the intensity of the spot light for outdoor scenes
            rnd_angle = rng.uniform(15, 120) if not test_flag else 120
            rnd_bw = rng.uniform(0.75 * rnd_angle, rnd_angle) if not test_flag else 5
            new_spot_emitter = etree.Element('emitter', {'type': 'spot'}) 
            new_spot_emitter.append(etree.Element('rgb', {'name': 'intensity', 'value': f'{rnd_rgb[0]} {rnd_rgb[1]} {rnd_rgb[2]}'}))
            new_spot_emitter.append(etree.Element('float', {'name': 'cutoffAngle', 'value': f'{rnd_angle}'}))
            new_spot_emitter.append(etree.Element('float', {'name': 'beamWidth', 'value': f'{rnd_bw}'}))
            new_spot_emitter_transform = etree.Element('transform', {'name': 'toWorld'})
            new_spot_emitter_transform.append(etree.Element('lookAt', {'origin': f'{origin[0]} {origin[1]} {origin[2]}', 
                                                                       'target': f'{target[0]} {target[1]} {target[2]}', 
                                                                       'up': '0 1 0'}))
            new_spot_emitter.append(new_spot_emitter_transform)
            root.append(new_spot_emitter)
            
            if indoor_flag:
                # Add area light (we don't add it in the outdoor scenes)
                new_area_shape = etree.Element('shape', {'type': 'rectangle'})
                new_area_shape_transform = etree.Element('transform', {'name': 'toWorld'})
                scale_element = etree.Element('scale', {'x': '100', 'y': '100', 'z': '100'})
                new_area_shape_transform.append(scale_element)
                new_area_shape_transform.append(etree.Element('lookAt', {'origin': f'{area_light_origin[0]} {area_light_origin[1]} {area_light_origin[2]}', 
                                                                        'target': f'{target[0]} {target[1]} {target[2]}', 
                                                                        'up': '0 1 0'}))
                new_area_shape.append(new_area_shape_transform)
                new_area_shape.append(etree.Element('boolean', {'name': 'flipNormals', 'value': 'false'}))
                new_area_emitter = etree.Element('emitter', {'type': 'area'})
                new_area_emitter.append(etree.Element('rgb', {'name': 'radiance', 'value': '1 1 1'}))
                new_area_shape.append(new_area_emitter)
                root.append(new_area_shape)
        
        element.attrib['id'] = f'{element_id}_{i}'
        ref_elements = element.findall('ref')
        for ref_element in ref_elements:
            ref_element.attrib['name'] = ref_element.attrib['id']
    
    return scene, indoor_flag

def remove_original_emitter(scene):
    
    root = scene.getroot()
    emitters = scene.findall('emitter')
    for emitter in emitters:
        root.remove(emitter)
    shapes = scene.findall('shape')
    for shape in shapes:
        if shape.find('emitter') is not None:
            shape.remove(shape.find('emitter'))
    return scene

def generate_scene():
    
    output_dir = os.path.join(args.base_dir, f'xml_{args.scene_name}')
    start_scene_id = args.start_idx
    end_scene_id = args.end_idx
    
    os.makedirs(output_dir, exist_ok=True)
    cam_filename = 'cam.txt'
    xml_dirs = ['xml', 'xml1'] if args.test_cam < 0 else ['xml']
    
    for i in range(start_scene_id, min(end_scene_id , len(valid_scenes))):
        cam_offset = 0
        for xml_dir in xml_dirs:
            input_dir = os.path.join(args.source_dir, f'{xml_dir}/{valid_scenes[i]}')
            input_path = os.path.join(input_dir, f'main.xml')
            camera_path = os.path.join(input_dir, cam_filename)
            
            origins = []
            targets = []
            ups = []
            with open(camera_path, 'r') as f:
                cam_num = int(f.readline())
                for _ in range(cam_num):
                    origin = f.readline()[:-1].split(' ')
                    target = f.readline()[:-1].split(' ')
                    up = f.readline()[:-1].split(' ')
                    origins.append([float(value) for value in origin])
                    targets.append([float(value) for value in target])
                    ups.append([float(value) for value in up])

            # Cycle the camera
            origins.append(origins[0]) 
            targets.append(targets[0])
            ups.append(ups[0])
                       
            origins = np.array(origins, dtype=np.float32)
            targets = np.array(targets, dtype=np.float32)
            ups = np.array(ups, dtype=np.float32)
            
            cam_num = origins.shape[0] - 1
            cam_start = 0 if args.test_cam < 0 else args.test_cam
            cam_end = cam_num if args.test_cam < 0 else args.test_cam + 1
            
            for j in range(cam_start, cam_end):
                current_camera = {'origin': origins[j:j+2], 'target': targets[j:j+2], 'up': ups[j:j+2]}
                cam_id = cam_offset + j if args.test_cam < 0 else args.test_idx
                np.random.seed(cam_id)
                
                scene = etree.parse(input_path)
                scene = remove_original_emitter(scene)
                scene, indoor_flag = set_shape(scene, i, cam_id, args.test_cam >= 0)
                scene = set_bsdf(scene, i, cam_id, indoor_flag, args.test_cam >= 0)
                
                for k in range(args.frame_num):
                    for mode in args.modes:
                        new_scene = copy.deepcopy(scene)
                        new_scene = set_basic(new_scene, mode, current_camera, k, 60)
                        root = new_scene.getroot()
                        etree.indent(root, space="    ")
                        os.makedirs(os.path.join(output_dir, f'{i}'), exist_ok=True)
                        new_scene.write(os.path.join(output_dir, f'{i}/{args.scene_name}_{i}_{cam_id}_{k}_{mode}.xml'))
            
            cam_offset += cam_num
            
if __name__ == "__main__":
    
    generate_scene()