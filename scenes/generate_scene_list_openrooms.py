import os
import json
import xml.etree.ElementTree as etree

if __name__ == "__main__":

    scene_name = 'indoors'
    source_dir = 'scenes/OpenRooms/scenes/xml'
    output_path = f'scenes/{scene_name}'
    valid_scenes = []
    scene_ids = sorted(os.listdir(source_dir))
    
    # Only keep the scene with windows
    for scene_id in scene_ids:
        
        xml_path = os.path.join(source_dir, f'{scene_id}/main.xml')
        scene = etree.parse(xml_path)
        window_flag = False
        for element in scene.iter():
            if element.tag == 'string' and 'value' in element.attrib and '/window' in element.attrib['value']:
                window_flag = True
        if window_flag:
            valid_scenes.append(scene_id)
    
    print(f"Valid Scene Number: {len(valid_scenes)}")
    with open(os.path.join(output_path, f'valid_scenes_{scene_name}.json'), 'w') as f:
        json.dump(valid_scenes, f)