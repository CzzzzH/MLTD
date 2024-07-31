# python scenes/generate_scene_list_openrooms.py   ### We have made it for you
python scenes/generate_xml_openrooms.py --start-idx 0 --end-idx 1 --frame-num 7
python scenes/generate_xml_openrooms.py --start-idx 690 --end-idx 691 --frame-num 7

cd mitsuba
source setpath.sh
cd ../

# Render one training sequence of different sample counts
python scenes/render_scenes.py --mode mlt+32 --end-idx 1 --end-cam 1  --valid-name valid_renderings_indoors_train.json
python scenes/render_scenes.py --mode mlt+64 --end-idx 1 --end-cam 1  --valid-name valid_renderings_indoors_train.json
python scenes/render_scenes.py --mode mlt+128 --end-idx 1 --end-cam 1  --valid-name valid_renderings_indoors_train.json
python scenes/render_scenes.py --mode ref --end-idx 1 --end-cam 1  --valid-name valid_renderings_indoors_train.json
python scenes/render_scenes.py --mode gt --end-idx 1 --end-cam 1  --valid-name valid_renderings_indoors_train.json

# Render one validation sequence of different sample counts
python scenes/render_scenes.py --mode mlt+32 --start-idx 690 --end-idx 691 --start-cam 8 --end-cam 9 \
    --valid-name valid_renderings_indoors_valid.json
python scenes/render_scenes.py --mode mlt+64 --start-idx 690 --end-idx 691 --start-cam 8 --end-cam 9 \
    --valid-name valid_renderings_indoors_valid.json
python scenes/render_scenes.py --mode mlt+128 --start-idx 690 --end-idx 691 --start-cam 8 --end-cam 9 \
    --valid-name valid_renderings_indoors_valid.json
python scenes/render_scenes.py --mode ref --start-idx 690 --end-idx 691 --start-cam 8 --end-cam 9 \
    --valid-name valid_renderings_indoors_valid.json
python scenes/render_scenes.py --mode gt --start-idx 690 --end-idx 691 --start-cam 8 --end-cam 9 \
    --valid-name valid_renderings_indoors_valid.json

# Generate motion vectors
python scenes/generate_motion_vector.py --start-idx 0 --end-idx 1
python scenes/generate_motion_vector.py --start-idx 690 --end-idx 691

# Generate training & validation data
python scenes/generate_data.py --data-list valid_renderings_indoors_train.json --start-idx 0 --end-idx 1 --dataset-name train
python scenes/generate_data.py --data-list valid_renderings_indoors_valid.json --start-idx 690 --end-idx 691 --dataset-name valid

# (Optional) Genearte patches (we use random crop in the training if there aren't given patches)
# python scenes/generate_patches.py --start-idx 0 --end-idx 1 --base-dir dataset/train_indoors
# python scenes/generate_patches.py --start-idx 690 --end-idx 691 --base-dir dataset/valid_indoors