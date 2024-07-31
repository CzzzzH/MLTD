### Test on Veach / Monkey / crystal (split for saving CPU memory)
python test.py \
       task_id=0 \
       num_workers=0 \
       test_dir="dataset/test_veach_32spp_split" \
       statistics_dir="statistics/test" \
       split=True

python test.py \
       task_id=0 \
       num_workers=0 \
       test_dir="dataset/test_monkey_128spp_split" \
       statistics_dir="statistics/test" \
       split=True

python test.py \
       task_id=0 \
       num_workers=0 \
       test_dir="dataset/test_crystal_128spp_split" \
       statistics_dir="statistics/test" \
       split=True


### Test on 20 OpenRooms Test scenes
python test.py \
       task_id=0 \
       num_workers=0 \
       test_dir="dataset/test_indoors_test_16spp" \
       statistics_dir="statistics/test"

python test.py \
       task_id=0 \
       num_workers=0 \
       test_dir="dataset/test_indoors_test_32spp" \
       statistics_dir="statistics/test"

python test.py \
       task_id=0 \
       num_workers=0 \
       test_dir="dataset/test_indoors_test_64spp" \
       statistics_dir="statistics/test"

python test.py \
       task_id=0 \
       num_workers=0 \
       test_dir="dataset/test_indoors_test_128spp" \
       statistics_dir="statistics/test"