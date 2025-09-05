python gen_img/gen_image.py --split_mode train --num_images 5000 --output_dir ./dataset/dataset-5000-no-same-color/train --mask_dir mask --crop_dir mask --split_file dataset/new_dataset/datasplit.json --same_color_ratio 0.0

python gen_img/gen_image.py --split_mode test --num_images 500 --output_dir ./dataset/dataset-5000-no-same-color/test --mask_dir mask --crop_dir mask --split_file dataset/new_dataset/datasplit.json --same_color_ratio 0.0
