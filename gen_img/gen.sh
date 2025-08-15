python gen_img/gen_image.py --split_mode train --num_images 5000 --output_dir ./dataset/new_dataset/train --mask_dir mask --crop_dir mask --split_file dataset/new_dataset/datasplit.json

python gen_img/gen_image.py --split_mode test --num_images 500 --output_dir ./dataset/new_dataset/test --mask_dir mask --crop_dir mask --split_file dataset/new_dataset/datasplit.json
