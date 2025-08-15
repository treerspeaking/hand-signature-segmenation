# python inference_folder.py --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_50/checkpoints/best_hand_segmentation_epoch=91_val/epoch_iou=0.8302.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-deeplabv3-old-base-voc

python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_57/checkpoints/best_hand_segmentation_epoch=44_val/epoch_iou=0.0000.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-unet-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_54/checkpoints/best_hand_segmentation_epoch=98_val/epoch_iou=0.8084.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-segformer-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_55/checkpoints/best_hand_segmentation_epoch=24_val/epoch_iou=0.8324.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-unetplusplus-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_56/checkpoints/best_hand_segmentation_epoch=91_val/epoch_iou=0.8160.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-deeplabv3-base