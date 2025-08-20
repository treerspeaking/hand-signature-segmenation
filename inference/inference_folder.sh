python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_59/checkpoints/best_hand_segmentation_epoch=55_val/iou_epoch=0.8206_val/f1_epoch=0.9013.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-unet-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_59/checkpoints/best_hand_segmentation_epoch=46_val/iou_epoch=0.8249_val/f1_epoch=0.9038.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-unet-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_60/checkpoints/best_hand_segmentation_epoch=47_val/iou_epoch=0.7148_val/f1_epoch=0.8328.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-segformer-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_61/checkpoints/best_hand_segmentation_epoch=44_val/iou_epoch=0.8245_val/f1_epoch=0.9036.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-unetplusplus-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/good_version9/checkpoints/best_hand_segmentation_epoch=98_val/epoch_iou=0.8546.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-deeplabv3-old-base-no-focal