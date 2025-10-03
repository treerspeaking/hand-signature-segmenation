python -m inference.inference_folder --checkpoint logs/hand_segmentation/version_97_finetune_old_dataset/checkpoints/best_hand_segmentation_epoch=76_val/iou_epoch=0.8617_val/f1_epoch=0.9256.ckpt --folder /home/treerspeaking/src/python/hand_seg/inference/renamed_data --output ./inference/out-unet-train-old-dataset

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_81_5000-base/checkpoints/best_hand_segmentation_epoch=32_val/iou_epoch=0.7909_val/f1_epoch=0.8831.ckpt --folder /home/treerspeaking/src/python/hand_seg/inference/encoded_data --output /home/treerspeaking/src/python/hand_seg/inference/out-unet-base-focal-dice-jaccard-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_76_reduce_focal/checkpoints/best_hand_segmentation_epoch=29_val/iou_epoch=0.8390_val/f1_epoch=0.9122.ckpt --folder /home/treerspeaking/src/python/hand_seg/inference/renamed_data --output ./inference/out-unet-base-focal-dice-jaccard-same-color-reduce_focal

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_59/checkpoints/best_hand_segmentation_epoch=46_val/iou_epoch=0.8249_val/f1_epoch=0.9038.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-unet-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_60/checkpoints/best_hand_segmentation_epoch=47_val/iou_epoch=0.7148_val/f1_epoch=0.8328.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-segformer-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/version_61/checkpoints/best_hand_segmentation_epoch=44_val/iou_epoch=0.8245_val/f1_epoch=0.9036.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-unetplusplus-base

# python -m inference.inference_folder --checkpoint /home/treerspeaking/src/python/hand_seg/logs/hand_segmentation/good_version9/checkpoints/best_hand_segmentation_epoch=98_val/epoch_iou=0.8546.ckpt --folder /home/treerspeaking/src/python/hand_seg/encoded_data --output ./inference/out-deeplabv3-old-base-no-focal