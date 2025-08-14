# Hand Signature segmenation

This is a project for performing hand signature segmentation from already sign and stamped documentation.

to run the project organize the dataset into 

dataset/
├── train/
│   ├── {id}.jpg           # Original hand images
│   ├── {id}_mask.jpg      # Binary segmentation masks
└── test/
    ├── {id}.jpg           # Original hand images  
    ├── {id}_mask.jpg      # Binary segmentation masks

after which run the bash command

```bash
bash train_deeplabv3plus.sh
```

