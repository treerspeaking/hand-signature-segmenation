from ultralytics.models import YOLO

# Load a model
model = YOLO("yolo11m.yaml")  # build a new model from YAML
model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11m.yaml").load("yolo11m.pt")  # build from YAML and transfer weights

# Train the model on sure-1 dataset
results = model.train(
    data="/home/treerspeaking/src/python/hand_seg/zero_shot/sure-1/data.yaml", 
    epochs=100, 
    imgsz=640,
    batch=16,
    name="sure-1-hand-detection",
    hsv_h=0.2,
    hsv_s=0.2,
    hsv_v=0.2,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.5,
    auto_augment=None,
)