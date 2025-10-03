
import torch
import numpy as np
import os
import sys
import pathlib
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from PIL import Image
from torchvision import transforms

# Conditional debugpy
if os.getenv('ENABLE_DEBUGPY', 'false').lower() == 'true':
    try:
        import debugpy
        debugpy.listen(5678)
        print("Debugpy listening on port 5678")
    except ImportError:
        print("debugpy not installed, skipping debug mode")


ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)

# Import BiRefNet
from BiRefNet.models.birefnet import BiRefNet

DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
BIREFNET_MODEL = os.getenv('BIREFNET_MODEL', 'zhengpeng7/birefnet')

# Initialize BiRefNet model
if DEVICE == 'cuda':
    torch.set_float32_matmul_precision(['high', 'highest'][0])

birefnet_model = BiRefNet.from_pretrained(BIREFNET_MODEL)
birefnet_model.to(DEVICE)
birefnet_model.eval()

# Image transformation pipeline
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model using BiRefNet for segmentation
    """

    def get_results(self, masks, probs, width, height, from_name, to_name, label):
        results = []
        for mask, prob in zip(masks, probs):
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:4]
            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })

        return results

    def _birefnet_predict(self, img_url, input_box=None, task=None):
        """Predict segmentation mask using BiRefNet with bounding box."""
        # Load image
        image_path = get_local_path(img_url, task_id=task.get('id'))
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        
        # Handle default box values (full image if no box provided)
        if input_box is None or len(input_box) != 4:
            box = [0, 0, w, h]
        else:
            box = input_box
            for idx_coord_value, coord_value in enumerate(box):
                if coord_value == -1:
                    box[idx_coord_value] = [0, 0, w, h][idx_coord_value]
        
        # Crop image to bounding box
        image_crop = image.crop(box)
        
        # Transform and prepare input
        input_tensor = transform_image(image_crop).unsqueeze(0).to(DEVICE)
        
        # Prediction
        with torch.no_grad():
            preds = birefnet_model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        
        # Create canvas at 1024x1024 (BiRefNet output size)
        canvas = torch.zeros((1024, 1024))
        
        # Calculate box coordinates in canvas space
        box_to_canvas = [
            int(round(coord_value * (1024 / w, 1024 / h)[idx_coord_value % 2])) 
            for idx_coord_value, coord_value in enumerate(box)
        ]
        
        # Resize prediction to fit the box region in canvas
        pred_resized = torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0),
            size=(box_to_canvas[3] - box_to_canvas[1], box_to_canvas[2] - box_to_canvas[0]),
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        # Place prediction in canvas
        canvas[box_to_canvas[1]:box_to_canvas[3], box_to_canvas[0]:box_to_canvas[2]] = pred_resized
        
        # Resize canvas to original image dimensions
        canvas_resized = torch.nn.functional.interpolate(
            canvas.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        # Convert to binary mask
        mask_array = (canvas_resized > 0.5).cpu().numpy().astype(np.uint8)
        
        # Calculate confidence score (average prediction value)
        prob = float(canvas_resized.mean())
        
        return {
            'masks': [mask_array],
            'probs': [prob]
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Returns the predicted mask for a bounding box that has been drawn."""

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
            return ModelResponse(predictions=[])

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']
        img_url = tasks[0]['data'][value]

        all_results = []

        # Collect all rectanglelabels from the context
        for ctx in context['result']:
            if ctx.get('type') != 'rectanglelabels':
                continue

            # Extract bounding box and label
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            box_width = ctx['value']['width'] * image_width / 100
            box_height = ctx['value']['height'] * image_height / 100
            input_box = [int(x), int(y), int(x + box_width), int(y + box_height)]
            selected_label = ctx['value']['rectanglelabels'][0]

            print(f'Input box is {input_box}, label is {selected_label}')

            # Run prediction for the current bounding box
            predictor_results = self._birefnet_predict(
                img_url=img_url,
                input_box=input_box,
                task=tasks[0]
            )

            # Get results in Label Studio format
            results = self.get_results(
                masks=predictor_results['masks'],
                probs=predictor_results['probs'],
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                label=selected_label
            )
            all_results.extend(results)

        if not all_results:
            print("No bounding box found in context")
            return ModelResponse(predictions=[])

        # Each prediction should be its own dictionary in the list
        predictions = [{
            'result': all_results,
            'model_version': self.get('model_version'),
        }]

        return ModelResponse(predictions=predictions)
    
