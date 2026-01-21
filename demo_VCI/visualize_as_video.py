import sys
import os
import json

import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

path = '../lib'
if path not in sys.path:
    sys.path.insert(0, path)

from models import *
from core.config import config, update_config
from utils.transforms import get_affine_transform, get_scale
from utils.vci import get_resize_transform, load_model, render_image_with_poses, show_video, rescale_image

config_file = "config.yaml"    # the base config file
update_config(config_file)

# overwrite the config
cam_file = 'calibration.json'  # the calibrated camera params
output_dir = 'output/deneme'          # the output directory saving the visualization results
seq = 'customized_sequence'    # specify the name of your sequence
input_base_path = "/data/vci"
sequence_name = "2025_10_28_ir_motions_001"
frame_range = 183
visualization_input_type = "mask"

backbone_file = '../backbone/pose_resnet50_panoptic.pth.tar' # backbone ResNet model
model_file = '../output/panoptic/jln64/model_best.pth.tar'   # pre-trained model


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])
ori_image_size = config.DATASET.ORI_IMAGE_SIZE
image_size = config.DATASET.IMAGE_SIZE
resize_transform = get_resize_transform(ori_image_size, image_size)

all_rgb_inputs = []
view_names = ["C0019.jpg","C0025.jpg","C0030.jpg","C0031.jpg","C0039.jpg"]
for frame_idx in tqdm(range(frame_range), desc=f"Loading rgb input frames"):
    inputs = []
    for view_name in view_names:
        image_path = os.path.join(input_base_path,sequence_name,f"frame_{(frame_idx*2):05d}","rgb",view_name) # Load every two frames, 15 FPS is enough tp preview
        input = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = rescale_image(input, image_size)
        input = transform(input)
        inputs.append(input)

    all_rgb_inputs.append(torch.stack(inputs, dim=0).unsqueeze(0))

all_mask_inputs = []
if (visualization_input_type == "mask"):
    for frame_idx in tqdm(range(frame_range), desc=f"Loading mask input frames"): # TODO: Reading monocolor masks
        inputs = []
        for view_name in view_names:
            image_path = os.path.join(input_base_path,sequence_name,f"frame_{(frame_idx*2):05d}","mask",f"mask_{view_name}") # Load every two frames, 15 FPS is enough tp preview
            input = cv2.imread(image_path, cv2.IMREAD_COLOR)
            input = rescale_image(input, image_size, is_mask = True)
            input = transform(input)
            inputs.append(input)

        all_mask_inputs.append(torch.stack(inputs, dim=0).unsqueeze(0))

# load camera params and metadata
with open(cam_file) as cfile:
    cameras = json.load(cfile)
meta = {'seq': [seq]}
resize_transform = torch.as_tensor(resize_transform, dtype=torch.float, device=config.DEVICE)


# load model
backbone, model = load_model(config, backbone_file, model_file)
view = 4
rendered_images = []
for frame_idx in tqdm(range(frame_range), desc=f"Predicting poses"):
    with torch.no_grad():
        prediction_inputs = all_rgb_inputs[frame_idx].to(config.DEVICE)

        if (visualization_input_type == "mask" and all_mask_inputs != []):
            visualization_inputs = all_mask_inputs[frame_idx].to(config.DEVICE)
        elif(visualization_input_type == "rgb" and all_rgb_inputs != []):
            visualization_inputs = prediction_inputs
        else:
            print("Error with the visualization input type")

        fused_poses, plane_poses, proposal_centers, input_heatmaps, _ = model(backbone=backbone, views=prediction_inputs, 
                                                                            meta=meta, cameras=cameras, 
                                                                            resize_transform=resize_transform)
        
        # visualization
        rendered_img = render_image_with_poses(config, visualization_inputs, fused_poses, meta, cameras, resize_transform, view)
        rendered_images.append(rendered_img)

show_video(rendered_images, interval=67, save_path=f"{output_dir}/{sequence_name}_v{view}.gif", resize_ratio=None)