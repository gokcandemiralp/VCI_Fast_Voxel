import sys
import os
import json

import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

path = '../lib'
if path not in sys.path:
    sys.path.insert(0, path)

from models import *
from core.config import config, update_config
from utils.transforms import get_affine_transform, get_scale
from utils.vci import get_resize_transform, load_model, render_image_with_poses, rescale_image, save_predictions_to_json

config_file = "config.yaml"    # the base config file
update_config(config_file)

# overwrite the config
cam_file = 'calibration.json'           # the calibrated camera params
output_dir = 'output/prediction_jsons'  # the output directory saving the prediction results
seq = 'customized_sequence'             # specify the name of your sequence
input_base_path = "/data/vci/motion_markers"
sequence_name = "2026_01_29_motion_markers_002"
frame_range = 209 # 209
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

backbone_file = '../backbone/pose_resnet50_panoptic.pth.tar' # backbone ResNet model
model_file = '../output/panoptic/jln64/model_best.pth.tar'   # pre-trained model


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# transform = transforms.Compose([transforms.ToTensor(), normalize])
transform_vis = transforms.ToTensor()
ori_image_size = config.DATASET.ORI_IMAGE_SIZE
image_size = config.DATASET.IMAGE_SIZE
resize_transform = get_resize_transform(ori_image_size, image_size)

all_rgb_inputs = []
view_names = ["C0019.png","C0025.png","C0030.png","C0031.png","C0039.png"]

# -----------------------------------------------

missing_files = []
print(f"Checking existence of {frame_range * len(view_names)} images...")
for frame_idx in range(frame_range):
    for view_name in view_names:
        # Construct the exact path used in the main loop
        image_path = os.path.join(input_base_path, sequence_name, f"frame_{frame_idx:05d}", "rgb", view_name)
        
        if not os.path.exists(image_path):
            missing_files.append(image_path)

if missing_files:
    print(f"ERROR: Found {len(missing_files)} missing image(s):")
    for path in missing_files[:10]: # Print first 10 only to avoid spamming
        print(f"   - {path}")
    if len(missing_files) > 10:
        print(f"   ... and {len(missing_files) - 10} more.")
    
    # Stop execution here so you don't get crashes later
    sys.exit(1) 

print("All images found. Starting load...")

# -----------------------------------------------

for frame_idx in tqdm(range(frame_range), desc=f"Loading rgb input frames"):
    inputs = []
    for view_name in view_names:
        image_path = os.path.join(input_base_path,sequence_name,f"frame_{frame_idx:05d}","rgb",view_name) # Load every two frames, 15 FPS is enough tp preview
        input = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = rescale_image(input, image_size)
        input = transform_vis(input)
        inputs.append(input)

    all_rgb_inputs.append(torch.stack(inputs, dim=0).unsqueeze(0))

# load camera params and metadata
with open(cam_file) as cfile:
    cameras = json.load(cfile)
meta = {'seq': [seq]}
resize_transform = torch.as_tensor(resize_transform, dtype=torch.float, device=config.DEVICE)


# load model
backbone, model = load_model(config, backbone_file, model_file)
view = 4
rendered_images = []
all_poses_list = []
for frame_idx in tqdm(range(frame_range), desc=f"Predicting poses"):
    with torch.no_grad():
        prediction_inputs = all_rgb_inputs[frame_idx].to(config.DEVICE)
        fused_poses, plane_poses, proposal_centers, input_heatmaps, _ = model(backbone=backbone, views=prediction_inputs, 
                                                                            meta=meta, cameras=cameras, 
                                                                            resize_transform=resize_transform)
        all_poses_list.append(fused_poses)

poses = torch.cat(all_poses_list, dim=0)
save_predictions_to_json(poses, f"{output_dir}/seq{sequence_name[-3:]}_j15_t{timestamp}.json", config)