import torch
import numpy as np
import matplotlib
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision
import cv2
import os
import json
import re

from IPython.display import HTML
matplotlib.use('Agg')
matplotlib.rcParams['animation.embed_limit'] = 200.0

from models import *
from utils.transforms import get_affine_transform, get_scale, affine_transform_pts_cuda as do_transform
from utils.cameras import project_pose
from utils.vis import colors, is_valid_coord


# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]

# panoptic
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
         [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

def get_resize_transform(ori_image_size, image_size):
    r = 0
    c = np.array([ori_image_size[0] / 2.0, ori_image_size[1] / 2.0])
    s = get_scale((ori_image_size[0], ori_image_size[1]), image_size)
    trans = get_affine_transform(c, s, r, image_size)
    return trans


# load the model and the Pose-ResNet backbone
def load_model(config, backbone_file, model_file):
    print('=> loading models...')
    
    backbone = eval('models.' + config.BACKBONE + '.get')(config)
    print('=> loading weights of the backbone')
    backbone.load_state_dict(torch.load(backbone_file))
    backbone = backbone.to(config.DEVICE)
    backbone.eval()   

    model = eval('models.' + config.MODEL + '.get')(config)
    print('=> load model state {}'.format(model_file))
    model.load_state_dict(torch.load(model_file))
    model = model.to(config.DEVICE)
    model.eval()
    
    return backbone, model

def render_image_with_poses(config, images, poses, meta, cameras, resize_transform, view_idx=0, show_joint_ids=False):
    batch_size, num_views, _, height, width = images.shape
    max_people = poses.shape[1]
    num_joints = poses.shape[2]

    # Initialize ndarr as None in case the view_idx is invalid
    ndarr = None

    if (view_idx < num_views):
        # 1. Prepare the base image from the batch
        batch_image = images[:, view_idx].flip(1)
        grid = torchvision.utils.make_grid(batch_image, 1, padding=0, normalize=True)
        
        # Convert to numpy format for OpenCV: (H, W, C) and BGR color space
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy().copy()
        
        # Determine skeletal structure (limbs) based on joint count
        limbs = eval("LIMBS{}".format(num_joints))

        for i in range(batch_size):
            curr_seq = meta['seq'][i]

            for n in range(max_people):
                # Filter out low-confidence poses
                if poses[i, n, 0, 4] < config.CAPTURE_SPEC.MIN_SCORE:
                    continue

                # Get unique color for each person (RGB to BGR for OpenCV)
                color = np.flip(np.array(matplotlib.colors.to_rgb(colors[int(n % 10)]))) * 255
                
                # Project 3D pose to 2D image coordinates
                pose_2d = project_pose(poses[i, n, :, :3], cameras[curr_seq][view_idx])
                pose_2d = do_transform(pose_2d, resize_transform)
                
                # Draw Joints
                for j in range(num_joints):
                    if is_valid_coord(pose_2d[j], width, height):
                        xc = pose_2d[j][0]
                        yc = i * height + pose_2d[j][1]
                        cv2.circle(ndarr, (int(xc), int(yc)), 8, color, -1)
                
                # Draw Limbs (Lines)
                for limb in limbs:
                    parent = pose_2d[limb[0]]
                    child = pose_2d[limb[1]]
                    
                    if is_valid_coord(parent, width, height) and is_valid_coord(child, width, height):
                        px, py = parent[0], i * height + parent[1]
                        cx, cy = child[0], i * height + child[1]
                        cv2.line(ndarr, (int(px), int(py)), (int(cx), int(cy)), color, 4)

                # Draw Joint IDs
                for j in range(num_joints):
                    if is_valid_coord(pose_2d[j], width, height):
                        xc = pose_2d[j][0]
                        yc = i * height + pose_2d[j][1]

                        if show_joint_ids:
                            # Offset text slightly to not cover the exact center
                            text_pos = (int(xc) + 4, int(yc) - 4) 
                            cv2.putText(
                                ndarr, 
                                str(j), 
                                text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,             # Font scale
                                (255, 255, 255), # White text
                                3,               # Thickness
                                cv2.LINE_AA
                            )
                            cv2.putText(
                                ndarr, 
                                str(j), 
                                text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,             # Font scale
                                (5, 5, 155),     # Red text
                                1,               # Thickness
                                cv2.LINE_AA
                            )

    if ndarr is not None:
        ndarr = cv2.cvtColor(ndarr, cv2.COLOR_BGR2RGB)
                
    return ndarr
    
def save_rendered_view(image_array, output_dir, filename):
    if image_array is None:
        print("Warning: Provided image array is None. Skipping save.")
        return False

    # FIX: Check if it's a PyTorch Tensor and convert to NumPy
    if isinstance(image_array, torch.Tensor):
        # Move to CPU, detach from graph, and convert to numpy
        image_array = image_array.detach().cpu().numpy()

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, filename)
    
    # Final check: OpenCV expects (Height, Width, Channels) and uint8 type
    # If your render function produces (Channels, Height, Width), we must permute
    if image_array.ndim == 3 and image_array.shape[0] == 3:
        image_array = image_array.transpose(1, 2, 0)

    # OpenCV uses BGR, ensure it's uint8
    image_array = image_array.astype(np.uint8)
    
    success = cv2.imwrite(save_path, image_array)
    return success

def rescale_image(image, target_size, is_mask=False):
    # target_size[0] is Width, target_size[1] is Height
    width = int(target_size[0])
    height = int(target_size[1])

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)

def show_video(frames, interval=200, save_path="video.gif", resize_ratio=None):
    print(f"Frame count: {len(frames)}")
    
    h, w = frames[0].shape[:2]

    if resize_ratio is not None:
        h = int(h * resize_ratio)
        w = int(w * resize_ratio)
        frames = [cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in frames]

    dpi = 100
    fig_w = w / dpi
    fig_h = h / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    im = ax.imshow(frames[0], cmap='gray', animated=True)

    def update(frame):
        im.set_array(frames[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=interval, blit=True
    )

    if save_path is not None:
        if not save_path.endswith('.gif'):
            save_path += '.gif'
        print(f"Saving GIF to {save_path}...")
        ani.save(save_path, writer=PillowWriter(fps=1000 // interval))
        print("Saved.")

    plt.close(fig)
    return HTML(ani.to_jshtml())

def save_predictions_to_json(poses, output_path, config):
    output_data = {}
    poses = poses.cpu()
    
    batch_size = poses.shape[0]
    num_joints = poses.shape[2]

    for frame_idx in range(batch_size):
        frame_key = str(frame_idx)
        output_data[frame_key] = {}
        person_idx = 0 
        
        score = poses[frame_idx, person_idx, 0, 4].item()
        
        if score >= config.CAPTURE_SPEC.MIN_SCORE:
            for joint_idx in range(num_joints):
                x = poses[frame_idx, person_idx, joint_idx, 0].item()
                y = poses[frame_idx, person_idx, joint_idx, 1].item()
                z = poses[frame_idx, person_idx, joint_idx, 2].item()
                
                output_data[frame_key][str(joint_idx)] = [x, y, z]
        else:
            print(f"Frame {frame_idx}: Person 0 score ({score:.4f}) is below threshold.")


    json_str = json.dumps(output_data, indent=4)
    def compact_coordinates(match):
        # Extract the inner content (e.g., "-0.1, \n -0.8, \n -0.3")
        content = match.group(1)
        # Remove all whitespace and newlines, then reconstruct with just commas
        # This results in: "-0.1,-0.8,-0.3"
        compact_content = ",".join(val.strip() for val in content.split(','))
        return f"[{compact_content}]"

    # Apply the regex substitution
    json_str = re.sub(r'\[\s*([-\d.,\seE]+?)\s*\]', compact_coordinates, json_str)

    # 4. Write the processed string to file
    try:
        with open(output_path, 'w') as f:
            f.write(json_str)
        print(f"Successfully saved predictions to {output_path}")
    except IOError as e:
        print(f"Error saving JSON file: {e}")