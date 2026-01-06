import os
import json
import numpy as np

def swap_yz(matrices, space='camera'):
    res = matrices.copy()
    if space == 'camera':
        # Swaps the output axes
        res[:, [1, 2], :] = res[:, [2, 1], :]
    elif space == 'world':
        # Swaps the input axes
        res[:, :, [1, 2]] = res[:, :, [2, 1]]
    return res

def invert_extrinsics(extrinsic_matrices):
    n = extrinsic_matrices.shape[0]
    inverted_list = []

    for i in range(n):
        M = extrinsic_matrices[i]
        R = M[:3, :3]
        t = M[:3, 3]

        # Calculate inverse
        R_inv = R.T
        t_inv = -R_inv @ t

        # Reconstruct
        M_inv = np.eye(4)
        M_inv[:3, :3] = R_inv
        M_inv[:3, 3] = t_inv
        
        inverted_list.append(M_inv)

    return np.array(inverted_list)

def extract_extrinsics(cameras_raw):
    extrinsics_list = []
    
    for cam in cameras_raw:
        view_matrix_flat = cam.get('extrinsics', {}).get('view_matrix', [])
        
        # Convert to a 4x4 numpy array
        matrix_4x4 = np.array(view_matrix_flat).reshape(4, 4)
        extrinsics_list.append(matrix_4x4)

    return np.stack(extrinsics_list)

def extract_camera_matrices(cameras_list):
    matrices = []
    
    for cam in cameras_list:
        raw_matrix = cam.get('intrinsics', {}).get('camera_matrix', [])
        # Convert to numpy and reshape to 3x3
        matrix_3x3 = np.array(raw_matrix).reshape(3, 3)
        matrices.append(matrix_3x3)

    # Stack into a single (n, 3, 3) array
    return np.stack(matrices) if matrices else np.array([])

def extract_distortion_coefficients(cameras_list):
    dist_coeffs = [
        camera['intrinsics']['distortion_coefficients'] 
        for camera in cameras_list 
        if 'intrinsics' in camera and 'distortion_coefficients' in camera['intrinsics']
    ]
    
    return np.array(dist_coeffs)

def read_calibration_raw(input_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    cameras = data.get('cameras', [])
    return cameras

def read_calibration_panoptic(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cameras = data.get('customized_sequence', [])
    return cameras