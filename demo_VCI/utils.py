import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def print_euler_angles(extrinsic_matrices, degrees=True):
    for i, matrix in enumerate(extrinsic_matrices):
        # Extract the 3x3 rotation matrix from the top-left corner
        rotation_matrix = matrix[:3, :3]
        
        # Convert to a Scipy Rotation object
        # Note: We assume 'zyx' sequence for Pitch/Yaw/Roll 
        # but this depends on your specific coordinate system.
        r = R.from_matrix(rotation_matrix)
        
        # Get Euler angles
        # 'xyz' returns angles about X, then Y, then Z.
        # In many CV pipelines: X=Pitch, Y=Yaw, Z=Roll
        angles = r.as_euler('xyz', degrees=degrees)
        pitch, yaw, roll = angles
        
        unit = "degrees" if degrees else "radians"
        print(f"Camera {i}:")
        print(f"  Pitch (X): {pitch:7.2f} {unit}")
        print(f"  Yaw   (Y): {yaw:7.2f} {unit}")
        print(f"  Roll  (Z): {roll:7.2f} {unit}")
        print("-" * 30)


def extract_extrinsics_panoptic(cameras_list):
    extrinsics = []
    
    for cam in cameras_list:
        R = np.array(cam['R']) # Shape (3, 3)
        T = np.array(cam['T']) # Shape (3, 1)
        

        rt_matrix = np.eye(4)
        rt_matrix[:3, :3] = R
        rt_matrix[:3, 3] = T.reshape(3)
        
        extrinsics.append(rt_matrix)
        
    return np.stack(extrinsics, axis=0)

def extract_camera_matrices_panoptic(cameras_list):
    intrinsic_matrices = []

    for cam in cameras_list:
        K = np.array([
            [cam['fx'], 0,         cam['cx']],
            [0,         cam['fy'], cam['cy']],
            [0,         0,         1        ]
        ], dtype=np.float64)
        
        intrinsic_matrices.append(K)

    return np.array(intrinsic_matrices)

def extract_distortion_coefficients_panoptic(cameras_list):
    dist_coeffs = []
    
    for cam in cameras_list:
        # Extract k (radial): k1, k2, k3 are usually in cam['k']
        # Extract p (tangential): p1, p2 are usually in cam['p']
        k1, k2, k3 = cam['k'][0][0], cam['k'][1][0], cam['k'][2][0]
        p1, p2 = cam['p'][0][0], cam['p'][1][0]
        
        # Standard OpenCV format: [k1, k2, p1, p2, k3]
        dist_coeffs.append([k1, k2, p1, p2, k3])
        
    return np.array(dist_coeffs)

def m_to_mm(extrinsic_matrices):
    n = extrinsic_matrices.shape[0]
    mm_list = []

    for i in range(n):
        M = extrinsic_matrices[i]
        R = M[:3, :3]
        t = M[:3, 3] * 1000

        # Reconstruct
        M_mm = np.eye(4)
        M_mm[:3, :3] = R
        M_mm[:3, 3] = t
        
        mm_list.append(M_mm)

    return np.array(mm_list)

def invert_position(extrinsic_matrices):
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
        M_inv[:3, :3] = R
        M_inv[:3, 3] = t_inv
        
        inverted_list.append(M_inv)

    return np.array(inverted_list)

def extract_extrinsics_raw(cameras_raw):
    extrinsics_list = []
    
    for cam in cameras_raw:
        view_matrix_flat = cam.get('extrinsics', {}).get('view_matrix', [])
        
        # Convert to a 4x4 numpy array
        matrix_4x4 = np.array(view_matrix_flat).reshape(4, 4)
        extrinsics_list.append(matrix_4x4)

    return np.stack(extrinsics_list)

def extract_camera_matrices_raw(cameras_list):
    matrices = []
    
    for cam in cameras_list:
        raw_matrix = cam.get('intrinsics', {}).get('camera_matrix', [])
        # Convert to numpy and reshape to 3x3
        matrix_3x3 = np.array(raw_matrix).reshape(3, 3)
        matrices.append(matrix_3x3)

    # Stack into a single (n, 3, 3) array
    return np.stack(matrices) if matrices else np.array([])

def extract_distortion_coefficients_raw(cameras_list):
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

def save_calibration_panoptic(output_file, extrinsic_matrices, camera_matrices, distortion_coefficients):
    customized_sequence = []
    num_cameras = extrinsic_matrices.shape[0]

    for i in range(num_cameras):
        # 1. Extract Rotation (3x3) and Translation (3x1)
        # Note: Panoptic format usually expects R and T from the extrinsic matrix
        R = extrinsic_matrices[i, :3, :3].tolist()
        T = extrinsic_matrices[i, :3, 3:4].tolist() # Keeps it as [[x], [y], [z]]

        # 2. Extract Intrinsics
        K = camera_matrices[i]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # 3. Extract Distortion (k1, k2, k3) and (p1, p2)
        # OpenCV format: [k1, k2, p1, p2, k3]
        dist = distortion_coefficients[i]
        k = [[float(dist[0])], [float(dist[1])], [float(dist[4])]]
        p = [[float(dist[2])], [float(dist[3])]]

        # Create camera dictionary
        cam_data = {
            "R": R,
            "T": T,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k": k,
            "p": p
        }
        customized_sequence.append(cam_data)

    output_data = {
        "customized_sequence": customized_sequence
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Successfully saved calibration to {output_file}")