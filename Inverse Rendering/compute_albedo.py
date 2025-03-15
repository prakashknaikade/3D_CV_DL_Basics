import json
import numpy as np
import cv2
import OpenEXR
import Imath
import os
from tqdm import tqdm 

def read_exr_image(filepath): # Read an EXR image and return it as a NumPy array
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    data = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in channels if c in header['channels']]

    if not data:
        raise ValueError(f"No valid channels found in {filepath}")

    image = np.stack(data, axis=-1).reshape(size[1], size[0], len(data))
    return image


def parse_calibration(calibration_file):
    # Open and load the calibration JSON file
    with open(calibration_file, 'r') as f:
        data = json.load(f)
    cameras, lights = {}, {}
    
    # Parse camera data
    for cam in data['cameras']:
        cam_id = cam['camera_id']
        cameras[cam_id] = {
            'intrinsics': {
                'width': int(cam['width']),
                'height': int(cam['height']),
                'focal_length': float(cam['focal_px']),
            },
            'extrinsics': {
                # Convert rotation matrix to NumPy array and reshape
                'R': np.array([float(val) for val in cam['rig_extrinsics_R']]).reshape(3, 3),
                # Convert translation vector to NumPy array and reshape
                't': np.array([float(val) for val in cam['rig_extrinsics_t']]).reshape(3, 1),
            },
        }
        
    # Parse light data
    for light in data['lights']:
        light_id = light['light_id']
        lights[light_id] = {
            'extrinsics': {
                # Convert rotation matrix to NumPy array and reshape
                'R': np.array([float(val) for val in light['rig_extrinsics_R']]).reshape(3, 3),
                # Convert translation vector to NumPy array and reshape
                't': np.array([float(val) for val in light['rig_extrinsics_t']]).reshape(3, 1),
            },
            'properties': {
                'type': light['type'],
                'diameter': float(light.get('diameter', 0)),
                'radiance': float(light.get('radiance', 0)),
                'power': float(light.get('power', 0)),
            },
        }
    return cameras, lights

def compute_albedo(calibration_file, images_dir, output_dir):
    # Parse the calibration data
    cameras, lights = parse_calibration(calibration_file)

    # Iterate over each camera
    for cam_id, cam_data in cameras.items():
        print(f"Processing Camera {cam_id}...")
        
        # Extract Camera parameters
        intrinsics = cam_data['intrinsics']
        extrinsics = cam_data['extrinsics']
        
        # Construct the camera intrinsic matrix K
        K = np.array([
            [intrinsics['focal_length'], 0, intrinsics['width'] / 2],
            [0, intrinsics['focal_length'], intrinsics['height'] / 2],
            [0, 0, 1],
        ])
        R_cam = extrinsics['R']
        t_cam = extrinsics['t']

        # paths to normal map and mask
        sensor_name = f'sensor_{cam_id.zfill(3)}'
        sensor_dir = os.path.join(images_dir, sensor_name)
        normal_map_path = os.path.join(sensor_dir, 'all_normal.exr')
        mask_path = os.path.join(sensor_dir, 'all_mask.exr')
        # Load normal map and mask
        try:
            normal_map = read_exr_image(normal_map_path)
            mask = read_exr_image(mask_path)[..., 0] # Use the first channel if multiple exist
            mask = (mask > 0).astype(np.float32)     # Convert to binary mask
        except Exception as e:
            print(f"Error loading normal map or mask for camera {cam_id}: {e}")
            continue

        h, w = normal_map.shape[:2] # image dimensions
        num_lights = len(lights)
        num_channels = 3  # RGB channels
        
        I = np.zeros((h * w, num_lights, num_channels)) # Intensities per pixel, light, and channel
        L = np.zeros((h * w, num_lights, 3))            # Light directions per pixel and light

        # Prepare pixel coordinates (flattened)
        ys, xs = np.indices((h, w))
        xs = xs.flatten()
        ys = ys.flatten()

        # Camera rays in camera coordinates
        pixel_coords = np.vstack((xs, ys, np.ones_like(xs))) # Homogeneous coordinates, 3xN array where each column is [x_i, y_i, 1]
        K_inv = np.linalg.inv(K)                             # Inverse of the intrinsic matrix
        cam_rays = K_inv @ pixel_coords  # Shape: (3, N)     # Ray directions in camera coordinates

        # Transform rays to world coordinates
        cam_rays_world = R_cam.T @ cam_rays  # Shape: (3, N)

        # Camera position in world coordinates
        cam_pos_world = -R_cam.T @ t_cam  # X_world ​= R_cam⊤ ​(X_camera ​− t_cam​) , Shape: (3, 1) 

        # Ray-plane intersection at y = 0.25m
        plane_y = 0.25 
        ry = cam_rays_world[1, :]
        py = cam_pos_world[1, 0]
        t_vals = (plane_y - py) / ry # Compute t where the ray intersects the plane
        
        # Identify valid t values (intersection in front of the camera)
        valid_t = (t_vals > 0) # & np.isfinite(t_vals) 
        t_vals[~valid_t] = np.inf  # Ignore rays that do not intersect plane

        # Compute world coordinates of the intersection points
        world_coords = cam_pos_world + cam_rays_world * t_vals  # Shape: (3, N)

        # Process each light
        for idx, (light_id, light_data) in enumerate(lights.items()):
            # Load HDR image
            image_path = os.path.join(sensor_dir, f'light_{light_id.zfill(3)}.exr')
            try:
                hdr_image = read_exr_image(image_path)
                # Reshape the image and apply the mask
                intensity = hdr_image.reshape(-1, num_channels) * mask.flatten()[:, np.newaxis]  # Shape: (N, 3)
                I[:, idx, :] = intensity # Store intensity values
            except Exception as e:
                print(f"Error loading HDR image for light {light_id}: {e}")
                continue

            # Light position in world coordinates
            R_light = light_data['extrinsics']['R']
            t_light = light_data['extrinsics']['t']
            light_pos_world = -R_light.T @ t_light  # Shape: (3, 1)

            # Compute light directions from surface points to the light source
            light_dirs = light_pos_world - world_coords  # Shape: (3, N)
            norms = np.linalg.norm(light_dirs, axis=0)
            norms[norms == 0] = 1e-6  # Avoid division by zero
            light_dirs_normalized = (light_dirs / norms)  # Shape: (3, N)
            L[:, idx, :] = light_dirs_normalized.T  # Shape: (N, 3)

        # Reshape normal map and normalize the normal map
        N = normal_map.reshape(-1, 3)
        N_norms = np.linalg.norm(N, axis=1, keepdims=True)
        N_norms[N_norms == 0] = 1e-6
        N_normalized = N / N_norms

        # Prepare the mask and identify valid pixels
        mask_flat = mask.flatten()
        valid_pixels = mask_flat > 0

        # Photometric stereo for each color channel I_i = rho * (L_i * N)
        albedo = np.zeros((h * w, num_channels))
        V = np.einsum('ijk,ik->ij', L[valid_pixels], N_normalized[valid_pixels])  # V_ij = (L_ij . N_i) Shape: (num_valid_pixels, num_lights) 
        
        # # with broadcassting
        # # N_normalized[valid_pixels]: Shape (num_valid_pixels, 1, 3)
        # N_expanded = N_normalized[valid_pixels][:, np.newaxis, :]  # Shape: (num_valid_pixels, 1, 3)

        # # L[valid_pixels]: Shape (num_valid_pixels, num_lights, 3)
        # # Compute element-wise multiplication and sum over axis 2
        # V = np.sum(L[valid_pixels] * N_expanded, axis=2)  # Shape: (num_valid_pixels, num_lights)

        I_valid = I[valid_pixels]  # Shape: (num_valid_pixels, num_lights, 3)

        # Handle specular highlights
        INTENSITY_THRESHOLD = np.percentile(I_valid[I_valid > 0], 90)  # Threshold at the 90th percentile

        albedo_valid = np.zeros((np.sum(valid_pixels), num_channels))  # Albedo values for valid pixels

        for c in range(num_channels):  # Loop over color channels
            print(f"Processing channel {c}...")
            I_c = I_valid[..., c]  # Intensity for channel c
            albedo_c = np.zeros(np.sum(valid_pixels))
            
            # Loop over each valid pixel
            for i in tqdm(range(V.shape[0]), desc=f'Computing albedo for camera {cam_id}, channel {c}'):
                V_i = V[i] # Dot products for pixel i
                I_i = I_c[i] # Intensity values for pixel i
                
                # Identify valid observations (exclude zeros and specular highlights)
                valid_obs = (V_i > 0) & (I_i > 0) & (I_i < INTENSITY_THRESHOLD)
                
                if np.count_nonzero(valid_obs) >= 3:
                    # Use least squares to solve for albedo
                    V_i = V_i[valid_obs] 
                    I_i = I_i[valid_obs]
                    rho, _, _, _ = np.linalg.lstsq(V_i[:, np.newaxis], I_i, rcond=None)
                    albedo_c[i] = np.clip(rho[0], 0, 1)
                else:
                    albedo_c[i] = 0  # Insufficient data

            albedo_valid[:, c] = albedo_c  # Store the albedo values for channel c

        # Assign albedo to full map
        albedo[valid_pixels] = albedo_valid
        albedo_map = albedo.reshape(h, w, num_channels)

        # Save the albedo map as an image
        os.makedirs(output_dir, exist_ok=True)
        albedo_filename = os.path.join(output_dir, f'camera_{cam_id}_albedo.png')
        # Convert albedo map to 8-bit RGB image
        albedo_map_uint8 = np.clip(albedo_map * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(albedo_filename, cv2.cvtColor(albedo_map_uint8, cv2.COLOR_RGB2BGR))
        print(f"Albedo map saved for camera {cam_id} at {albedo_filename}")


if __name__ == '__main__':
    calibration_file = 'data/configuration/calibration.json'
    images_dir = 'data/images/HDR'
    output_dir = 'output_Threshold'

    compute_albedo(calibration_file, images_dir, output_dir)