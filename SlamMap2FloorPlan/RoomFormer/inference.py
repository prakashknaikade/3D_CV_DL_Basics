import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from shapely.geometry import Polygon
from models import build_model
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan
import os

def outputs_to_polygons(outputs, threshold=0.5):
    """
    Convert model outputs to a list of room polygons.

    Args:
        outputs (dict): Model output containing 'pred_logits' and 'pred_coords'.
        threshold (float): Confidence threshold for corner predictions (default: 0.5).

    Returns:
        list: List of polygons, each as a numpy array of [x, y] coordinates.
    """
    pred_logits = outputs['pred_logits']  # Shape: [1, num_queries, max_corners]
    pred_corners = outputs['pred_coords']  # Shape: [1, num_queries, max_corners, 2]
    fg_mask = torch.sigmoid(pred_logits) > threshold  # Shape: [1, num_queries, max_corners]

    polygons = []
    for j in range(pred_corners.shape[1]):  # Iterate over queries (potential rooms)
        mask = fg_mask[0, j]  # Shape: [max_corners]
        corners = pred_corners[0, j][mask]  # Shape: [num_valid, 2]
        if len(corners) >= 4:  # Minimum corners for a polygon
            corners = corners.cpu().numpy() * 255  # Scale from [0, 1] to [0, 255]
            corners = np.round(corners).astype(int)
            if Polygon(corners).area >= 100:  # Filter by area
                polygons.append(corners)
    return polygons

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description="RoomFormer inference on a single density image")
    parser.add_argument('--input_image', type=str, required=True,
                        help="Path to the 256x256 grayscale density image")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/roomformer_scenecad.pth',
                        help="Path to the pre-trained RoomFormer checkpoint")
    parser.add_argument('--output_dir', type=str, default='solution',
                        help="Path to save the output visualization")
    args = parser.parse_args()

    # Model configuration (hardcoded to match typical RoomFormer defaults)
    model_args = argparse.Namespace(
        backbone='resnet50',
        position_embedding='sine',
        num_feature_levels=4,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        hidden_dim=256,
        nheads=8,
        num_queries=800,
        num_polys=20,
        dec_n_points=4,
        enc_n_points=4,
        query_pos_type='sine',
        with_poly_refine=True,
        masked_attn=False,
        semantic_classes=-1,  # Non-semantic floorplan
        # Additional args to satisfy build_model, set to defaults
        lr_backbone=0,
        dilation=False,
        position_embedding_scale=2 * np.pi,
        dropout=0.1,
        aux_loss=False,
        device='cuda',
        num_workers=2,
        seed=42,
    )

    # output_dir='solution' 
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = build_model(model_args, train=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    # Load and preprocess the input image
    density_image = Image.open(args.input_image).convert('L')  # Grayscale
    
    # Default metadata values if YAML is not provided
    resolution = 0.05  # meters per pixel
    origin = [0, 0, 0]  # [x, y, theta]
    occupied_thresh = 0.65
    free_thresh = 0.25
    
    if density_image.size != (256, 256):
        map_img = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
        # Resize the image to 256x256
        resized_img = cv2.resize(map_img, (256, 256), interpolation=cv2.INTER_AREA)
        # Convert to probability map (0 to 1)
        prob_map = resized_img / 255.0
        binary_map = np.zeros_like(resized_img, dtype=np.uint8)
        binary_map[prob_map >= occupied_thresh] = 0  # Walls
        binary_map[prob_map <= free_thresh] = 255    # Free space

        # Clean up noise using morphological operations
        # kernel = np.ones((1, 1), np.uint8)
        # cleaned_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        # cleaned_map = cv2.morphologyEx(cleaned_map, cv2.MORPH_OPEN, kernel)
        # cleaned_map = cv2.Canny(cleaned_map, 15, 50, apertureSize=3)
        
        density_image = binary_map    
    transform = transforms.ToTensor()  # Converts to [0, 1], shape: [1, 256, 256]
    input_tensor = transform(density_image).unsqueeze(0).to(device)  # Shape: [1, 1, 256, 256]

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Extract polygons
    polygons = outputs_to_polygons(outputs)

    # Visualize results
    room_polys = [np.array(r) for r in polygons]
    floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
    image_name = os.path.splitext(os.path.basename(args.input_image))[0]
    cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_pred_floorplan.png'), floorplan_map)
                   
    # Overlay polygons on density map
    density_map = np.array(density_image)[:, :, np.newaxis]  # Shape: [256, 256, 1]
    density_map = np.repeat(density_map, 3, axis=2)  # Shape: [256, 256, 3]
    pred_room_map = np.zeros_like(density_map)
    for room_poly in polygons:
        pred_room_map = plot_room_map(room_poly, pred_room_map)
    pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
    # Save the visualization
    cv2.imwrite(os.path.join(args.output_dir, f'{image_name}_pred_room_map.png'), pred_room_map) 
    print(f"Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()