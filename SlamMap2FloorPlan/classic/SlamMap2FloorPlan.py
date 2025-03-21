import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
import svgwrite
import os
import argparse  # For parsing command-line arguments

def load_map(pgm_path, yaml_path=None):
    """Load the SLAM map and its metadata."""
    map_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise ValueError(f"Failed to load PGM file: {pgm_path}")

    # Default metadata values if YAML is not provided
    resolution = 0.05  # meters per pixel
    origin = [0, 0, 0]  # [x, y, theta]
    occupied_thresh = 0.65
    free_thresh = 0.25

    if yaml_path:
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            resolution = yaml_data.get('resolution', resolution)
            origin = yaml_data.get('origin', origin)
            occupied_thresh = yaml_data.get('occupied_thresh', occupied_thresh)
            free_thresh = yaml_data.get('free_thresh', free_thresh)

    return map_img, resolution, origin, occupied_thresh, free_thresh

def preprocess_map(map_img, occupied_thresh, free_thresh):
    """Binarize and clean the map."""
    prob_map = map_img / 255.0
    binary_map = np.zeros_like(map_img, dtype=np.uint8)
    binary_map[prob_map >= occupied_thresh] = 255  # Walls
    binary_map[prob_map <= free_thresh] = 0        # Free space

    # Apply morphological operations to clean the map
    kernel = np.ones((2, 2), np.uint8)
    cleaned_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
    cleaned_map = cv2.morphologyEx(cleaned_map, cv2.MORPH_OPEN, kernel)
    
    # # Visualize the preprocessed map
    # plt.figure(figsize=(8, 8))
    # plt.imshow(cleaned_map, cmap='gray')
    # plt.title('Preprocessed Map')
    # plt.axis('off')
    # plt.show()
    
    return cleaned_map

def extract_walls(cleaned_map):
    """Extract wall lines using edge detection and Hough Transform."""
    edges = cv2.Canny(cleaned_map, 150, 250, apertureSize=3)
    
    # # Visualize the edges
    # plt.figure(figsize=(8, 8))
    # plt.imshow(edges, cmap='gray')
    # plt.title('Edge Detection')
    # plt.axis('off')
    # plt.show()
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, 
                            minLineLength=20, maxLineGap=50)
    if lines is None:
        return []
    
    # Convert lines to a list of tuples
    line_segments = [tuple(line[0]) for line in lines]
    return line_segments

def lines_to_contours(wall_lines, shape):
    """Convert wall lines to contours and approximate them as polygons."""
    # Draw lines on a blank black image
    line_img = np.zeros(shape, dtype=np.uint8)
    for (x1, y1, x2, y2) in wall_lines:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)

    # Find contours in the line image
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate each contour to a polygon
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 30:  # Skip very small contours
            continue
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        polygons.append(approx.reshape(-1, 2))
    
    return polygons

def save_vector_output(walls, output_path, image_shape):
    """Generate an SVG vector representation of the walls."""
    height, width = image_shape 
    dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')
    for x1, y1, x2, y2 in walls:
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke=svgwrite.rgb(0, 0, 0, '%')))
    dwg.save()

def get_floorplan(pgm_path, yaml_path, output_dir):
    """Main function to generate a floor plan from a SLAM map."""
    # 1. Load the map and metadata
    map_img, resolution, origin, occupied_thresh, free_thresh = load_map(pgm_path, yaml_path)
    
    # 2. Preprocess the map
    cleaned_map = preprocess_map(map_img, occupied_thresh, free_thresh)
    
    # Pathnames to save the results
    image_name = os.path.splitext(os.path.basename(pgm_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_gt_floorplan = os.path.join(output_dir, f'{image_name}_floor_plan_gt.png')
    output_floorplan = os.path.join(output_dir, f'{image_name}_floor_plan.png')
    output_gt_contour_floorplan = os.path.join(output_dir, f'{image_name}_contour_floor_plan_gt.png')
    output_contour_floorplan = os.path.join(output_dir, f'{image_name}_contour_floor_plan.png')
    output_svg = os.path.join(output_dir, f'{image_name}_floor_plan.svg')
    
    # 3. Detect walls
    walls = extract_walls(cleaned_map)
    
    # Save (and Visualize) the gt with extracted walls
    walls_img_gt = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in walls:
        cv2.line(walls_img_gt, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(walls_img_gt, cmap='gray')
    # plt.title('Walls')
    # plt.axis('off')
    # plt.show()
    cv2.imwrite(output_gt_floorplan, walls_img_gt)
    
    # Save (and Visualize)  the gt with extracted walls
    walls_img = np.zeros_like(cleaned_map)
    for x1, y1, x2, y2 in walls:
        cv2.line(walls_img, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(walls_img, cmap='gray')
    # plt.title('Walls')
    # plt.axis('off')
    # plt.show()
    cv2.imwrite(output_floorplan, walls_img)
    
    # Save vector output as SVG
    save_vector_output(walls, output_svg, map_img.shape)
    
    # 4. Convert walls to polygons via contours
    polygons = lines_to_contours(walls, map_img.shape)
    
    # Save (and Visualize) the gt with polygons
    polygon_img_gt = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    for poly in polygons:
        pts = poly.reshape((-1, 1, 2))
        cv2.polylines(polygon_img_gt, [pts], True, 255, 1)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(polygon_img_gt, cmap='gray')
    # plt.title('Walls using Contours')
    # plt.axis('off')
    # plt.show()
    cv2.imwrite(output_gt_contour_floorplan, polygon_img_gt)
    
    # Save (and Visualize) the polygons
    polygon_img = np.zeros_like(map_img)
    for poly in polygons:
        pts = poly.reshape((-1, 1, 2))
        cv2.polylines(polygon_img, [pts], True, 255, 1)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(polygon_img, cmap='gray')
    # plt.title('Walls using Contours')
    # plt.axis('off')
    # plt.show()
    cv2.imwrite(output_contour_floorplan, polygon_img)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a floor plan from a SLAM map.")
    parser.add_argument("--pgm_path", type=str, required=True, help="Path to the PGM file (SLAM map).")
    parser.add_argument("--yaml_path", type=str, default="../data/room1.yaml", help="Path to the YAML file (metadata).")
    parser.add_argument("--output_dir", type=str, default="output_floorplans", help="Directory to save the output files.")
    args = parser.parse_args()
    
    # Run the floor plan generation
    get_floorplan(args.pgm_path, args.yaml_path, args.output_dir)