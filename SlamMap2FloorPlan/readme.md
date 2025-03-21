
# Floor Plan Reconstruction

In this repo I am exploring [classical](/SlamMap2FloorPlan/classic) and [deep learning](/SlamMap2FloorPlan/RoomFormer) based computer vision approaches to reconstruct floor plans from 3D SLAM maps.

## Directory Structure

```bash
SlamMap2FloorPlan
├── classic
│   ├── output_floorplans
│   ├── readme.md
│   └── SlamMap2FloorPlan.py
├── data
│   ├── room1.jpg
│   ├── room1.pgm
│   ├── room1_solution.png
│   ├── room1.yaml
│   ├── room2.pgm
│   └── room3.pgm
├── readme.md
├── RoomFormer
│   ├── checkpoints
│   ├── data_preprocess
│   ├── datasets
│   ├── detectron2
│   ├── diff_ras
│   ├── engine.py
│   ├── eval.py
│   ├── imgs
│   ├── inference.py
│   ├── LICENSE
│   ├── main.py
│   ├── models
│   ├── __pycache__
│   ├── README.md
│   ├── requirements.txt
│   ├── s3d_floorplan_eval
│   ├── scenecad_eval
│   ├── solution
│   ├── stru3d
│   ├── tools
│   └── util
└── Slam_Challenge.pdf
```
