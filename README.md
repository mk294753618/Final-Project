# Final-Project
  This is the final project for the Mathematical Foundations of Data Science course at BUPT
  
# Introduction
  In this project, we replicated the YOLOv3-SPP model to track characters from the online game "Valorant." Although YOLOv3 is not the most advanced object detection algorithm, we chose it because of its ease of replication and       practicality. Our objective was to leverage this model's capability to identify and follow the movements of various in-game characters. This implementation demonstrates the practical application of YOLOv3-SPP in dynamic environments and its utility potential in the field of gaming analytics.

# Environment
  NVIDIA GeForce RTX 3070  
  CUDA Version: 12.4  
  python 3.9  
  third-party library requirements are in 'final_project/requirements.txt'  

# File Description
    │  detect.py
    │  detectui.py
    │  detectui.ui
    │  evaluate.py
    │  models.py
    │  requirements.txt
    │  train.py
    │  __init__.py
    │
    ├─checkpoints
    ├─config
    │      yolov3.cfg
    │      yolov3_spp.cfg
    │
    ├─data
    │  │  train.txt
    │  │  valid.txt
    │  │  valorant.data
    │  │  valorant.names
    │  │
    │  ├─train_image
    │  │
    │  ├─train_label
    │  │
    │  ├─valid_image
    │  │
    │  └─valid_label
    │
    ├─logs
    │
    ├─output
    ├─samples
    │
    └─utils
        data.py
        logger.py
        loss.py
        parse_config.py
        transforms.py
        utils.py
        __init__.py

        
