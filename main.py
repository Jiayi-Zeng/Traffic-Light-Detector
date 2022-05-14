import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync

# Model
device=''
device = select_device(device)
weights = '/home/lk/yolo/yolov3/runs/train/exp/weights/best.pt'  # or yolov3-spp, yolov3-tiny, custom
model = DetectMultiBackend(weights, device='0', dnn=False)


# Images
img = '/home/lk/yolo/yolov3/data/datasets/Apollo_demo_data/testsets/images/00000.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()
