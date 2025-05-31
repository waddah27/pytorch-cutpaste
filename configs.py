import os
import torch
from torchvision import transforms

# from pathlib import Path
# ROOT_DATA_DIR = Path(__file__).parent / "Data"
ROOT_DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "model-carpet-2025-04-25_01_15_21.tch")
IMG_SIZE = 512
EMBED_SAMPLE_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFECT_TYPE = "carpet"
DATA_MODE = 'train'
BATCH_SIZE = 64

transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Initialize density estimator

FIT_DENSITY_ESTIMATOR = False
THRESHOLD = 210 # TODO this is for now empirical value, try to find better estimation to the thresh..

FRAME_RATE = 30
