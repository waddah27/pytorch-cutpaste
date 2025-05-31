import os
import cv2
import torch
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
from configs import *
from model import ProjectionNet
from torch.utils.data import DataLoader
from dataset import MVTecAT
from density import GaussianDensityTorch
from inference import get_train_embeds
# RTSP stream URL
load_dotenv()

# Create a VideoCapture object
CAMERA_1 = os.getenv('CAMERA_1')
cap = cv2.VideoCapture(CAMERA_1)



def preprocess_large_frame(frame, target_size=IMG_SIZE//2):
    """
    Process a single high-res frame (2160x3840x3) for inference
    Args:
        frame: numpy array or PIL Image (H,W,C)
        target_size: model input size (default 256 for CutPaste)
    Returns:
        torch.Tensor (1,3,H,W) normalized and resized
    """
    transform = transforms.Compose([])
    transform.transforms.append(transforms.Resize((target_size, target_size))),
    transform.transforms.append(transforms.ToTensor()),
    transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]))
    
    return transform(frame).unsqueeze(0)  # add batch dim

def process_entire_frame(frame, model, density_estimator):
    """Process full frame at once"""
    inputs = preprocess_large_frame(frame).to(DEVICE)
    with torch.no_grad():
        embed, _ = model(inputs)
    return density_estimator.predict(embed).item()

def process_by_patches(frame, model, density_estimator, patch_size=IMG_SIZE):
    """
    Process frame in overlapping patches and aggregate scores
    Returns:
        max_score: most anomalous patch score
        mean_score: average anomaly level
        anomaly_map: score heatmap (for visualization)
    """
    # Convert to PIL for patch extraction
    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(frame)
    
    # Create grid
    w, h = frame.size
    grid = []
    for y in range(0, h, patch_size//2):  # 50% overlap
        for x in range(0, w, patch_size//2):
            grid.append(frame.crop((x, y, x+patch_size, y+patch_size)))
    
    # Process patches
    scores = []
    for patch in grid:
        inputs = preprocess_large_frame(patch).to(DEVICE)
        with torch.no_grad():
            embed, _ = model(inputs)
        scores.append(density_estimator.predict(embed).item())
    
    # Create anomaly map
    # num_cols = (w // (patch_size//2)) + 1
    # anomaly_map = np.array(scores).reshape(-1, num_cols)
    
    return {
        'max_score': max(scores),
        'mean_score': np.mean(scores),
        # 'anomaly_map': anomaly_map
    }




# Initialize model (as in your original code)
checkpoint_path = MODEL_DIR
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Adjust head_layers based on the checkpoint inspection
head_layers = [IMG_SIZE] * 2 + [128]  #TODO Update this based on the checkpoint inspection
classes = checkpoint["out.weight"].shape[0]

model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()


train_embed = get_train_embeds(model, transform)
train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

density_estimator = GaussianDensityTorch()
# Fit the density estimator (skip if not needed)
if FIT_DENSITY_ESTIMATOR:
    density_estimator.fit(train_embed)
else:
    print("Skipping density fitting for quick testing...")
    density_estimator.mean = torch.zeros(IMG_SIZE)  # Placeholder mean
    density_estimator.inv_cov = torch.eye(IMG_SIZE)  # Placeholder covariance matrix


def analyze_video_frame(frame):
    # For numpy arrays (OpenCV frames)
    if isinstance(frame, np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Choose processing method
    if DEVICE == 'cuda' and torch.cuda.mem_get_info()[0] > 12e9:  # >12GB VRAM
        score = process_entire_frame(frame, model, density_estimator)
        return {'score': score}
    else:
        return process_by_patches(frame, model, density_estimator)

# READING FRAMES
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id+=1
    if frame_id%FRAME_RATE!=0:
        continue
    results = analyze_video_frame(frame)
    print(f"Anomaly score: {results['max_score']:.2f}")
    
    # Visualize if using patch mode
    if False:
        if 'anomaly_map' in results:
            heatmap = cv2.resize(results['anomaly_map'], 
                                (frame.shape[1], frame.shape[0]))
            cv2.imshow('Anomaly Heatmap', heatmap)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()