import os
import cv2
import torch
from dotenv import load_dotenv
from configs import *
from model import ProjectionNet
from density import GaussianDensityTorch
from inference import *

# RTSP stream URL
load_dotenv()

# Create a VideoCapture object
CAMERA_1 = os.getenv('CAMERA_1')
cap = cv2.VideoCapture(CAMERA_1)


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


# READING FRAMES
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id+=1
    if frame_id%FRAME_RATE!=0:
        continue
    results = analyze_video_frame(frame, model, density_estimator)
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