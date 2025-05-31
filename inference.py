
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset import MVTecAT
from model import ProjectionNet
from density import GaussianDensityTorch
from configs import *


def get_train_embeds(model, transform):
    # train data / train kde
    test_data = MVTecAT(ROOT_DATA_DIR, DEFECT_TYPE, EMBED_SAMPLE_SIZE, transform=transform, mode=DATA_MODE)

    dataloader_train = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    train_embed = []
    with torch.no_grad():
        for x in dataloader_train:
            embed, logit = model(x.to(DEVICE))

            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference on a single image
def infer_image(model, image_path):
    # Preprocess the image
    image = preprocess_image(image_path).to(DEVICE)
    
    # Get embeddings from the model
    with torch.no_grad():
        embed, logit = model(image)
    
    # Normalize embeddings
    embed = torch.nn.functional.normalize(embed, p=2, dim=1)
    
    # Use the density estimator to compute anomaly score
    anomaly_score = density_estimator.predict(embed)
    
    return anomaly_score.cpu().numpy()

# Main script
if __name__ == '__main__':
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Inference on a single image')
    parser.add_argument('--image_path', type=str, default="/home/segura/pytorch-cutpaste/Data/carpet/train/good/001.png",  help='Path to the input image')
    parser.add_argument('--model_path', type=str, default="models/model-carpet-2025-04-25_01_15_21.tch", help='Path to the trained model weights')
    parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA for inference (default: False)')
    args = parser.parse_args()
    print(f"Using device: {DEVICE}")

    # Load the checkpoint and inspect its structure
    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # print("Checkpoint keys and shapes:")
    # for key, value in checkpoint.items():
        # print(f"{key}: {value.shape}")

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

    # Perform inference
    anomaly_score = infer_image(model, args.image_path)
    print(f"Anomaly Score: {anomaly_score}")
    if anomaly_score > THRESHOLD:
        print("GOOD")
    else:
        print("ANOMALY DETECTED!")