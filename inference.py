from torchvision import transforms
from PIL import Image
import torch
from model import ProjectionNet
from density import GaussianDensityTorch

# Preprocess the input image
def preprocess_image(image_path, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference on a single image
def infer_image(model, image_path, density_estimator, device="cpu"):
    # Preprocess the image
    image = preprocess_image(image_path).to(device)
    
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
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--cuda', default=False, type=bool, help='Use CUDA for inference (default: False)')
    args = parser.parse_args()

    # Set device
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the checkpoint and inspect its structure
    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print("Checkpoint keys and shapes:")
    for key, value in checkpoint.items():
        print(f"{key}: {value.shape}")

    # Adjust head_layers based on the checkpoint inspection
    head_layers = [512] * 2 + [128]  # Update this based on the checkpoint inspection
    classes = checkpoint["out.weight"].shape[0]

    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Initialize density estimator
    density_estimator = GaussianDensityTorch()

    # Fit the density estimator (skip if not needed)
    print("Skipping density fitting for quick testing...")
    density_estimator.mean = torch.zeros(128)  # Placeholder mean
    density_estimator.inv_cov = torch.eye(128)  # Placeholder covariance matrix

    # Perform inference
    anomaly_score = infer_image(model, args.image_path, density_estimator, device)
    print(f"Anomaly Score: {anomaly_score}")