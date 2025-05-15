import torch
from torchvision import transforms
from .resnet9 import ResNet9

def get_default_device():
    """Always return CPU device"""
    return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def load_model(device=None):
    """Load the trained model"""
    if device is None:
        device = get_default_device()
    
    # Initialize model
    model = ResNet9(num_classes=38)
    
    # Load state dict
    model_path = "/home/vinit-soni/PLANTCARE_PROJECT/app/models/model.pt"  # Update with your actual path
    model = torch.jit.load(model_path, map_location=device)
    # Move to device and set eval mode
    model = to_device(model, device)
    model.eval()
    
    return model

def get_transforms():
    """Get image transformations matching training"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])