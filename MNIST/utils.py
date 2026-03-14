import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import os
import logging

# Load configuration
project_root = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(project_root, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Make paths absolute
config['data']['root'] = os.path.join(project_root, config['data']['root'].lstrip('./'))
config['output']['models_dir'] = os.path.join(project_root, config['output']['models_dir'].lstrip('./'))
config['output']['results_dir'] = os.path.join(project_root, config['output']['results_dir'].lstrip('./'))
config['output']['plots_dir'] = os.path.join(project_root, config['output']['plots_dir'].lstrip('./'))
config['output']['log_file'] = os.path.join(project_root, config['output']['log_file'].lstrip('./'))

# Ensure output directories exist
os.makedirs(config['output']['models_dir'], exist_ok=True)
os.makedirs(config['output']['results_dir'], exist_ok=True)
os.makedirs(config['output']['plots_dir'], exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config['output']['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# 1. Data
# ----------------------------
def get_data_loaders(batch_size=None, dataset=None):
    if batch_size is None:
        batch_size = config['training']['batch_size']
    if dataset is None:
        # Reload config to get current value
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, "config.yaml")
        with open(config_path, "r") as f:
            current_config = yaml.safe_load(f)
        dataset = current_config['experiment']['dataset']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['data']['normalize_mean'], config['data']['normalize_std'])
    ])

    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            root=config['data']['root'],
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.MNIST(
            root=config['data']['root'],
            train=False,
            download=True,
            transform=transform
        )
    elif dataset == "fashion_mnist":
        train_dataset = datasets.FashionMNIST(
            root=config['data']['root'],
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=config['data']['root'],
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    logger.info(f"Loaded {dataset} dataset with {len(train_dataset)} train and {len(test_dataset)} test samples")
    return train_loader, test_loader

# ----------------------------
# 2. Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['model']['input_size'], width),
            nn.ReLU(),
            nn.Linear(width, config['model']['output_size'])
        )

    def forward(self, x):
        return self.net(x)

def save_model(model, width, seed, epoch=None):
    """Save model to disk"""
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    if epoch is not None:
        path = f"{config['output']['models_dir']}/model_w{width}_s{seed}_e{epoch}.pth"
    else:
        path = f"{config['output']['models_dir']}/model_w{width}_s{seed}.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"Saved model to {path}")

def load_model(width, seed, epoch=None):
    """Load model from disk"""
    if epoch is not None:
        path = f"{config['output']['models_dir']}/model_w{width}_s{seed}_e{epoch}.pth"
    else:
        path = f"{config['output']['models_dir']}/model_w{width}_s{seed}.pth"
    model = MLP(width)
    model.load_state_dict(torch.load(path))
    logger.info(f"Loaded model from {path}")
    return model