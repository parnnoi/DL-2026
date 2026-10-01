import sys
import time
import argparse
import logging
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from einops import rearrange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTIM = {"adadelta": torch.optim.Adadelta,
         "adagrad": torch.optim.Adagrad,
         "adam": torch.optim.Adam,
         "nadam": torch.optim.NAdam,
         "sgd": torch.optim.SGD,
         "rmsprop": torch.optim.RMSprop
        }

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def load_cifar10(root="./data", split="train", batch_size=16):
    train = split=="train"
    raw_dataset = torchvision.datasets.CIFAR10(root=root, train=train)
    inputs = rearrange(torch.tensor(raw_dataset.data, dtype=torch.float32, device=device) / 255.0, "b h w c -> b c h w")
    labels = torch.tensor(raw_dataset.targets, dtype=torch.long, device=device)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 4 * 4)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple CNN classification model on CIFAR10 dataset")
    parser.add_argument("--config",
                        dest="config",
                        required=True,
                        type=str,
                        help="Path to configuration file (TOML format)."
                       )
    args = parser.parse_args()
    config_path = args.config

    logger.info(f"Loading configs...")
    config = toml.load(config_path)

    if config["seed"]["manual_seed"] :
        seed = config["seed"]["seed_value"]
        logger.info(f"\tRunning with manual seed [{seed}] an deterministic algorithms...")
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    dataset_root = config["dataset"]["root"]
    batch_size = config["dataset"]["batch_size"]

    logger.info(f"Loading dataset...")
    trainloader = load_cifar10(root=dataset_root, split="train", batch_size=batch_size)
    testloader = load_cifar10(root=dataset_root, split="test", batch_size=batch_size)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    logger.info(f"Defining model")
    model = SimpleCNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optim_class = OPTIM[config["optimizer"]["name"].lower()]
    optimizer = optim_class(params=model.parameters(), **config["optimizer"]["params"])

    num_epochs = config["training"]["n_epochs"]
    logger.info(f"Begin training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        loss = 0.0
        for data in trainloader:
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss += loss.item()
        logger.info(f"\tEpoch: {epoch + 1} | Loss: {loss:.3f}")
        loss = 0.0
    logger.info(f"Training completed")
    logger.info(f"Begin evaluation...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0], data[1]
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Accuracy of the network on the test images: {accuracy:.2f}%")