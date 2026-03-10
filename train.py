import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------
# 1. Data
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

# 2. Model
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, width),
            nn.ReLU(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.net(x)



# 3. Loss and optimizer


# 4. Training loop
def train(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1)  # flatten [B,1,28,28] -> [B,784]

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 5. Evaluation
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

import torch.nn.functional as F

def evaluate_confidence(model):

    model.eval()

    all_confidences = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.view(images.size(0), -1)

            outputs = model(images)

            probs = F.softmax(outputs, dim=1)

            confidence = torch.max(probs, dim=1).values

            all_confidences.extend(confidence.tolist())

    avg_confidence = sum(all_confidences) / len(all_confidences)

    print("Average confidence:", avg_confidence)


def compute_ece(model, data_loader, n_bins=10):
    model.eval()

    all_confidences = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(images.size(0), -1)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            confidences, predictions = torch.max(probs, dim=1)

            all_confidences.append(confidences)
            all_predictions.append(predictions)
            all_labels.append(labels)

    all_confidences = torch.cat(all_confidences)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    accuracies = all_predictions.eq(all_labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = all_confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

for width in [32, 128, 256, 512, 1024]:
    print(f"Training MLP with width {width}...")
    sum_ece = 0.0
    for seed in range(3):
        model = MLP(width)
        train(model)
        evaluate(model)
        evaluate_confidence(model)
        ece = compute_ece(model, test_loader)
        print(f"ECE: {ece:.4f}")
        sum_ece += ece

    print(f"Average ECE for width {width}: {sum_ece/3:.4f}")

import torch
import torch.nn.functional as F
