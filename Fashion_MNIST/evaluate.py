
import torch
import torch.nn.functional as F
import json
from Fashion_MNIST.utils import config, logger

# ----------------------------
# 5. Evaluation
# ----------------------------
def evaluate(model, test_loader):
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
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

def evaluate_confidence(model, test_loader):
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
    logger.info(f"Average confidence: {avg_confidence:.4f}")
    return avg_confidence

def compute_ece(model, data_loader, n_bins=None):
    if n_bins is None:
        n_bins = config['evaluation']['n_bins']

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

    ece_value = ece.item()
    logger.info(f"ECE: {ece_value:.4f}")
    return ece_value

def save_results(results, width, seed):
    """Save evaluation results to JSON"""
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    path = f"{config['output']['results_dir']}/results_w{width}_s{seed}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {path}")

def load_results(width, seed):
    """Load evaluation results from JSON"""
    path = f"{config['output']['results_dir']}/results_w{width}_s{seed}.json"
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # Example usage: evaluate a trained model
    from utils import MLP, get_data_loaders
    train_loader, test_loader = get_data_loaders()
    model = MLP(width=128)  # Example model
    accuracy = evaluate(model, test_loader)
    confidence = evaluate_confidence(model, test_loader)
    ece = compute_ece(model, test_loader)

    results = {
        'accuracy': accuracy,
        'confidence': confidence,
        'ece': ece
    }
    save_results(results, 128, 0)