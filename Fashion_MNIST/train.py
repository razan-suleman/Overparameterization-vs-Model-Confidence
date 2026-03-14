
import torch.nn as nn
import torch.optim as optim
from utils import get_data_loaders, MLP, set_seed, save_model, config, logger
from evaluate import evaluate, evaluate_confidence, compute_ece, save_results

# 3. Training
def train_model(model, train_loader, epochs=None, lr=None):
    if epochs is None:
        epochs = config['training']['epochs']
    if lr is None:
        lr = config['training']['learning_rate']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def run_experiment(width, seed):
    """Run a single experiment with given width and seed"""
    logger.info(f"Starting experiment: width={width}, seed={seed}")

    set_seed(seed)

    # Get data
    train_loader, test_loader = get_data_loaders()

    # Create and train model
    model = MLP(width)
    train_model(model, train_loader)

    # Save trained model
    save_model(model, width, seed)

    # Evaluate model
    accuracy = evaluate(model, test_loader)
    confidence = evaluate_confidence(model, test_loader)
    ece = compute_ece(model, test_loader)

    # results
    results = {
        'width': width,
        'seed': seed,
        'accuracy': accuracy,
        'confidence': confidence,
        'ece': ece
    }
    save_results(results, width, seed)

    return results

if __name__ == "__main__":
    all_results = []

    for width in config['experiment']['widths']:
        width_results = []
        for seed in config['experiment']['seeds']:
            result = run_experiment(width, seed)
            width_results.append(result)

        avg_accuracy = sum(r['accuracy'] for r in width_results) / len(width_results)
        avg_confidence = sum(r['confidence'] for r in width_results) / len(width_results)
        avg_ece = sum(r['ece'] for r in width_results) / len(width_results)

        logger.info(f"Average for width {width}: Accuracy={avg_accuracy:.4f}, Confidence={avg_confidence:.4f}, ECE={avg_ece:.4f}")

        all_results.extend(width_results)
