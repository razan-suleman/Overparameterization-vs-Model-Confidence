import matplotlib.pyplot as plt
import json
import os
import yaml

# Load configuration
project_root = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(project_root, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def load_experiment_results():
    """Load all experiment results from JSON files"""
    results_dir = config['output']['results_dir']
    widths = config['experiment']['widths']
    seeds = config['experiment']['seeds']

    results = {}
    for width in widths:
        width_results = []
        for seed in seeds:
            path = f"{results_dir}/results_w{width}_s{seed}.json"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    result = json.load(f)
                    width_results.append(result)
        if width_results:
            results[width] = width_results

    return results

def aggregate_results(results):
    """Aggregate results by width"""
    widths = []
    accuracies = []
    confidences = []
    eces = []

    for width, width_results in results.items():
        widths.append(width)
        avg_accuracy = sum(r['accuracy'] for r in width_results) / len(width_results)
        avg_confidence = sum(r['confidence'] for r in width_results) / len(width_results)
        avg_ece = sum(r['ece'] for r in width_results) / len(width_results)

        accuracies.append(avg_accuracy)
        confidences.append(avg_confidence)
        eces.append(avg_ece)

    return widths, accuracies, confidences, eces

def plot_results(widths, accuracies, confidences, eces):
    """Create and save plots"""
    plots_dir = config['output']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)

    # Accuracy vs Width
    plt.figure(figsize=(10, 6))
    plt.plot(widths, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Network Width", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Effect of Network Width on Accuracy", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/accuracy_vs_width.png", dpi=300, bbox_inches='tight')
    plt.show()

    
    # Confidence vs Width
    plt.figure(figsize=(10, 6))
    plt.plot(widths, confidences, marker='o', linewidth=2, markersize=8, color='orange')
    plt.xlabel("Network Width", fontsize=12)
    plt.ylabel("Average Confidence", fontsize=12)
    plt.title("Effect of Network Width on Prediction Confidence", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/confidence_vs_width.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ECE vs Width
    plt.figure(figsize=(10, 6))
    plt.plot(widths, eces, marker='o', linewidth=2, markersize=8, color='red')
    plt.xlabel("Network Width", fontsize=12)
    plt.ylabel("Expected Calibration Error (ECE)", fontsize=12)
    plt.title("Effect of Network Width on Calibration", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/ece_vs_width.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Combined plot
    plt.figure(figsize=(12, 8))
    plt.plot(widths, accuracies, label="Accuracy", marker='o', linewidth=2, markersize=8)
    plt.plot(widths, confidences, label="Confidence", marker='s', linewidth=2, markersize=8)
    plt.plot(widths, eces, label="ECE", marker='^', linewidth=2, markersize=8)
    plt.xlabel("Network Width", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Network Width Effects: Accuracy, Confidence, and Calibration", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/combined_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load and plot results
    results = load_experiment_results()
    if results:
        widths, accuracies, confidences, eces = aggregate_results(results)
        plot_results(widths, accuracies, confidences, eces)
        print("Plots saved to", config['output']['plots_dir'])
    else:
        print("No results found. Run training first.")