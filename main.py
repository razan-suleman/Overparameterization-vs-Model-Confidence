#!/usr/bin/env python3
"""
Main script for running overparameterization vs model confidence experiments.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Run overparameterization vs model confidence experiments")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion_mnist"],
                       help="Dataset to use (overrides config)")
    parser.add_argument("--train", action="store_true",
                       help="Run training experiments")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation on saved models")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots from results")
    parser.add_argument("--all", action="store_true",
                       help="Run training, evaluation, and plotting")

    args = parser.parse_args()

    # Update config if dataset specified
    if args.dataset:
        import yaml
        config_path = args.config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config['experiment']['dataset'] = args.dataset
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    # Determine what to run
    if args.all:
        args.train = args.evaluate = args.plot = True
    elif not any([args.train, args.evaluate, args.plot]):
        # Default: run everything
        args.train = args.evaluate = args.plot = True

    # Run training
    if args.train:
        print("Running training experiments...")
        if args.dataset == "fashion_mnist" or (not args.dataset and config.get('experiment', {}).get('dataset') == 'fashion_mnist'):
            from Fashion_MNIST.train import run_experiment
            dataset_module = "Fashion_MNIST"
        else:
            from MNIST.train import run_experiment
            dataset_module = "MNIST"

        # Import config to get parameters
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        all_results = []
        for width in config['experiment']['widths']:
            for seed in config['experiment']['seeds']:
                result = run_experiment(width, seed)
                all_results.append(result)

    # Run evaluation (if not already done during training)
    if args.evaluate and not args.train:
        print("Running evaluation...")
        # This would load saved models and evaluate them
        # For now, evaluation is integrated with training

    # Generate plots
    if args.plot:
        print("Generating plots...")
        if args.dataset == "fashion_mnist" or (not args.dataset and config.get('experiment', {}).get('dataset') == 'fashion_mnist'):
            from Fashion_MNIST.plot_results import main as plot_main
        else:
            from MNIST.plot_results import main as plot_main
        plot_main()

if __name__ == "__main__":
    main()