# Overparameterization vs Model Confidence

## Overview

Neural networks often produce **overconfident predictions**, even when they are incorrect. This project investigates how **increasing network width (overparameterization)** affects prediction confidence and calibration in a simple neural classifier.

The goal is to understand whether larger neural networks become **more confident than their accuracy justifies**, leading to **miscalibrated probability estimates**.

## Research Question
How does increasing the width of a neural network affect:
* prediction accuracy
* average confidence
* Expected Calibration Error (ECE)

Specifically: Do wider networks become more overconfident?

## Datasets

Two datasets were used to test whether the observed behavior is consistent across tasks.

### MNIST
* 60,000 training images
* 10,000 test images
* 28 × 28 grayscale images
* 10 digit classes

### Fashion-MNIST
* Same format as MNIST
* 28 × 28 grayscale images
* Clothing classification dataset
* Slightly more complex than MNIST


## Model Architecture

A simple fully connected neural network (MLP):

```
Input (784)
↓
Linear (784 → width)
↓
ReLU
↓
Linear (width → 10)
```

Widths tested: 32, 128, 256, 512, 1024

## Training Configuration

* **Optimizer:** Adam
* **Loss Function:** Cross-Entropy
* **Epochs:** 5
* **Batch Size:** 128

Each width was trained **three times with different random seeds** to reduce randomness in the results.

## Evaluation Metrics

### Accuracy

Standard classification accuracy on the test set.

### Average Confidence
Average maximum predicted probability: confidence = max(p_i)
This measures how confident the model is in its predictions.

### Expected Calibration Error (ECE)

ECE measures how well predicted probabilities match actual correctness.
A perfectly calibrated model would satisfy:
confidence ≈ accuracy

Higher ECE indicates **worse calibration**.

## Results

### MNIST – Average ECE

| Width | Average ECE |
| ----- | ----------- |
| 32    | 0.0064      |
| 128   | 0.0031      |
| 256   | 0.0046      |
| 512   | 0.0057      |
| 1024  | 0.0072      |

---

### Fashion-MNIST – Example Results

| Width | ECE    |
| ----- | ------ |
| 32    | 0.0059 |
| 128   | 0.0019 |
| 256   | 0.0040 |
| 512   | 0.0032 |
| 1024  | 0.0093 |

---

## Observed Trends

Across both datasets:

* **Accuracy improves initially** as model width increases but quickly saturates.
* **Prediction confidence steadily increases** with network size.
* **Calibration improves at moderate widths but worsens again for very wide networks.**

This suggests three regimes:

### Small Models (width = 32)

Models **underfit the data**, producing uncertain predictions and higher calibration error.

### Medium Models (width ≈ 128)

Models achieve the best balance between **accuracy and confidence**, resulting in the **lowest calibration error**.

### Large Models (width ≥ 256)

Prediction confidence continues to increase even when accuracy stops improving, leading to **overconfidence** and increasing calibration error.

---

## Key Insight

Increasing network width makes neural networks **more confident**, but not necessarily **more accurate**.

As a result, highly overparameterized models can become **miscalibrated**, producing probability estimates that are overly confident relative to their true accuracy.

---

## Visualizations

The experiment generates plots showing:

* Network width vs accuracy
* Network width vs average confidence
* Network width vs Expected Calibration Error (ECE)

These plots illustrate how prediction confidence grows faster than accuracy as model capacity increases.

---

## How to Run

### Clone the repository

```
git clone <repo_url>
cd Overparameterization-vs-Model-Confidence
```

### Install dependencies

```
pip install torch torchvision matplotlib numpy
```

### Run the experiment

```
python train.py
```

## Conclusion

This project demonstrates that **overparameterization can increase prediction confidence faster than it improves accuracy**, leading to **growing miscalibration in wide neural networks**.

Understanding this behavior is important when building machine learning systems that must provide **reliable probability estimates**, especially in safety-critical applications.

