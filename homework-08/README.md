# Homework 8: Neural Networks and Deep Learning

- Status: Finished
- Instructions: [`cohorts/2026/homework/08-deep-learning/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/08-deep-learning/homework.md)

## Objective

This homework uses PyTorch to classify curly and straight hair images. I followed the fixed 2026 setup: the supplied archive, the existing train and evaluation split, the specified small convolutional model, deterministic CPU training, and a second training phase with augmentation.

## Approach

I downloaded `data.zip` and checked its SHA-256 checksum before extracting it. The archive contains 800 usable training images and 201 held-out evaluation images. The training script is the course-provided `reference_train.py`; I ran it locally in the Homework 8 virtual environment and kept the resulting history files in this folder for verification.

The model has one convolutional layer, max pooling, one hidden layer, and one output logit. I used `BCEWithLogitsLoss`, SGD with learning rate `0.002` and momentum `0.8`, and the exact transforms from the assignment. The augmented phase continued from the baseline model and optimizer instead of starting over.

## Verified results

| Q | Question | Answer |
|---|----------|--------|
| 1 | Loss function for one output logit | `nn.BCEWithLogitsLoss()` |
| 2 | Trainable parameter count | `20,073,473` |
| 3 | Median baseline training accuracy | `0.81` |
| 4 | Population standard deviation of baseline training loss | `0.139` |
| 5 | Mean augmented evaluation loss | `0.575` |
| 6 | Mean of the last five augmented evaluation accuracies | `0.72` |

The unrounded values came from `history_baseline.json` and `history_augmented.json`. The data archive checksum was `9e53453e3017502f22860cb08558f0cdb343e102fb61feab75960616185e7d67`.

## Reproduce

From this directory, create a Python 3.11 virtual environment and install `requirements-cpu.txt`. Then download and verify the fixed archive, extract it as `data/`, and run:

```bash
python reference_train.py --data-dir data --output-dir runs/reference
```

The host used for this run has Python 3.14, so its installed CPU wheels were compatible newer builds rather than the course's Python 3.11 pins. The assignment's tolerances are designed for these minor CPU wheel differences, and the fixed seed and procedure produced the values recorded above. The local virtual environment, downloaded dataset, and generated `runs/` directory are excluded from Git because they can be recreated from the files in this folder.