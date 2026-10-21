# Neural Networks – Task 1: Single Layer Perceptron & Adaline

A Streamlit app that trains and tests a **Single Layer Perceptron (SLP)** and an **Adaline** on the Penguins dataset. You choose two features and two species, set the training parameters, and the app shows the decision boundary, the confusion matrix and the test accuracy. It can also classify a single sample you type in.

## Project structure

```
.
├── app.py             # Streamlit GUI + the glue that connects it to the algorithms
├── preprocessing.py   # Loading, missing values, encoding, train/test split, scaling
├── SLP.py             # Perceptron class (train / test)
├── adaline.py         # Adaline training and testing functions
├── penguins.csv       # Dataset (150 rows: 50 per species)
├── brain_logo.png     # Logo used by the GUI
└── .gitignore
```

## Setup

Requires Python 3.9 or newer.

```bash
pip install streamlit numpy pandas matplotlib pillow
```

## Run

From the project folder (the one containing `app.py`):

```bash
streamlit run app.py
```

## How to use

1. Pick an algorithm on the landing page (Single Layer Perceptron or Adaline).
2. In the sidebar, choose **Feature 1**, **Feature 2**, **Class 1** and **Class 2**. The two features and the two classes must be different.
3. Set the **learning rate (eta)** and the **number of epochs**. For Adaline, also set the **MSE threshold**. Tick or untick **Add bias**.
4. Press **Train & Test**.
5. The results page shows the test accuracy, the decision boundary (train points as circles, test points as crosses), and the confusion matrix on the test set.
6. To classify a new penguin, enter the two feature values in **original units** under *Classify a single sample* and press **Classify**.

## Data handling (`preprocessing.py`)

- **Split:** for each chosen class, 30 samples are picked at random for training and the other 20 for testing (60 train / 40 test in total, no repeated samples).
- **Missing values:** rows can't be dropped, so missing numeric values are filled with the mean of the same species. A missing `OriginLocation` would be filled with the most common value.
- **OriginLocation:** encoded as numbers in alphabetical order of the location names. This order is arbitrary, so the numeric codes have no real meaning. To classify a sample with this feature, type the code number.
- **Scaling:** features are standardized (z-score) using the training set's mean and standard deviation, so a large feature such as `BodyMass` does not dominate a small one such as `CulmenDepth`.
- **Labels:** the algorithms use +1 and -1. In the GUI, Class 1 is -1 and Class 2 is +1.

## Algorithms

### Single Layer Perceptron (`SLP.py`)

- Net value `v = w · x`, with the bias treated as `w0` and `x0 = 1` (when **Add bias** is on).
- Activation: signum, `y = +1` if `v >= 0`, otherwise `-1`.
- Weights start as small random numbers. For each sample, if `y != t`, the update is `w = w + eta * (t - y) * x`.
- Training runs for the chosen number of epochs (no early stopping).
- `test(X, t)` does not change the weights. It returns the confusion matrix and the accuracy.

### Adaline (`adaline.py`)

- Linear output during training: `y = v = w · x + b`, with the update `w = w + eta * (t - y) * x` after every sample.
- After each epoch the MSE is computed. Training stops early when the MSE falls below the **MSE threshold**, or when the epochs run out.
- At test time the output goes through the signum function to get -1 or +1.

### Confusion matrix layout

Rows are the true class and columns are the predicted class, with Class 1 first:

|                      | Predicted Class 1 | Predicted Class 2 |
|----------------------|-------------------|-------------------|
| **True Class 1**     | TN                | FP                |
| **True Class 2**     | FN                | TP                |

Here "positive" means Class 2. Accuracy is `(TP + TN) / total`.

## How the GUI is connected

`train_and_evaluate(algo, cfg)` in `app.py` loads the data, calls `split_data`, trains the chosen algorithm, tests it, and returns everything the GUI needs. The weights are learned on standardized features, so they are converted back to original units (`w1`, `w2`, `b`) before the decision boundary is drawn. The line is `w1*x1 + w2*x2 + b = 0`.

## Notes

- **Results change between runs.** The train/test split and the initial weights are random each time you press **Train & Test**. For reproducible results, pass a seed to `split_data(...)` inside `train_and_evaluate` (for example `seed=42`).
- **Separable pairs** such as Adelie vs Gentoo and Chinstrap vs Gentoo can often reach 100% accuracy.
- **Overlapping pairs** such as Adelie vs Chinstrap are not linearly separable, so a single line cannot classify them perfectly. The Perceptron in particular does not settle on such data, so its accuracy varies a lot between runs.
- With **Add bias** unchecked, the line passes through the origin of the standardized feature space.

## Team

Add team member names here.
