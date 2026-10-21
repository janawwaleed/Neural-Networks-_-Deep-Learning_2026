from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "penguins.csv"

FEATURES = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
CLASSES = ["Adelie", "Chinstrap", "Gentoo"]

TRAIN_PER_CLASS = 30
TEST_PER_CLASS = 20
SAMPLES_PER_CLASS = 50


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)  
    df = fill_missing(df) #func call
    df = encode_origin(df) #func call
    return df


def fill_missing(df):
    df = df.copy()
    numeric_cols = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
    for col in numeric_cols:
        df[col] = df.groupby("Species")[col].transform(lambda s: s.fillna(s.mean()))

    if df["OriginLocation"].isna().any():
        df["OriginLocation"] = df["OriginLocation"].fillna(df["OriginLocation"].mode()[0])
    return df


def encode_origin(df):
    df = df.copy()
    locations = sorted(df["OriginLocation"].unique())
    mapping = {name: i for i, name in enumerate(locations)}
    df["OriginLocation"] = df["OriginLocation"].map(mapping)
    return df


def get_class_block(df, cls):

    start = CLASSES.index(cls) * SAMPLES_PER_CLASS
    return df.iloc[start:start + SAMPLES_PER_CLASS]


def split_data(df, class1, class2, feature1, feature2, seed=None, return_stats=False):
    cols = [feature1, feature2]

    # Each class is a block of 50 rows in the CSV -> slice that block out
    c1 = get_class_block(df, class1)
    c2 = get_class_block(df, class2)

    # Shuffle inside the block, then slice: first 30 -> train, next 20 -> test
    c1 = c1.sample(frac=1, random_state=seed)
    c2 = c2.sample(frac=1, random_state=seed)

    c1_train, c1_test = c1.iloc[:TRAIN_PER_CLASS], c1.iloc[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]
    c2_train, c2_test = c2.iloc[:TRAIN_PER_CLASS], c2.iloc[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]

    X_train = np.vstack([c1_train[cols].to_numpy(dtype=float), c2_train[cols].to_numpy(dtype=float)])
    X_test = np.vstack([c1_test[cols].to_numpy(dtype=float), c2_test[cols].to_numpy(dtype=float)])
    y_train = np.concatenate([np.ones(TRAIN_PER_CLASS), -np.ones(TRAIN_PER_CLASS)])
    y_test = np.concatenate([np.ones(TEST_PER_CLASS), -np.ones(TEST_PER_CLASS)])

    # Shuffle the training set so the model doesn't see all of class 1 then all of class 2
    order = np.random.default_rng(seed).permutation(len(X_train))
    X_train, y_train = X_train[order], y_train[order]

    mean = X_train.mean(axis=0)   # kept so the GUI can convert weights back to original units
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train, X_test = standardize(X_train, X_test)
    if return_stats:
        return X_train, y_train, X_test, y_test, mean, std
    return X_train, y_train, X_test, y_test


def standardize(X_train, X_test):
    """Z-score scaling: (x - mean) / std, using training stats for both sets."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # avoid dividing by zero if a feature is constant
    return (X_train - mean) / std, (X_test - mean) / std


if __name__ == "__main__":
    # Quick check: run "python data.py"
    df = load_data()
    print("Rows:", len(df), "| Missing values left:", df.isna().sum().sum())
    print(df.groupby("Species").size())

    X_tr, y_tr, X_te, y_te = split_data(df, "Adelie", "Gentoo", "CulmenLength", "FlipperLength", seed=1)
    print("Train:", X_tr.shape, "Test:", X_te.shape)
    print("Train labels +1/-1:", (y_tr == 1).sum(), (y_tr == -1).sum())