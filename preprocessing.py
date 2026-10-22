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

    # slice each class(50 rows)
    c1 = get_class_block(df, class1)
    c2 = get_class_block(df, class2)

    # shuffle each class then split 30 train 20 test
    c1 = c1.sample(frac=1, random_state=seed)
    c2 = c2.sample(frac=1, random_state=seed)

    c1_train, c1_test = c1.iloc[:TRAIN_PER_CLASS], c1.iloc[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]
    c2_train, c2_test = c2.iloc[:TRAIN_PER_CLASS], c2.iloc[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]

    X_train = np.vstack([c1_train[cols].to_numpy(dtype=float), c2_train[cols].to_numpy(dtype=float)])
    X_test = np.vstack([c1_test[cols].to_numpy(dtype=float), c2_test[cols].to_numpy(dtype=float)])
    y_train = np.concatenate([np.ones(TRAIN_PER_CLASS), -np.ones(TRAIN_PER_CLASS)])
    y_test = np.concatenate([np.ones(TEST_PER_CLASS), -np.ones(TEST_PER_CLASS)])

    # shuffle training data so that each class is learnt equally
    order = np.random.default_rng(seed).permutation(len(X_train))
    X_train, y_train = X_train[order], y_train[order]

    mean = X_train.mean(axis=0)       # kept so the GUI can convert weights back to original units
    std = X_train.std(axis=0)
    std[std == 0] = 1                 # avoid dividing by zero if a feature is constant
    X_train = (X_train - mean) / std  # z-score scaling
    X_test = (X_test - mean) / std    # test set uses the TRAINING mean and std
    if return_stats:
        return X_train, y_train, X_test, y_test, mean, std
    return X_train, y_train, X_test, y_test



if __name__ == "__main__":
    df = load_data()
    print("Rows:", len(df), "| Missing values left:", df.isna().sum().sum())
    print(df.groupby("Species").size())

    X_tr, y_tr, X_te, y_te = split_data(df, "Adelie", "Gentoo", "CulmenLength", "FlipperLength", seed=1)
    print("Train:", X_tr.shape, "Test:", X_te.shape)
    print("Train labels +1/-1:", (y_tr == 1).sum(), (y_tr == -1).sum())