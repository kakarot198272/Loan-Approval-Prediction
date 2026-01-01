import pandas as pd
from sklearn.model_selection import train_test_split

def load_train_test(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def split_xy(train_df, target_col: str = "loan_status"):
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    return X, y

def make_train_valid_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

