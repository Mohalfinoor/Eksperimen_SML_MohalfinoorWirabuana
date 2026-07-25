"""
Nama File : automate_MohalfinoorWirabuana.py
Deskripsi : Script otomatis untuk melakukan preprocessing
            pada Bank Marketing Dataset.
"""


# Import Library

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



# Konfigurasi Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "bank-full.csv"

OUTPUT_DIR = BASE_DIR / "preprocessing"

TRAIN_OUTPUT = OUTPUT_DIR / "train_preprocessed.csv"

TEST_OUTPUT = OUTPUT_DIR / "test_preprocessed.csv"



# Load Dataset

def load_data(path: Path) -> pd.DataFrame:
    """
    Membaca dataset dari file CSV.
    """

    print("=" * 60)
    print("Membaca dataset...")
    print("=" * 60)

    df = pd.read_csv(
        path,
        sep=";"
    )

    print(f"Jumlah data : {df.shape[0]}")
    print(f"Jumlah fitur : {df.shape[1]}")

    return df



# Verifikasi Dataset

def verify_dataset(df: pd.DataFrame) -> None:
    """
    Memastikan dataset siap diproses.
    """

    print("\nVerifikasi Dataset")

    missing = df.isnull().sum().sum()
    duplicate = df.duplicated().sum()

    print(f"Missing Value : {missing}")
    print(f"Duplicate Data : {duplicate}")



# Preprocessing

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan preprocessing dataset.
    """

    print("\nMelakukan preprocessing...")

    
    # Encoding Target

    encoder = LabelEncoder()

    df["y"] = encoder.fit_transform(df["y"])

    
    # One-Hot Encoding
    
    df = pd.get_dummies(
        df,
        drop_first=True
    )

    print("Preprocessing selesai.")

    return df



# Split Dataset

def split_dataset(df: pd.DataFrame):

    print("\nMembagi data train dan test...")

    X = df.drop("y", axis=1)

    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    train = X_train.copy()

    train["y"] = y_train

    test = X_test.copy()

    test["y"] = y_test

    print(f"Train : {train.shape}")

    print(f"Test  : {test.shape}")

    return train, test



# Simpan Dataset

def save_dataset(train: pd.DataFrame,
                 test: pd.DataFrame) -> None:
    """
    Menyimpan dataset hasil preprocessing.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    test.to_csv(
        TEST_OUTPUT,
        index=False
    )

    print("\nDataset berhasil disimpan.")

    print(TRAIN_OUTPUT)

    print(TEST_OUTPUT)



# Main Program

def main():

    df = load_data(DATASET_PATH)

    verify_dataset(df)

    df = preprocess_data(df)

    train, test = split_dataset(df)

    save_dataset(train, test)

    print("\nPreprocessing selesai.")




# Eksekusi Program

if __name__ == "__main__":
    main()