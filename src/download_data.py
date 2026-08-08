import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    dataset = load_breast_cancer()

    df = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    df["target"] = dataset.target

    df.to_csv("data/breast_cancer.csv", index=False)

    print("Dataset saved successfully!")
    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()