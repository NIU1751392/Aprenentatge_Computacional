import pandas as pd

def load_data():
    df = pd.read_csv("dataset/cwurData.csv")
    return df


def explore_data(df):
    print("\n--- DIMENSIONS ---")
    print(df.shape)

    print("\n--- COLUMNES ---")
    print(df.columns.tolist())

    print("\n--- PRIMERES FILES ---")
    print(df.head())

    print("\n--- INFORMACIÓ ---")
    print(df.info())

    print("\n--- VALORS NULS ---")
    print(df.isnull().sum())

    print("\n--- ESTADÍSTIQUES ---")
    print(df.describe())


if __name__ == "__main__":
    df = load_data()
    #explore_data(df)