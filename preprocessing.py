from sklearn.model_selection import train_test_split

def prepare_data(df):
    """
    Separa les variables predictives (X)
    de la variable objectiu (y).
    """

    features = [
        "quality_of_education",
        "alumni_employment",
        "quality_of_faculty",
        "publications",
        "influence",
        "citations",
        "patents"
    ]

    target = "world_rank"

    X = df[features].copy()
    y = df[target].copy()

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test