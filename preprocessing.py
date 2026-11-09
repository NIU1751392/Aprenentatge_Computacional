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