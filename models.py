from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression


def get_models():
    models = {
        "linear_model": LinearRegression(),
        "random_forest": RandomForestClassifier(),
        "knn": KNeighborsClassifier()
    }

    return models