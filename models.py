from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


def get_models():
    models = {
        "linear_model": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=42),
        "knn": KNeighborsRegressor()
    }

    return models