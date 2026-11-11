from data import load_data
from preprocessing import *
from models import get_models
from metrics import evaluate_ranking

def main():
    df = load_data()

    X,y = prepare_data(df) 

    X_train, X_test, y_train, y_test = split_data(X,y)

    #print("Train:", X_train.shape)
    #print("Test:", X_test.shape)

    models=get_models()
    resultats={}

    #Entrenem i avaluem cada model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)

        scores=evaluate_ranking(y_test.to_numpy(),y_pred)
        resultats[name]=scores

    print("\nRANKING EVALUATION RESULTS")
    print("-" * 50)

    for model_name, scores in resultats.items():

        print(f"\n{model_name}")

        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")

if __name__=="__main__":
    main()

