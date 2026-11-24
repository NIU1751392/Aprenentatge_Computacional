from data import load_data
from preprocessing import *
from models import get_models
from metrics import evaluate_ranking
from graphics import plot_rankings
import pandas as pd
import numpy as np

def main():
    df = load_data()

    X,y = prepare_data(df) 

    X_train, X_test, y_train, y_test = split_data(X,y)

    #print("Train:", X_train.shape)
    #print("Test:", X_test.shape)

    models=get_models()
    resultats={}
    prediccions = {}


    #Entrenem i avaluem cada model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)

        scores=evaluate_ranking(y_test.to_numpy(),y_pred)
        resultats[name]=scores

        # Guardem les prediccions
        prediccions[name] = y_pred

    print("\nRANKING EVALUATION RESULTS")
    print("-" * 50)

    for model_name, scores in resultats.items():

        print(f"\n{model_name}")

        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")

    print("\n--- COMPARACIÓ: REAL VS PREDICTE ---")
    
    # 1. Creem un DataFrame amb el rànquing real
    df_comp = pd.DataFrame({"Real": y_test.values})
    
    # 2. Afegim les prediccions netejades de cada model
    for name, y_pred in prediccions.items():
        arr = np.array(y_pred)
        # Opcional: netegem igual que al gràfic (mínim 1 i enters)
        df_comp[f"Pred_{name}"] = np.clip(arr, 1, None).round().astype(int)
        
    # 3. Mostrem per pantalla (per exemple, les 15 primeres files)
    # Això et filtrarà i mostrarà només les que estan en el Top 100
    print(df_comp[df_comp["Real"] <= 1000].to_string(index=False))

    #Graphics
    plot_rankings(y_test, prediccions)


if __name__=="__main__":
    main()

