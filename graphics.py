import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def plot_rankings(y_test, prediccions):
    
    # 1. Filtrem les dades perquè només agafin el Top 100 real (<= 100)
    # Comprovem si y_test és una sèrie de pandas o un array de numpy
    mask = y_test <= 1000
    y_test_filtered = y_test[mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 2. Netegem les prediccions i apliquem el mateix filtre a cadascuna
    prediccions_neteades = {}
    for name, y_pred in prediccions.items():
        arr = np.array(y_pred)
        # Netegem (enters >= 1) i apliquem la màscara del Top 100
        arr_net = np.clip(arr, 1, None).round().astype(int)
        prediccions_neteades[name] = arr_net[mask]

    for ax, (name, y_pred) in zip(axes, prediccions_neteades.items()):

        # Dibuixem els punts filtrats
        ax.scatter(y_test_filtered, y_pred, alpha=0.6, s=20)

        # Línia de predicció perfecta adaptada al rang 1-100
        ax.plot(
            [0.5, 1000.5],
            [0.5, 1000.5],
            'r--',
            label="Predicció perfecta"
        )

        ax.set_title(f"{name}")
        ax.set_xlabel("Ranking real")
        ax.set_ylabel("Ranking predit")

        # Posem els límits del gràfic del 0.5 al 100.5
        ax.set_xlim(0.5, 1000.5)
        ax.set_ylim(0.5, 1000.5)

        #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Ranking real vs. ranking predit")
    plt.tight_layout()
    plt.show()