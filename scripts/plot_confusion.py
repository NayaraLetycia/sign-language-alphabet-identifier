# plot_confusion.py
# Carrega confusion_matrix.npy e classes.json e plota a matriz.

import os
import json
import numpy as np
import matplotlib.pyplot as plt

ART_DIR = "artifacts"

def main():
    cm_path = os.path.join(ART_DIR, "confusion_matrix.npy")
    classes_path = os.path.join(ART_DIR, "classes.json")

    if not os.path.exists(cm_path):
        print("Arquivo de matriz de confusão não encontrado:", cm_path)
        return
    if not os.path.exists(classes_path):
        print("Arquivo de classes não encontrado:", classes_path)
        return

    cm = np.load(cm_path)
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Verdadeiro",
        xlabel="Previsto",
        title="Matriz de Confusão (Validação)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Anotações
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
