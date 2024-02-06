import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust


def nan_replace(tabel):
    nume_variabile = list(tabel.columns)
    for each in nume_variabile:
        if any(tabel[each].isna()):
            if is_numeric_dtype(tabel[each]):
                tabel[each].fillna(tabel[each].mean(), inplace=True)
            else:
                tabel[each].fillna(tabel[each].mode()[0], inplace=True)

def partitie(matrice, nr_clusteri, p, instante):
    index_diferenta_maxima = p - nr_clusteri
    prag = (matrice[index_diferenta_maxima, 2] + matrice[index_diferenta_maxima + 1, 2]) / 2

    desen = plt.figure(figsize=(9, 9))
    ax = desen.add_subplot(1, 1, 1)

    ax.set_title("Partitionare cu " + str(nr_clusteri)
                 + " clusteri")
    hclust.dendrogram(matrice, labels=instante, ax=ax, color_threshold=prag)

    n = p + 1

    c = np.arange(n)

    for i in range(n - nr_clusteri):
        k1 = matrice[i, 0]
        k2 = matrice[i, 1]
        c[c == k1] = n + i
        c[c == k2] = n + i

    coduri = pd.Categorical(c).codes
    return np.array(["cod" + str(cod + 1) for cod in coduri])


def show():
    plt.show()
