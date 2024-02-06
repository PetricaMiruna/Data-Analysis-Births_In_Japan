import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hclust
import seaborn as sns
from sklearn.decomposition import PCA
from utils import *


def execute():

    #CLUSTERIZARE

    tabelBirths = pd.read_csv("C:\\DSAD\\ProiectDSAD\\birthsClusterizare.csv", index_col=0)

    instante = list(tabelBirths.index)
    variabile = list(tabelBirths)[:11]
    print(variabile)

    n = len(instante)
    m = len(variabile)



    nan_replace(tabelBirths)
    x = tabelBirths[variabile].values


    matrice = hclust.linkage(x, method='ward')
    #np.set_printoptions(precision=4, suppress=True)
    print('Ierarhie de clusterizare: ', matrice)

    p = n - 1

    index_diferenta_maxima = np.argmax(matrice[1:, 2] - matrice[:(p - 1), 2])
    print("Index diferenta maxima", index_diferenta_maxima)
    print(matrice[index_diferenta_maxima])
    nr_clusteri = p - index_diferenta_maxima
    print("Nr clusteri: ", nr_clusteri)



    partitie_optima = partitie(matrice, nr_clusteri, p, instante)
    print("Partitie optima: ", partitie_optima)

    partitie_optima_fisier = pd.DataFrame(data={
        "Cluster": partitie_optima
    },
        index=instante)
    partitie_optima_fisier.to_csv("C:\\DSAD\\ProiectDSAD\\PartitieOptima.csv")


    partitie3Clusteri = partitie(matrice, 3, p, instante)
    print("Partitie cu 3 clusteri: ", partitie3Clusteri)

    partitie_3_fisier = pd.DataFrame(data={
        "Cluster": partitie3Clusteri
    },
        index=instante)
    partitie_3_fisier.to_csv("C:\\DSAD\\ProiectDSAD\\Partitie3Clusteri.csv")

    show()

    #REDUCEREA DIMENSIONALITATII
    print("----------------------Reducerea dimensionalitatii:-----------------------")

    tabelBirthsJapan = pd.read_csv("C:\\DSAD\\ProiectDSAD\\birthsJapanRD.csv", index_col=0)
    nan_replace(tabelBirthsJapan)



    lista_variabile = list(tabelBirthsJapan)[:]
    print(lista_variabile)

    var_standardizate = (tabelBirthsJapan[lista_variabile] - np.mean(tabelBirthsJapan[lista_variabile], axis=0)) / np.std(tabelBirthsJapan[lista_variabile], axis=0)
    b = tabelBirthsJapan[lista_variabile].values
    n, m = b.shape


    componente_principale = PCA()
    componente_principale.fit(b)
    print(componente_principale)

    val_proprii = componente_principale.explained_variance_
    vectori_proprii = componente_principale.components_
    componente_rezultate = componente_principale.transform(b)

    np.set_printoptions(precision=4, suppress=True)

    print("Valorile proprii:", val_proprii)
    print("Vectorii proprii:", vectori_proprii)
    print("Componente principale rezultate:", componente_rezultate)

    #Identificarea numarului de componente semnificative

    where = np.where(val_proprii > 1)

    nr_comp_kaiser = len(where[0])
    print(" Conform Criteriului Kaiser, numarul de componente principale semnificative este: ", nr_comp_kaiser)


    C3 = componente_rezultate[:, 3]
    C4 = componente_rezultate[:, 4]


    #Scatter plot


    plt.scatter(C3, C4)
    plt.xlabel("Componenta Principala 3")
    plt.ylabel("Componenta Principala 4")
    plt.title("Scatter Plot pentru componentele principale")
    plt.show()

    #Bar plot

    plt.bar(range(1, len(val_proprii) + 1), val_proprii)
    plt.xlabel('Componenta Principala')
    plt.ylabel('Valoare Proprie')
    plt.title('Bar Plot pentru valorile proprii')
    plt.show()



if __name__ == '__main__':
    execute()
