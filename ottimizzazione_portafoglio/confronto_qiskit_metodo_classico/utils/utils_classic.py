from itertools import product
import numpy as np
import pandas as pd

def espandi_bounds(lista_bounds):

    '''Espande i bounds e crea le tuple con i valori interi più i bounds compresi, in ordine non decrescente'''

    lst = []
    for i in lista_bounds:
        lst.append(tuple(range(i[0], i[1] + 1)))
    return lst

def conta_combinazioni_classiche(limiti_espansi):

    '''Ritorna il numero di combinazioni (dati il numero degli asset ed i bounds).'''

    if len(limiti_espansi) == 2:
        output = len(list(product(limiti_espansi[0], limiti_espansi[1])))
    if len(limiti_espansi) == 3:
        output = len(list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2])))
    if len(limiti_espansi) == 4:
        output = len(list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3])))
    if len(limiti_espansi) == 5:
        output = len(list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3], limiti_espansi[4])))
    if len(limiti_espansi) == 6:
        output = len(list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3], limiti_espansi[4], limiti_espansi[5])))
    if len(limiti_espansi) == 7:
        output = len(list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3], limiti_espansi[4], limiti_espansi[5], limiti_espansi[6])))
    return output


def iterazione_classica(limiti_espansi, fondi, array_prezzi, logaritmi_degli_apprezzamenti_giornalieri, matrice_covarianze, propensione_al_rischio):

    '''Svolge l'iterazione classica esaustiva su tutte le possibili combinazioni e ritorna un df con le combinazioni e gli h'''

    if len(limiti_espansi) == 2:
        combinazioni = list(product(limiti_espansi[0], limiti_espansi[1]))
    elif len(limiti_espansi) == 3:
        combinazioni = list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2]))
    elif len(limiti_espansi) == 4:
        combinazioni = list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3]))
    elif len(limiti_espansi) == 5:
        combinazioni = list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3], limiti_espansi[4]))
    elif len(limiti_espansi) == 6:
        combinazioni = list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3], limiti_espansi[4], limiti_espansi[5]))
    elif len(limiti_espansi) == 7:
        combinazioni = list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3], limiti_espansi[4], limiti_espansi[5], limiti_espansi[6]))

    combinazioni = tuple(tuple(map(float, t)) for t in combinazioni) 
    c = 0
    d = 0
    lista_combinazioni = [] # inizializzo una lista vuota nella quale appenderò le combinazioni che soddisfano il budget
    lista_h = [] # qui appenderò tutti gli autovalori calcolati con la mia formula
    for w in combinazioni:
        c += 1
        price = np.matmul(array_prezzi.T, w)
        if price == fondi:
            w = np.array(w)
            h = -(np.matmul(w, logaritmi_degli_apprezzamenti_giornalieri.iloc[-1, :].T)) + np.matmul(np.matmul(w.T, matrice_covarianze), w) * propensione_al_rischio ####q*w.T*cov_matrix*w
            lista_combinazioni.append(w)
            lista_h.append(h)
            d += 1
    h_best = lista_h[0]
    return pd.DataFrame(list(zip(lista_combinazioni, lista_h))), h_best