import pandas as pd
import numpy as np
# per modificare il codice originale di qiskit senza alterare la libreria originale
from docplex.mp.advmodel import AdvModel
from qiskit_optimization.translators import from_docplex_mp
mdl = AdvModel(name="Portfolio optimization")
from itertools import product

def crea_dataset_da_csv(file_csv, variabile_da_parsare):

    '''Crea il dataset per il nostro problema a partire da un file csv. I valori mancanti vengono fillati con la media.'''

    df = pd.read_csv(file_csv, parse_dates = [variabile_da_parsare])[3:]
    df = df.rename(columns = {'Unnamed: 0': 'Date', 'Adj Close': 'Asset_1', 'Adj Close.1': 'Asset_2', 'Adj Close.2': 'Asset_3', 'Adj Close.3': 'Asset_4'})
    df = df.drop(columns = ['Date'])
    df = df.apply(pd.to_numeric)
    return df.apply(lambda x: x.fillna(x.mean()),axis=0)

def crea_lista_prezzi(p_asset_1, p_asset_2, p_asset_3, p_asset_4):

    '''Dati i prezzi dei quattro asset restituisce la lista di questi.
    N.B.= I prezzi devono essere ordinati.'''

    return [p_asset_1, p_asset_2, p_asset_3, p_asset_4]

def apprezzamenti(df):

    '''Calcola i log degli apprezzamenti.'''

    return df.pct_change().apply(lambda x: np.log(1+x))

def covarianza(df):

    '''Calcola la matrice di covarianza.'''

    return df.pct_change().apply(lambda x: np.log(1+x)).cov()

def correlazione(df):

    '''Calcola la matrice di correlazione.'''

    return df.pct_change().apply(lambda x: np.log(1+x)).corr()

def medie_ritorni_attesi(log_apprez):

    '''Calcola le medie dei ritorni attesi a partire dai log degli apprezzamenti.'''

    return log_apprez.mean()

def ds_annuali(df):

    '''Calcola le deviazioni standuard annuali.'''

    return df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(df.shape[0]))

def dataframe_ritorni_e_sqm(ritorni, sqm):

    '''Crea un dataframe con ritorni attesi e dev standard degli asset.'''

    assets =  pd.concat([ritorni, sqm], axis=1) 
    assets.columns = ['Returns', 'Volatility']
    return assets

def conta_asset(df):

    '''Conta gli asset presenti nel dataframe e ritorna l'intero'''

    return df.shape[1]

def elementi_utili(file_csv, variabile_da_parsare, p_asset_1, p_asset_2, p_asset_3, p_asset_4):

    '''Ritorna le variabili con gli elementi necessari per fare portfolio optimization'''

    dataset = crea_dataset_da_csv(file_csv, variabile_da_parsare)
    lst_prezzi = crea_lista_prezzi(p_asset_1, p_asset_2, p_asset_3, p_asset_4) 
    log_apprezz_quod = apprezzamenti(dataset)
    covarianze = covarianza(dataset)
    #corr_matrix = correlazione(dataset)
    medie_er = medie_ritorni_attesi(log_apprezz_quod)
    sqm_ann = ds_annuali(dataset)
    df_assets = dataframe_ritorni_e_sqm(medie_er, sqm_ann)
    print('Dataframe con ritorni attesi e deviazioni standard degli asset:' + '\n')
    print(df_assets)
    num_assets = conta_asset(dataset)
    return dataset, lst_prezzi, log_apprezz_quod, covarianze, medie_er, sqm_ann, df_assets, num_assets

def crea_problema_quadratico(df_assets, covarianze, propensione_al_rischio, fondi, prezzi, limiti, numero_titoli):

    '''Scrive il problema quadratico così come pensato da Italo e Francesco. 
    Reference principale: certo2022comparison (presente in drive QC-Finance/articoli)'''

    # portfolio = PortfolioOptimization(
    #     expected_returns = ritorni.to_numpy(), cov_matrix = covarianze.to_numpy(), risk_factor = propensione_al_rischio, 
    #         budget = fondi, bounds = limiti)
    cov_matrix = covarianze.to_numpy()
    if limiti is not None:
        x = [mdl.integer_var(lb = limiti[i][0], ub = limiti[i][1], name = f"x_{i}") for i in range(numero_titoli)]
    else:
        x = [mdl.binary_var(name=f"x_{i}") for i in range(numero_titoli)] 
    quad = mdl.quad_matrix_sum(cov_matrix, x) # performa il calcolo x’Qx (per la seconda parte della funzione oggetto)
    linear = np.dot(df_assets['Returns'], x) # per la prima parte della funzione oggetto
    mdl.minimize(propensione_al_rischio * quad - linear) # vincolo: minimizza questa roba. Q AGISCE SULLE COVARIANZE E SI CAMBIA IL SEGNO DEI RITORNI
    mdl.add_constraint(mdl.sum(x[i] * prezzi[i] for i in range(numero_titoli)) == fondi)
    return from_docplex_mp(mdl) # traduce "in termini di qiskit" un mp problem in un problema quadratico

def liste_combinazioni_e_fval(result):

    '''Variante della funzione print_result() di Italo. Ritorna la lista con le migliori combinazioni ed un'ulteriore lista
    con gli autovalori per ogni combinazione.'''

    lista_combinazioni = []
        # prendo soltanto le combinazioni che restituiscono SUCCESS nello stato (di fatto me ne sto fregando delle probabilità che non considero più)
    lista_fval = []
    for i in range(len(result.__dict__['_samples'])):
        if str(result.__dict__['_samples'][i].status) == 'OptimizationResultStatus.SUCCESS':
            lista_combinazioni.append(result.__dict__['_samples'][i].x) # le combinazioni all'intero ci vanno come array numpy
            lista_fval.append(result.__dict__['_samples'][i].fval)
    return lista_combinazioni, lista_fval

def varianza_portafoglio(df_combinazioni, matrice_delle_covarianze):

    '''Calcola la varianza di un portfolio dati i weights degli asset e la matrice di covarianza.'''

    lista_varianze_portafoglio = []
    for combo in range(df_combinazioni.shape[0]):
            lista_varianze_portafoglio.append(np.dot(df_combinazioni['combinazione'][combo].T, np.dot(matrice_delle_covarianze, df_combinazioni['combinazione'][0]))) # controllare che la combinazione sia corretta
    return lista_varianze_portafoglio

# BACKEND = 'ibmq_montreal' 
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
# # provider_reale = IBMQ.get_provider(hub = 'ibm-q', group = 'open', project = 'main')
# # provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'simulators', project = 'simulators') # per simulatori
# backend_reale = provider_reale.get_backend(BACKEND) 
# job_reale = execute(circuit, backend = backend_reale, shots=num_shots)

def espandi_bounds(lista_bounds):

    '''Espande i bounds e crea le tuple con i valori interi pi+ù i bounds compresi, in ordine non decrescente'''

    lst = []
    for i in lista_bounds:
        lst.append(tuple(range(i[0], i[1] + 1)))
    return lst

def iterazione_classica(limiti_espansi, fondi, array_prezzi, logaritmi_degli_apprezzamenti_giornalieri, matrice_covarianze, propensione_al_rischio):

    '''Svolge l'iterazione classica esaustiva su tutte le possibili combinazioni e ritorna un df con le combinazioni e gli h'''

    combinazioni = list(product(limiti_espansi[0], limiti_espansi[1], limiti_espansi[2], limiti_espansi[3]))
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
    return pd.DataFrame(list(zip(lista_combinazioni, lista_h)))


#########################
# FORSE UTILIA
# def index_to_selection(i, num_assets):
#     s = "{0:b}".format(i).rjust(num_assets)
#     x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
#     return x


# def print_result(result):
#     selection = result.x # result._raw_samples o result.__dict__['_samples'] (meglio)
#     value = result.fval
#     print("Optimal: selection {}, value {:.4f}".format(selection, value))
#     eigenstate = result.min_eigen_solver_result.eigenstate 
#     #eigenvector = eigenstate if isinstance(eigenstate, np.ndarray) else eigenstate.to_matrix()
#     eigenvector = np.array(list(eigenstate.items()))[:, 1].astype('float')
#     probabilities = np.abs(eigenvector) ** 2
#     i_sorted = reversed(np.argsort(probabilities))
#     print("\n----------------- Full result ---------------------")
#     print("selection\tvalue\t\tprobability")
#     print("---------------------------------------------------")
#     for i in i_sorted:
#         x = index_to_selection(i, num_assets)
#         # convert() converte il problema da lineare a QUBO
#         value = QuadraticProgramToQubo().convert(qp).objective.evaluate(x)
#         # value = portfolio.to_quadratic_program().objective.evaluate(x)
#         probability = probabilities[i]
#         print("%10s\t%.4f\t\t%.4f" % (x, value, probability))

# # fai funzione da stringa a risultato compatto:
# stringa = '10111111'
# punto_attuale = 0
# vv = 0
# for v in bounds:
#     vvv = vv + v[1]
#     string = stringa[punto_attuale: vvv]
#     punto_attuale += 2
#     vv += v[1]
#     print(string)

# # fai funzione che somma i numeri di una stringa

# n = 0
# prova = list('48')
# for i, v in enumerate(prova):
#     n += int(prova[i])
# print(n)