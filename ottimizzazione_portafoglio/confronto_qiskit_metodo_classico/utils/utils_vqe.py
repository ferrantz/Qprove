from docplex.mp.advmodel import AdvModel
from qiskit_optimization.translators import from_docplex_mp
mdl = AdvModel(name="Portfolio optimization")
import numpy as np

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
            lista_varianze_portafoglio.append(np.dot(df_combinazioni['combinazione'][combo].T, np.dot(matrice_delle_covarianze, df_combinazioni['combinazione'][combo]))) # controllare che la combinazione sia corretta
    return lista_varianze_portafoglio