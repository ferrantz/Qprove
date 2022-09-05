import pandas as pd
import numpy as np

def crea_dataset_da_csv(file_csv, numero_asset):

    '''Crea il dataset per il nostro problema a partire da un file csv. I valori mancanti vengono fillati con la media.'''

    df = pd.read_csv(file_csv)[1:] # , parse_dates = [variabile_da_parsare])[3:]
    df = df.drop(columns = ['Date'])
    lst_nomi_variabili = [] # mi serve per dare i nomi alle variabili in maniera progressiva
    for i in range(df.shape[1]):
        lst_nomi_variabili.append('Asset_' + str(i + 1))
    df.columns = lst_nomi_variabili    
    df = df.apply(pd.to_numeric)
    df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    return df.iloc[: , :numero_asset] # gli dico di prendere i primi numero_asset titoli

def crea_lista_prezzi(p_asset_1, p_asset_2, p_asset_3, p_asset_4, p_asset_5, p_asset_6, p_asset_7, numero_asset):

    '''Dati i prezzi dei quattro asset restituisce la lista di questi.
    N.B.= I prezzi devono essere ordinati.'''

    return [p_asset_1, p_asset_2, p_asset_3, p_asset_4, p_asset_5, p_asset_6, p_asset_7][:numero_asset]

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

def elementi_utili(file_csv, numero_asset, p_asset_1, p_asset_2, p_asset_3, p_asset_4, p_asset_5, p_asset_6, p_asset_7):

    '''Ritorna le variabili con gli elementi necessari per fare portfolio optimization'''

    dataset = crea_dataset_da_csv(file_csv, numero_asset)
    lst_prezzi = crea_lista_prezzi(p_asset_1, p_asset_2, p_asset_3, p_asset_4, p_asset_5, p_asset_6, p_asset_7, numero_asset) 
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

def slicing_bounds(BOUNDS, numero_asset):

    '''Seleziona i primi n asset dato un bounds completo arbitrario'''

    return BOUNDS[: numero_asset]