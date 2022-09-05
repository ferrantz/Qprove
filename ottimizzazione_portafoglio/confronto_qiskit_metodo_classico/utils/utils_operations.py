import pandas as pd
import numpy as np

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