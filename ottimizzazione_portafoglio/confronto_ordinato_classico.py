import numpy as np
import pandas as pd
from operator import itemgetter
from itertools import product

PREZZO_ASSET_1 = 50
PREZZO_ASSET_2 = 70
PREZZO_ASSET_3 = 25
PREZZO_ASSET_4 = 30
lista_prezzi = [PREZZO_ASSET_1, PREZZO_ASSET_2, PREZZO_ASSET_3, PREZZO_ASSET_4]

test_2 = pd.read_csv('ottimizzazione_portafoglio/ds_completo_un_anno.csv', parse_dates = ['Unnamed: 0'])[3:]
aaa = [col for col in test_2 if col.startswith('Adj Close')]
test_2 = test_2[aaa]
#test_2 = test_2.loc[:, test_2.columns.str.startswith('Adj Close')].columns

test_2 = test_2.apply(pd.to_numeric)
test = test_2.apply(lambda x: x.fillna(x.mean()),axis=0)
log_apprezzamenti_quotidiani = test.pct_change().apply(lambda x: np.log(1+x))
cov_matrix = test.pct_change().apply(lambda x: np.log(1+x)).cov()
corr_matrix = test.pct_change().apply(lambda x: np.log(1+x)).corr()
medie_diversi_er = log_apprezzamenti_quotidiani.mean()
ann_sd = test.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(test.shape[0]))
assets = pd.concat([medie_diversi_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']
q = 1
num_assets = 4 #len(test.columns)

p = np.array([10, 20, 30, 40]) # prezzi


bounds = list(((0, 1), ) * test_2.shape[1]) 


def espandi_bounds(lista_bounds):
    lst = []
    for i in lista_bounds:
        lst.append(tuple(range(i[0], i[1] + 1)))
    return lst


bounds_espanso = espandi_bounds(bounds)
combinazioni = list(product(bounds_espanso[0], bounds_espanso[1], bounds_espanso[2], bounds_espanso[3]))



budget_min = 30
budget_max = 30
c = 0
d = 0
# ris = pd.DataFrame(columns = ('w', 'price', 'h'))
ris = []
print(len(combinazioni))

for w in combinazioni:
    c += 1
    price = np.matmul(p.T, w)
    if price >= budget_min and price <= budget_max:
        w = np.array(w)
        h = -(np.matmul(w, log_apprezzamenti_quotidiani.iloc[-1, :].T)) + np.matmul(np.matmul(w.T, cov_matrix), w) * q ####q*w.T*cov_matrix*w
        ris.append([w, price, h])
        d += 1
        
print(assets)        
print('Tutte le possibili combinazioni sono: ' + str(c))
print('Le combinazioni filtrate sono: ' + str(d))
print(sorted(ris, key=itemgetter(2)))