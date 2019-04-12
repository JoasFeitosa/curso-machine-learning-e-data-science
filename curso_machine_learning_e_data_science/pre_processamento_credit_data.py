# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:43:37 2019

@author: Joás
"""

import pandas as pd
base = pd.read_csv('credit-data.csv')
base.describe()

# corrigindo idades negativas na base de dados
base.loc[base['age'] < 0]

# apagar a coluna
base.drop('age', 1, inplace=True)

# apagar somente os registros com problemas
base.drop(base[base.age < 0].index, inplace=True)

# preencher os valores manualmente
# preencher os valores com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

# tratando valores nullos 
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

# separando previsores e classes
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:, 0:3])

# Escalonamento de atributos / padronização dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)