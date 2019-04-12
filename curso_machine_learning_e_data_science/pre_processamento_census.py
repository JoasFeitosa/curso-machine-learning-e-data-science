# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 01:37:30 2019

@author: Joás
"""

import pandas as pd

base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Transformação dos tipos dos atributos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_previsores = LabelEncoder()
#labels = labelenconder_previsores.fit_transform(previsores[:,1])
previsores[:,1] = labelenconder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelenconder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelenconder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelenconder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelenconder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelenconder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelenconder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelenconder_previsores.fit_transform(previsores[:,13])

onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# Escalonamento de atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores - scaler.fit_transform(previsores)