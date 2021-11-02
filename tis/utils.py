
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import RobustScaler

def data_loader(files_normal:list, files_attack:list, path:str='data/'): 
    """
    """
    df_normal = read_files(files=files_normal, path=path)
    df_attack = read_files(files=files_attack, path=path)
    Xn = df_normal.values
    Xa, Ya = df_attack.values[:, :-1], df_attack.values[:, -1]
    X, Y, keys = prepare_dataset(Xn, Xa, Ya)
    transformer = RobustScaler().fit(X)
    X = transformer.transform(X)
    return X, Y, keys 


def read_files(files:list, path:str='data/'): 
    """
    """
    df = None
    for file in files: 
        dft = pd.read_csv(''.join([path, file]))
        if df is None: 
            df = dft
        else: 
            df.append(dft)
    return df 


def prepare_dataset(Xn, Xa, Ya): 
    """
    """
    X = np.vstack((Xn, Xa))
    Y = np.zeros(len(Xn))
    Y_temp = np.zeros(len(Xa))
    keys = ['Normal']
    for i, val in enumerate(np.unique(Ya)): 
        Y_temp[Ya == val] = i+1
        keys.append(val)
    Y = np.concatenate((Y, Y_temp))
    i = np.random.permutation(len(Y))
    X, Y = X[i], Y[i]
    return X, Y, keys