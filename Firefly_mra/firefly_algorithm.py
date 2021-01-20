# Firefly Algorithm で重回帰分析

import numpy as np
import pandas as pd
import random as rd
from sklearn.preprocessing import StandardScaler


# 使うデータセット
from sklearn.datasets import load_boston

data = load_boston()
data_x = pd.DataFrame(data.data, columns=data.feature_names)
data_y = pd.Series(data.target)

# データを小さく削る
data_x = data_x.drop(range(20, 506))
data_y = data_y.drop(range(20, 506))


# 説明変数の数
N = data_x.shape[1]

# データ数
D = data_x.shape[0]

# ホタルの初期位置
pos = np.zeros(N)
for i in range(N):
    pos[i] = rd.uniform(-1, 1)


# 標準化
sc = StandardScaler()
exSData = sc.fit_transform(data_x)
resSData = (data_y - data_y.mean()) / data_y.std()


# 評価値
def valueF(pos):
    f = 0
    g = 0
    for i in range(D):
        for j in range(N):
            g += pos[j] * exSData[i][j]
        
        f += pow((resSData[i] - g), 2)
            
    return f


# ホタルの移動 =========================================================
# ホタルの数
firefly = 30

# 初期位置(解候補の行列)
X = np.zeros((firefly, N))
for i in range(firefly):
    for j in range(N):
        X[i][j] = rd.uniform(-1, 1)
        
# 距離関数
def euclid_d(xi, xj):
    d = 0
    d = np.linalg.norm(xi - xj)

    return d


# ランベルト・ベールの法則
def attract(Fi, Fj):
    beta0 = 1
    gamma = 0.5
    e = 2.71828
    d = euclid_d(Fi, Fj)
    
    return beta0 * pow(e, - (gamma * d**2))

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(attract(a, b))

def update_x(xi, Fi, Fj):
    x = np.zeros(N)
    alpha = 0.5
    epsv = np.zeros(N)
    
    for i in range(N):
        epsv[i] = rd.uniform(-0.5, 0.5)

    return xi + attract(Fi, Fj) * (Fj - Fi) + alpha * epsv
        

