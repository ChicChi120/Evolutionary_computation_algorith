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
# ホタルの数とその行列
firefly = 4
firefly_matrix = np.zeros((firefly, N))

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


def move_x(Fi, Fj):
    alpha = 0.5
    epsv = rd.uniform(-0.5, 0.5)

    return Fi + attract(Fi, Fj) * (Fj - Fi) + alpha * epsv
        
# 生成し置換する
def change(X, firefly_matrix):
    Fi = np.zeros(N)
    Fj = np.zeros(N)
        
    # Fi が Fj より明るかったら移動
    for i in range(0, firefly, 2):
        for j in range(N):
            Fi[j] = X[i][j]
            Fj[j] = X[i+1][j]
            
        if (valueF(Fi) < valueF(Fj)):
            # Fi の代わりに firefly_matrix の半分を更新する
            for j in range(N):
                firefly_matrix[i][j] = move_x(Fi[j], Fj[j])

        else:
            for j in range(N):
                firefly_matrix[i][j] = X[i][j]
        '''
        for j in range(N):
            firefly_matrix[i+1][j] = X[i+1][j]
        '''   

    return firefly_matrix

# firefly_matrix = change(X, firefly_matrix)

# X を改善
def recreate_x(X, firefly_matrix):
    X = firefly_matrix

    for i in range(1, firefly, 2):
        for j in range(N):
            X[i][j] = np.array([rd.uniform(-1, 1)])

    return X
            

# 最前解を記録する
def best_sol(matrix, pos):
    pos_tmp = np.zeros(N)

    for i in range(firefly):
        for j in range(N):
            pos_tmp[j] = matrix[i][j]

        if (valueF(pos) > valueF(pos_tmp)):
            pos = pos_tmp
        
    return pos
    
            

if __name__ == '__main__':

    # 終了条件
    eps = pow(10, -5)

    # 最大反復回数
    max_generarion = 1000

    
    for generation in range(max_generarion):
        firefly_matrix = change(X, firefly_matrix)

        X = recreate_x(X, firefly_matrix)

        pos = best_sol(firefly_matrix, pos)

        
    
    