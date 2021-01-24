# Bat Algorithm で重回帰分析

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


# コウモリの初期位置
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

# コウモリの移動 ========================================================
# コウモリの数とその行列
bat = 30

# 初期位置(解候補の行列)
X = np.zeros((bat, N))
for i in range(bat):
    for j in range(N):
        X[i][j] = rd.uniform(-1, 1)

# 周波数ベクトル
f = np.zeros(bat)

# パルス率ベクトル
pr = np.zeros(bat)

# パルス率の収束値と収束の速さ
R0 = 0.6
gamma = 0.2
e = 2.71828

# 音量行列
A = np.ones((bat, N))

# 音量の減少率
alpha = 0.9

# 速度行列
V = np.zeros((bat, N))

# 移動速度更新関数
def v_new(X, pos):
    for i in range(bat):
        f[i] = rd.uniform(0, 1)

    for i in range(bat):
        for j in range(N):
            V[i][j] = V[i][j] + f[i] * (pos[j] - X[i][j])

    return V

# 最良のコウモリの位置に移動
def best_bat(X, V):
    for i in range(bat):
        for j in range(N):
            X[i][j] = X[i][j] + V[i][j]
            
    return X

# 良いコウモリの位置に移動
def better_bat(X, A, pos):
    Abar = A.mean()

    for i in range(bat):
        for j in range(N):
            X[i][j] = pos[j] + Abar * rd.uniform(-1, 1)
            
    return X    
    
# パルス率と音量の更新
def update_pr(pr, A, generaton):
    for i in range(bat):
        pr[i] = R0 * (1 - pow(e, - gamma * generaton))
        
        for j in range(N):
            A[i][j] = alpha * A[i][j]
            
    return pr, A

# 最良の解を保存する
def best_sol(matrix, pos):
    pos_tmp = np.zeros(N)

    for i in range(bat):
        for j in range(N):
            pos_tmp[j] = matrix[i][j]

        if (valueF(pos) > valueF(pos_tmp)):
            pos = pos_tmp
        
    return pos


if __name__ == '__main__':

    # 終了条件
    eps = pow(10, -5)

    # 最大反復回数
    max_generation = 1000

    for generaton in range(max_generation):

        # 最良コウモリの方向に移動
        V = v_new(X, pos)
        X = best_bat(X, V)

        if (rd.uniform(-1, 1) > pr[0]):
            # 良いコウモリの近くに新しい位置を生成
            X = better_bat(X, A, pos)

        # ランダムに新しい位置を生成
        for i in range(N):
            X[bat-1][i] = rd.uniform(-1, 1)

        # 新しい位置が元の位置より評価が高い
        pos_tmp = best_sol(X, pos)
        if (valueF(pos) > valueF(pos_tmp)):

            if (rd.uniform(-1, 1) < A[0][0]):
                pos = pos_tmp
                
                pr, A = update_pr(pr, A, generaton)

        print(valueF(pos))

    print(pos)
            

    
