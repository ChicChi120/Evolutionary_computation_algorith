# ABC で重回帰分析

import numpy as np
import pandas as pd
import random as rd


df = pd.read_csv('hanoi.csv', header=None)

# 説明変数の数
N = df.shape[1] - 1

# データ数
D = df.shape[0]

# 食料源の初期位置
pos = np.zeros(N)
for i in range(N):
    pos[i] = rd.uniform(-1, 1)


exSData = df.iloc[:, [0, 1]].copy()
resSData = df[N]


# 標準化
for i in range(exSData.shape[1]):
    exSData[i] = (exSData[i] - exSData.mean()[i]) / exSData.std()[i]
exSData = exSData.T
resSData = (resSData - resSData.mean()) / resSData.std()

# 評価値
def valueF(pos):
    f = 0
    g = 0
    for i in range(D):
        for j in range(N):
            g += pos[j] * exSData[i][j]
        
        f += pow((resSData[i] - g), 2)
            
    return f


# 収穫バチ (employed bee) =============================================

# 収穫バチの数とその行列
e_bee = 30
e_bee_matrix = np.zeros((N, e_bee))

# 初期の食料源 (解候補の行列)
X = np.zeros((N, e_bee))
for i in range(N):
    for j in range(e_bee):
        X[i][j] = rd.uniform(-1, 1)

# 生成し置換する
def change(matrix, bee):
    for i in range(bee):
        for j in range(N):
            matrix[j][i] = X[j][rd.randint(0, bee-1)]

        k = rd.randint(0, N-1)   
        matrix[k][i] = matrix[k][i] + rd.uniform(-1, 1) * (matrix[k][i] - X[k][rd.randint(0, bee-1)])
        
        pos_tmp = np.zeros(N)
        for p in range(N):
            pos_tmp[p] = matrix[p][i]
        
        # 小さくなったら更新 -> 小さくなかったら更新しない
        if valueF(pos) < valueF(pos_tmp):
            for p in range(N):
                matrix[p][i] = X[p][i]

    return matrix

# e_bee_matrix = change(e_bee_matrix, e_bee)


# 追従バチ (onlooker bee) =============================================

# 追従バチの数とその行列
o_bee = 10
o_bee_matrix = np.zeros((N, o_bee))

# 追従する収穫バチを選ぶ確率
def prob_b(matrix, k):
    listf = np.zeros(e_bee)

    for i in range(e_bee):
        pos_damy = np.zeros(N)

        for j in range(N):
            pos_damy[j] = matrix[j][i]

        listf[i] = valueF(pos_damy)

    vmax = listf.max()
    vmin = listf.min()

    trVal = np.zeros(e_bee)

    for i in range(e_bee):
        trVal[i] = (vmax - listf[i]) / (vmax - vmin)

    prob = trVal[k] / trVal.sum()

    return prob

def onlooker_bee(o_bee_matrix):
    # ハチ用のリスト
    bee_list = []
    for i in range(e_bee):
        bee_list.append(i)
        
    # 重み用のリスト
    weight_list = []
    for i in range(e_bee):
        weight_list.append(prob_b(e_bee_matrix, i))
        
    index_list = rd.choices(bee_list, k=o_bee, weights=weight_list)
    
    for i in range(o_bee):
        for j in range(N):
            o_bee_matrix[j][i] = e_bee_matrix[j][index_list[i]]

    return o_bee_matrix


# 収穫バチと同様の処理をする
# o_bee_matrix = change(o_bee_matrix, o_bee)


# 偵察バチ (scout bee) =============================================
# ずっと変化がない食料源(列) を見つけ置換する
def scout_bee(X):
    for i in range(e_bee):
        if (np.all(X[:, i] == e_bee_matrix[:, i])):
            X[:, i] = np.array([rd.uniform(-1, 1), rd.uniform(-1, 1)])

    return X

# 最良の食料源を更新する
def best_sol(pos):
    best_pos = np.zeros(N)
    for i in range(N):
        best_pos[i] = o_bee_matrix[i][0]
    
    if valueF(pos) > valueF(best_pos):
        pos = best_pos

    return pos

    


if __name__ == "__main__":

    # 終了条件
    esp = pow(10, -5)

    # 最大反復回数
    max_generation = 100
    
    for generation in range(max_generation):

        # 収穫バチフェーズ
        e_bee_matrix = change(e_bee_matrix, e_bee)

        # 追従バチフェーズ
        onlooker_bee(o_bee_matrix)
        o_bee_matrix = change(o_bee_matrix, o_bee)

        # 偵察バチフェーズ
        X = scout_bee(X)

        # 最良食料源の記録
        pos = best_sol(pos)

        if valueF(pos) < esp:
            print(generation)
            break

        
    # 標準偏回帰係数の出力
    print("y = ({:.3f}) x1 + ({:.3f}) x2".format(pos[0], pos[1]))