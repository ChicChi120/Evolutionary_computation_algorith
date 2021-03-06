# 進化計算アルゴリズム入門を実装してみた
[進化計算アルゴリズム入門 生物の行動科学から導く最適解](https://www.ohmsha.co.jp/book/9784274222382/) の擬似コードを参考に Python で実装した. 分析するデータには scikit-learn の Toy データセットである load_boston を使った.

## 人工蜂コロニーアルゴリズムで重回帰分析
人工蜂コロニーアルゴリズム (artificial bee colony algorithm, ABC)は, ミツバチの採餌行動を模倣した最適解探索アルゴリズムである. 収穫バチ (employed bee), 追従バチ (onlocker bee), 偵察バチ (scout bee) の 3 つの役割を担うハチを用意し最適解を更新する.  
他のアルゴリズムと比べ 300 から 400 世代と早く局所解に達してしまい, その後抜け出すことができていないため多様性をどう作るかが今後の課題.

## ホタルアルゴリズムで重回帰分析
ホタルアルゴリズム (firefly algorithm) は, ホタルの求愛行動を模倣した最適解探索アルゴリズムである. 強い光を放つホタルに他のホタルが近づき, それ以外のホタルはランダムに飛ぶように設定することで最適解を見つけ近づいていく.  
安定して局所解に到達する. パラメータの値の調整によってはもう少し性能の向上が期待できると思う.

## コウモリアルゴリズムで重回帰分析
コウモリアルゴリズム (bat algorithm) は, コウモリが超音波を使って獲物や障害物を特定する反響定位行動を模倣したアルゴリズムである. 周波数, パルス率, 音量の 3 つの変量を元にコウモリは速度を変えて最適解を導く.  
残渣は３つの中で一番小さい値を出した. しかし, 安定性ではホタルアルゴリズムに劣る. より大きいデータセットではどうなるか試してみる.