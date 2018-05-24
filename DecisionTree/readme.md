## 决策树

## 目录结构
-ContactLensPredict  # 根据给定数据集预测使用什么类型的隐形眼镜

--- ID3.py # 用ID3实现

--- lenses.txt # 数据集

## 注意
上述代码是根据机器学习实战进行书写，发现书中原始代码存在一些问题

书中没有将数据集和数据产生的结果分离进行处理（也就是没有将数据前四列和最后一列物理隔离，只是每次使用时根据列号处理），而建树时随着不断生成树，
可能最后只会剩下最后一列，而这一列对应这不同类型的情况具体应该戴什么种类的隐形眼镜，是不应该作为候选结点，但是原始代码也会对最后一列进行建树。