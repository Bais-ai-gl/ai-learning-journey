# Day 7：Transformer 概念 · 手写单头注意力（纯 numpy）
# 用法：py 02_attention_numpy.py   （只依赖 numpy，py -m pip install numpy）
# 目的：理解注意力公式  softmax(Q·K^T / √d) · V 的每一步

import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 softmax：先减最大值防溢出"""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """缩放点积注意力"""
    d_k = K.shape[-1]                        # 键向量的维度
    scores = Q @ K.T / np.sqrt(d_k)          # ① Q 和 K 点积求相关性，再除以 √d 稳定梯度
    if mask is not None:
        scores = scores + mask               # ② (可选)掩码：把不该看的位置置 -inf
    weights = softmax(scores)                # ③ softmax 归一化成"注意力权重"
    return weights @ V, weights              # ④ 用权重对 V 加权求和

np.random.seed(0)
seq_len, d_model = 4, 8                     # 4 个 token，每个 8 维
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

out, w = attention(Q, K, V)

print("Q/K/V 形状：", Q.shape)
print("注意力权重矩阵（每行代表一个 query 对 4 个 key 的关注度）：")
print(w.round(2))
print("每行权重和（应都为 1）：", w.sum(axis=1).round(4))
print("输出形状：", out.shape)

# 思考：
# 1) Q/K/V 三个矩阵分别扮演什么角色？
# 2) 为什么除以 √d_k？
# 3) mask 的作用是什么？（生成式模型里很重要）
