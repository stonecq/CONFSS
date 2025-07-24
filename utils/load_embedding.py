import torch
from tqdm import tqdm
import numpy as np

def load_embedding(path, dim, word_index):
    print("加载embedding")
    np.random.seed(11)
    word_embedding = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            if len(values) != dim + 1 :
                print(f"跳过格式错误行：{line[:20]}...")
                continue
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_embedding[word] = vector

    embedding_matrix = np.zeros((len(word_index), dim))
    embedding_matrix[1] = np.random.normal(size=(dim,)).astype('float32')
    found_count = 0

    # 构建嵌入矩阵
    for word, idx in word_index.items():
        # 跳过填充符(0)和UNK(1)
        if idx in [0, 1]:
            continue
        vector = word_embedding.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
            found_count += 1
        else:
            # 随机初始化
            embedding_matrix[idx] = np.random.normal(size=(dim,)).astype('float32')

    # 输出统计信息
    coverage = found_count / (len(word_index) - 2)  # 排除 0(pad) 和 1(unk)
    print(f"加载完成！覆盖词汇：{found_count}/{len(word_index) - 2} ({coverage:.2%})")

    return torch.tensor(embedding_matrix,dtype=torch.float32)