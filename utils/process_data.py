import jieba
import nltk
import numpy as np
import os
import json

def load_data(data_dir):
    data_format = ['train.json', 'val.json', 'test.json']
    data_paths  = [os.path.join(data_dir, filename) for filename in data_format]

    def load_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return map(load_json, data_paths)

def load_stop_words(file_path):
    """从txt文件加载停用词"""
    stopwords = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()  # 去除首尾空白字符
                if word:  # pip install hanlp
                    stopwords.add(word)
    except FileNotFoundError:
        print(f"警告：停用词文件 {file_path} 未找到，使用空停用词列表")
    return stopwords


def split_dataset(dataset, split_rate, random_seed=None):
    """
    将数据集按比例划分为训练集、验证集和测试集

    参数:
        dataset (list/np.ndarray): 输入数据集
        split_rate (list): 划分比例，如[8,1,1]代表比例8:1:1
        random_seed (int): 随机种子，保证可重复性

    返回:
        train, val, test: 划分后的三个数据集，类型与输入一致
    """
    # 检查参数合法性
    if len(split_rate) != 3:
        raise ValueError("split_rate必须包含3个元素")
    if sum(split_rate) <= 0:
        raise ValueError("比例数值需大于0")

    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)

    # 计算实际比例
    total = sum(split_rate)
    train_ratio, val_ratio, _ = [r / total for r in split_rate]

    # 生成随机索引
    n = len(dataset)
    indices = np.random.permutation(n)

    # 计算划分位置
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # 提取数据
    def get_data(indices):
        if isinstance(dataset, np.ndarray):
            return dataset[indices]
        return [dataset[i] for i in indices]

    return (
        get_data(indices[:train_end]),
        get_data(indices[train_end:val_end]),
        get_data(indices[val_end:])
    )

class DataProcessor:
    def __init__(self, stop_word, language='chinese'):
        self.stop_words = stop_word
        self.language = language

    def preprocess_text(self, text):
        if self.language == 'chinese':
            return self._preprocess_cn_text(text)
        else:
            return self._preprocess_en_text(text)

    def _preprocess_cn_text(self, text):
        # 分词
        words = jieba.cut(text)
        # 去停用词
        clean_words = [
            word.strip() for word in words
            if (word not in self.stop_words) and (not word.isspace())
        ]
        return ' '.join(clean_words)

    def _preprocess_en_text(self, text):
        text = text.lower()
        # 分词
        words = nltk.word_tokenize(text)
        # 去停用词
        clean_words = [
            word.strip() for word in words
            if (word not in self.stop_words) and (not word.isspace())
        ]
        return ' '.join(clean_words)