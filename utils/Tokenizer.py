from collections import defaultdict
import torch
import json
class MyTokenizer:

    def __init__(self, max_vocab_size=45000):
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = defaultdict(int)
        self.max_vocab_size = max_vocab_size

    def fit_on_texts(self, texts):
        # 统计词频
        for text in texts:
            for word in text.split():
                self.word_counts[word] += 1

        # 构建词汇表
        sorted_words = sorted(self.word_counts.items(), key=lambda x: (-x[1], x[0]))
        self.word2idx["<PAD>"] = 0
        self.word2idx["<UNK>"] = 1
        for idx, (word, _) in enumerate(sorted_words[:self.max_vocab_size - 2], start=2):
            self.word2idx[word] = idx

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in text.split():
                seq.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
            sequences.append(seq)
        return sequences

    def save_tokenizer_json(self, filepath):
        """保存关键参数到 JSON 文件"""
        save_data = {
            "word2idx": self.word2idx,
            "max_vocab_size": self.max_vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)



def load_tokenizer_json(filepath):
    """从 JSON 文件重建 tokenizer"""
    with open(filepath, 'r') as f:
        load_data = json.load(f)

    # 重建对象
    tokenizer = MyTokenizer(max_vocab_size=load_data["max_vocab_size"])
    tokenizer.word2idx = load_data["word2idx"]
    tokenizer.idx2word = {int(v): k for k, v in load_data["word2idx"].items()}  # 确保值类型正确
    return tokenizer

def pad_sequences(sequences, maxlen, padding='post', truncating='post', dtype=torch.long):
    """
    用 PyTorch 实现类似 Keras 的 pad_sequences 功能
    :param sequences: 输入序列列表 (List[List[int]])
    :param maxlen:    目标序列长度 (int)
    :param padding:   填充位置 ('pre' 或 'post')
    :param truncating:截断位置 ('pre' 或 'post')
    :param dtype:     输出张量类型 (默认 torch.long)
    :return:          [batch_size, maxlen] 的填充后张量
    """
    padded_sequences = []
    for seq in sequences:
        # 处理空序列
        if len(seq) == 0:
            padded = [0] * maxlen
            padded_sequences.append(padded)
            continue

        # 截断处理
        if len(seq) > maxlen:
            if truncating == 'post':
                truncated = seq[:maxlen]
            else:  # 'pre'
                truncated = seq[-maxlen:]
        else:
            truncated = seq

        # 填充处理
        pad_len = maxlen - len(truncated)
        if padding == 'post':
            padded = truncated + [0] * pad_len
        else:
            padded = [0] * pad_len + truncated

        padded_sequences.append(padded)

    return torch.tensor(padded_sequences, dtype=dtype)