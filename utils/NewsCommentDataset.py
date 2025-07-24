import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NewsCommentDataset(Dataset):
    def __init__(self, news, comments, emotions, labels):
        self.news = news          # List of news tensors (pre-padded)
        self.comments = comments  # List of comments (each comment is a list of indices)
        self.emotions = emotions  # List of emotions (2D list: [num_comments_per_news, ...])
        self.labels = labels      # List of labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.news[idx],
            self.comments[idx],
            self.emotions[idx],
            self.labels[idx]
        )


def collate_fn(batch):
    # 解压批次数据
    news, comments, emotions, labels = zip(*batch)

    max_comments = max(len(c) for c in comments)  # 动态适应不同batch
    max_comment_len = max(c.shape[-1] for com in comments for c in com)

    # --- 评论数据填充 ---
    padded_comments = []
    comment_masks = []

    for comment_list in comments:  # 每个新闻的评论列表
        # 截断/填充评论数量到max_comments
        truncated_comments = comment_list[:max_comments]
        padding_needed = max(0, max_comments - len(truncated_comments))

        # 填充到max_comments
        padded = torch.cat([
            truncated_comments,
            torch.zeros(padding_needed, max_comment_len, dtype=torch.long)
        ], dim=0)  # [max_comments, max_comment_len]

        # 生成评论级注意力掩码
        mask = torch.cat([
            torch.ones(len(truncated_comments), dtype=torch.bool),
            torch.zeros(padding_needed, dtype=torch.bool)
        ], dim=0)

        padded_comments.append(padded)
        comment_masks.append(mask)

    comments_tensor = torch.stack(padded_comments)  # [batch, max_comments, max_len]
    comment_masks = torch.stack(comment_masks)  # [batch, max_comments]

    # --- 情感数据填充 ---
    padded_emotions = []
    for emotion_list in emotions:
        truncated_emotions = emotion_list[:max_comments]
        padding_needed = max(0, max_comments - len(truncated_emotions))

        padded = torch.cat([
            truncated_emotions,
            torch.zeros(padding_needed, dtype=torch.float)
        ], dim=0)  # [max_comments, emotion_dim]

        padded_emotions.append(padded)

    emotions_tensor = torch.stack(padded_emotions)  # [batch, max_comments, emotion_dim]

    # --- 其他数据 ---
    news_tensor = torch.stack(news)  # [batch, news_len]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return  news_tensor, comments_tensor, comment_masks, emotions_tensor, labels_tensor
