import os.path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Model.Cluster import ClusterLoss
from utils.NewsCommentDataset import NewsCommentDataset, collate_fn
from utils.calculate import calculate_macro_f1, calculate_macro_precision, calculate_macro_recall, print_eval, \
    calculate_rmse
from utils.load_embedding import load_embedding
from utils.Tokenizer import MyTokenizer, pad_sequences
from Model.FakeNewsDetect import FakeNewsDetector
from utils.process_data import load_data


class MyModel:

    def __init__(self, config,device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.tokenizer = MyTokenizer(max_vocab_size= self.config.MAX_TOKENIZER_LEN)

        self.data_processor = None
        self.analyzer = None
        self.model = None

    def extract_feature(self, data, save_path):
        # 数据准备
        news, comments, labels, comments_emotion = self.decode_text_and_emotion(data)
        dataset = NewsCommentDataset(news, comments, comments_emotion, labels)
        data_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=16)  # 增大batch_size加速处理

        # 获取聚类数量K
        k = self.model.comments_means.cluster.cluster_centers.detach().cpu().numpy().shape[0]
        cluster_groups = {i: {'features': [], 'labels': [], 'texts': []} for i in range(k)}  # 添加texts字段

        # 特征提取
        self.model.eval()
        with torch.no_grad():
            for news_batch, comments_batch, comments_mask, emotions_batch, labels_batch in data_loader:
                # 设备转移
                news_batch = news_batch.to(self.device)
                comments_batch = comments_batch.to(self.device)
                emotions_batch = emotions_batch.to(self.device)
                comments_mask = comments_mask.to(self.device)
                labels_batch = labels_batch.to(self.device).long().flatten()

                # 前向传播
                _, combined_feat, cluster_assign, _ = self.model(news_batch, comments_batch, comments_mask,
                                                                 emotions_batch)

                # 转换为CPU处理
                feat_np = combined_feat.cpu().numpy()  # [B, N_C, D]
                mask_np = comments_mask.cpu().numpy()  # [B, N_C]
                labels_np = labels_batch.cpu().numpy()  # [B]
                assign_np = cluster_assign.cpu().numpy()  # [B, N_C, K]
                comments_np = comments_batch.cpu().numpy()  # [B, N_C, L]

                # 遍历每个样本
                for b in range(feat_np.shape[0]):
                    valid_idx = np.where(mask_np[b])[0]  # 有效评论索引
                    if len(valid_idx) == 0:
                        continue

                    # 提取有效数据
                    valid_feat = feat_np[b, valid_idx]  # [valid_num, D]
                    valid_assign = assign_np[b, valid_idx]  # [valid_num, K]
                    valid_id = np.argmax(valid_assign, axis=1)  # [valid_num]
                    valid_labels = labels_np[b]  # 标量

                    # 提取对应评论文本
                    valid_comments = comments_np[b, valid_idx]  # [valid_num, L]

                    # 处理每个有效评论
                    for idx_in_valid, cluster_id in enumerate(valid_id):
                        # 获取评论的索引序列并转换为文本
                        comment_indices = valid_comments[idx_in_valid]  # [L]
                        words = [
                            self.tokenizer.idx2word.get(int(idx), "<UNK>")
                            for idx in comment_indices
                            if idx != 0 and idx != 1 # 过滤填充符号
                        ]
                        text = ' '.join(words)

                        # 保存特征、标签和文本
                        cluster_groups[cluster_id]['features'].append(valid_feat[idx_in_valid])
                        cluster_groups[cluster_id]['labels'].append(valid_labels)
                        cluster_groups[cluster_id]['texts'].append(text)

        # 合并所有聚类数据
        all_features = []
        all_clusters = []
        all_news_labels = []
        all_texts = []

        for cluster_id in range(k):
            features = cluster_groups[cluster_id]['features']
            labels = cluster_groups[cluster_id]['labels']
            texts = cluster_groups[cluster_id]['texts']
            if len(features) == 0:
                continue
            all_features.append(features)
            all_clusters.extend([cluster_id] * len(features))
            all_news_labels.extend(labels)
            all_texts.extend(texts)

        all_features = np.concatenate(all_features, axis=0)
        all_clusters = np.array(all_clusters)
        all_news_labels = np.array(all_news_labels)
        all_texts = np.array(all_texts, dtype=object)  # 转换为NumPy数组

        # t-SNE降维
        tsne = TSNE(n_components=3, random_state=42)
        tsne_coords = tsne.fit_transform(all_features)

        # 保存数据（包含评论文本）
        np.savez_compressed(
            save_path,
            tsne_coords=tsne_coords,
            features=all_features,
            cluster_ids=all_clusters,
            news_labels=all_news_labels,
            comment_texts=all_texts  # 新增评论文本
        )

        print(f"特征已保存至：{save_path}")
        print(f"包含数据项：{', '.join(np.load(save_path).files)}")

        return tsne_coords, all_features, all_clusters, all_news_labels, all_texts


    def test(self,test_data):
        news, comments, labels, comments_emotion = self.decode_text_and_emotion(test_data)
        test_dataset =  NewsCommentDataset(news, comments, comments_emotion, labels)
        test_loader = DataLoader(test_dataset ,collate_fn=collate_fn, batch_size=1)
        test_metrics = self.evaluate(test_loader)
        print_eval(test_metrics)

    def train(self, train_data, val_data, batch_size, lr, epochs ,embedding_path, saved_dir):
        self._word2index_on_news_and_comment(train_data, val_data)
        # 创建数据加载器
        train_dataset = NewsCommentDataset(self.train_news, self.train_comments, self.train_emotions, self.train_labels)
        val_dataset = NewsCommentDataset(self.val_news, self.val_comments, self.val_emotions, self.val_labels)

        # 创建DataLoader，使用处理后的数据集
        train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)

        self.model = self._build_model(embedding_path, self.tokenizer.word2idx)
        self.model.to(self.device)
        # 定义优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        cluster_loss_fn = ClusterLoss(alpha=self.config.alpha)
        # 训练循环
        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            cls_loss = 0.0
            cluster_loss = 0.0
            # 训练阶段
            for news, comments, comments_mask, emotions, labels in train_loader:
                # 检查是否存在 NaN 或 inf
                assert not torch.isnan(news).any(), "NaN in news data"
                assert not torch.isinf(news).any(), "Inf in news data"

                assert not torch.isnan(comments).any(), "NaN in comments data"
                assert not torch.isinf(comments).any(), "Inf in comments data"

                assert not torch.isnan(emotions).any(), "NaN in emotions data"
                assert not torch.isinf(emotions).any(), "Inf in emotions data"

                assert not torch.isnan(labels).any(), "NaN in labels data"
                assert not torch.isinf(labels).any(), "Inf in labels data"

                news = news.to(self.device)
                comments = comments.to(self.device)
                emotions = emotions.to(self.device)
                comments_mask = comments_mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs, combined_comment_feat, cluster_assign, cluster_centers = self.model(news, comments, comments_mask,emotions)


                # 计算损失
                cls_loss = criterion(outputs, labels)
                cluster_loss = cluster_loss_fn(combined_comment_feat, cluster_assign,cluster_centers)
                total_loss = cls_loss + cluster_loss
                total_loss.backward()
                # # 添加梯度裁剪
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 评估阶段
            val_metrics = self.evaluate(val_loader)
            print(30*"*")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {total_loss:.4f} | cls_loss: {cls_loss:.4f} | cluster_loss: {cluster_loss:.4f}")
            print_eval(val_metrics)

            # 保存最佳模型
            if val_metrics['accuracy'] > best_acc:  # 保持使用准确率作为保存依据
                best_acc = val_metrics['accuracy']
                self.save_model(self.model, os.path.join(saved_dir, f"best_model.pth_{self.config.num_clusters}_{self.config.alpha}"))

        # 最终保存
        self.save_model(self.model, os.path.join(saved_dir,f"final_model.pth_{self.config.num_clusters}_{self.config.alpha}"))
        return

    def _build_model(self, embedding_path, word_index):
        embedding_tensor = load_embedding(embedding_path, self.config.embedding_dim, word_index)
        if embedding_tensor.device.type != 'cpu':
            embedding_tensor = embedding_tensor.cpu()
        model = FakeNewsDetector(
            embedding_matrix=embedding_tensor,
            news_hidden_dim=self.config.news_hidden_dim,
            comments_hidden_dim=self.config.comments_hidden_dim,
            combined_comment_mlp_dim=self.config.combined_comment_mlp_dim,
            comments_emotion_dim=self.config.comments_emotion_dim,
            num_clusters=self.config.num_clusters,
            class_num=self.config.class_num
        )
        return model

    def save_model(self, model, path):
        """保存模型和tokenizer"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': {
                'word2idx': self.tokenizer.word2idx,
                'max_vocab_size': self.tokenizer.max_vocab_size
            }
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, model_path, emotion_analyzer=None, data_processor=None, device='cuda'):
        """加载模型和tokenizer"""
        checkpoint = torch.load(model_path, map_location=device)
        # 重建tokenizer
        tokenizer = MyTokenizer(max_vocab_size=checkpoint['tokenizer']['max_vocab_size'])
        tokenizer.word2idx = checkpoint['tokenizer']['word2idx']
        tokenizer.idx2word = {v: k for k, v in tokenizer.word2idx.items()}
        self.model = FakeNewsDetector(
            embedding_matrix=torch.zeros((len(tokenizer.word2idx), self.config.embedding_dim)),
            news_hidden_dim=self.config.news_hidden_dim,
            comments_hidden_dim=self.config.comments_hidden_dim,
            combined_comment_mlp_dim=self.config.combined_comment_mlp_dim,
            comments_emotion_dim=self.config.comments_emotion_dim,
            num_clusters=self.config.num_clusters,
            class_num=self.config.class_num
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.analyzer = emotion_analyzer # for predict
        self.data_processor = data_processor # for predict

    def predict(self, news_text, comments):
        """
        预测新闻和评论的真实性
        :param news_text: 新闻文本（字符串）
        :param comments: 评论列表（每个元素是字符串）
        :return: 预测标签（0/1），预测概率（0~1）
        """
        # 确保模型和tokenizer已加载
        if self.model is None or self.tokenizer is None or self.analyzer is None:
            raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
        # 提取情感特征
        emotions = []
        cluster_groups = {i: {'features': [], 'labels': [], 'texts': []} for i in range(3)}
        for comment in comments:
            emotion = self.analyzer.predict(comment)
            emotions.append(emotion["confidence"]["positive"])

        # 将输入数据处理为模型需要的格式
        news_text = self.data_processor.preprocess_text(news_text)
        process_comments = [self.data_processor.preprocess_text(comment) for comment in comments]
        self.model.eval()
        with torch.no_grad():
            news_seq = self.texts_to_padded_sequences([news_text], self.config.MAX_NEWS_LENGTH)[0]  # 新闻处理
            comment_seqs = self.texts_to_padded_sequences(process_comments, self.config.MAX_COMMENT_LENGTH)  # 每条评论转序列

            news_tensor = torch.LongTensor(news_seq).unsqueeze(0).to(self.device)  # shape: [1, max_news_len]
            comments_tensor = torch.LongTensor(comment_seqs).unsqueeze(0).to(
                self.device)  # shape: [1, comment_count, max_comment_len]
            emotions_tensor = torch.FloatTensor(emotions).unsqueeze(0).to(
                self.device)  # shape: [1, comment_count, emotion_dim]

            #  模型预测
            outputs, _, cluster_assign, _ = self.model(news_tensor, comments_tensor, comment_emotion=emotions_tensor)
            assign_np = cluster_assign.squeeze(0).cpu().numpy()  # [B, N_C, K]
            # outputs = outputs.squeeze(0).cpu().numpy()

            valid_id = np.argmax(assign_np, axis=1)
            reason_comment = []

            for i in range(len(valid_id)):
                if valid_id[i] == 2:
                    reason_comment.append(i)


            probs = torch.softmax(outputs, dim=1)[0, 1].item()

            pred_label = 1 if probs >= 0.5 else 0

        return pred_label, probs, reason_comment

    def evaluate(self, data_loader):
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for news, comments, comments_mask, emotions, labels in data_loader:
                news = news.to(self.device)
                comments = comments.to(self.device)
                emotions = emotions.to(self.device)
                comments_mask = comments_mask.to(self.device)
                # 修正标签处理为长整型类别索引
                labels = labels.to(self.device).long().flatten()  # 确保形状为(batch,)

                outputs, *_ = self.model(news, comments, comments_mask, emotions)

                # 获取类别概率
                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs)
                all_labels.append(labels)

        probs_tensor = torch.cat(all_probs)
        labels_tensor = torch.cat(all_labels)

        # 计算预测结果
        preds_tensor = torch.argmax(probs_tensor, dim=1)
        labels_tensor = labels_tensor.long()

        # 计算分类指标
        accuracy = (preds_tensor == labels_tensor).float().mean().item()
        macro_f1 = calculate_macro_f1(labels_tensor, preds_tensor, self.config.class_num)
        macro_pre = calculate_macro_precision(labels_tensor, preds_tensor, self.config.class_num)
        macro_rec = calculate_macro_recall(labels_tensor, preds_tensor, self.config.class_num)
        rmse = calculate_rmse(labels_tensor, preds_tensor)

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_precision": macro_pre,
            "macro_recall": macro_rec,
            "rmse": rmse
        }



    def _word2index_on_news_and_comment(self, train_data, val_data):
        all_texts = []
        # 遍历训练数据
        for data in train_data:
            all_texts.append(data['news'])
            all_texts.extend(data['comments'])

        # 遍历测试数据
        for data in val_data:
            all_texts.append(data['news'])
            all_texts.extend(data['comments'])
        self.tokenizer.fit_on_texts(all_texts)
        self._reverse_word_index()

        # 处理训练数据
        self.train_news, self.train_comments, self.train_labels, self.train_emotions = self.decode_text_and_emotion(train_data)
        # 处理测试数据
        self.val_news, self.val_comments, self.val_labels, self.val_emotions= self.decode_text_and_emotion(val_data)


    def decode_text_and_emotion(self,datas):
        # 处理训练数据
        news = []
        comments = []
        comments_emotion = []
        labels = []
        for data in datas:
            # 处理新闻
            news_seq = self.texts_to_padded_sequences([data['news']], self.config.MAX_NEWS_LENGTH)[0]
            news.append(news_seq)

            # 处理评论
            comment_seqs = self.texts_to_padded_sequences(data['comments'], self.config.MAX_COMMENT_LENGTH)
            comments.append(comment_seqs)
            comments_emotion.append(torch.tensor(data['emotions']))
            # 保存标签
            labels.append(data['label'])

        return  news, comments, labels, comments_emotion

    def texts_to_padded_sequences(self,texts, max_length):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=max_length,
            padding='post',
            truncating='post'
        )
        return padded

    def _reverse_word_index(self):
        self.word_index = {value: key for key, value in self.tokenizer.word2idx.items()}
