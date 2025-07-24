import torch
import torch.nn as nn
import torch.nn.functional as F
class ClusterLayer(nn.Module):
    def __init__(self, num_clusters, mlp_dim):
        super().__init__()
        self.num_clusters = num_clusters

        # 可学习的聚类中心
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, mlp_dim),
            requires_grad=True
        )

        # 相似度计算参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        """
        输入:
            x: [B, N_C, mlp_dim] 转换后的评论特征
        输出:
            cluster_assign: [B, N_C, K] 聚类分配概率
            cluster_feat:   [B, K, mlp_dim] 聚合后的聚类特征
        """
        B, N_C, _ = x.shape
        self.temperature.data.clamp_(min=0.1, max=5.0)

        # 正则化
        x_norm = F.normalize(x, p=2, dim=-1)  # [B, N_C, D]
        centers_norm = F.normalize(self.cluster_centers, p=2, dim=-1)  # [K, D]

        # 计算相似度 (余弦相似度)
        similarity = torch.matmul(x_norm, centers_norm.transpose(0, 1))  # [B, N_C, K]

        if mask is not None:
            mask = mask.unsqueeze(-1)
            # 无效位置填充负无穷（softmax后接近0）
            similarity = similarity.masked_fill(mask == 0, -1e9)

        # 软聚类分配
        cluster_assign = F.softmax(similarity * self.temperature, dim=-1) # [B, N_C, K]

        # 特征聚合 （对每个样本的所有评论，按每个聚类的分配权重加权求和）
        cluster_feat = torch.einsum('bnk,bnd->bkd', cluster_assign, x)  # [B, K, D]

        return cluster_assign, cluster_feat



class ClusterLoss(nn.Module):
    def __init__(self, alpha=0.05, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps  # 防止除零的小量

    def forward(self, combined_comment_feat, cluster_assign, cluster_centers):
        """
        输入:
            combined_comment_feat: [B, N_C, D] 原始评论特征
            cluster_assign:        [B, N_C, K] 聚类分配概率
            cluster_centers:      [K, D] 可学习的聚类中心
        输出:
            loss: 标量损失值
        """
        B, N_C, D = combined_comment_feat.shape
        K = cluster_centers.shape[0]

        # --- 类内损失：加权特征到中心的距离 ---
        # 归一化特征和中心（与 ClusterLayer 一致）
        x_norm = F.normalize(combined_comment_feat, p=2, dim=-1)  # [B, N_C, D]
        centers_norm = F.normalize(cluster_centers, p=2, dim=-1)  # [K, D]
        # 计算每个特征到所有中心的余弦相似度
        distance_matrix = torch.matmul(x_norm, centers_norm.transpose(0, 1))
        # 加权平均
        embedding_loss = - torch.sum(cluster_assign * distance_matrix) / (B * N_C)



        # --- 类间损失：中心间距离的倒数 ---
        # 计算所有中心间的距离矩阵 [K, K]
        distance_center = torch.matmul(centers_norm, centers_norm.transpose(0,1))
        mask = torch.eye(K, dtype=torch.bool, device=distance_center.device)
        valid_center_distances = distance_center[~mask].view(K, K - 1)  # [K, K-1]
        center_loss = valid_center_distances.mean()

        # 总损失
        total_loss = self.alpha * (center_loss + embedding_loss)
        return total_loss