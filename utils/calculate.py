import torch
import torch.nn.functional as F

def calculate_macro_precision(labels, preds, num_classes):
    """
    计算宏平均精确率（Macro-average Precision）
    :param labels: 真实标签张量（一维）
    :param preds: 预测标签张量（一维）
    :param num_classes: 总类别数
    :return: 宏平均精确率（百分比）
    """
    precision_sum = 0.0
    for cls in range(num_classes):
        # 计算真阳性（TP）和假阳性（FP）
        tp = ((preds == cls) & (labels == cls)).sum().float()
        fp = ((preds == cls) & (labels != cls)).sum().float()

        # 处理分母为零的情况
        denominator = tp + fp
        precision_cls = tp / denominator if denominator > 0 else 0.0
        precision_sum += precision_cls

    return (precision_sum / num_classes).item()

def calculate_rmse(predictions, targets):
    predictions = predictions.float()
    targets = targets.float()
    rmse = torch.sqrt(F.mse_loss(targets,predictions))
    return rmse.item()


def calculate_macro_recall(labels, preds, num_classes):
    """
    计算宏平均召回率（Macro-average Recall）
    :param labels: 真实标签张量（一维）
    :param preds: 预测标签张量（一维）
    :param num_classes: 总类别数
    :return: 宏平均召回率
    """
    recall_sum = 0.0
    for cls in range(num_classes):
        # 计算真阳性（TP）和假阴性（FN）
        tp = ((preds == cls) & (labels == cls)).sum().float()
        fn = ((labels == cls) & (preds != cls)).sum().float()

        # 处理分母为零的情况
        denominator = tp + fn
        recall_cls = tp / denominator if denominator > 0 else 0.0
        recall_sum += recall_cls

    return (recall_sum / num_classes).item()


def calculate_macro_f1(labels, preds, num_classes):
    """
    计算宏平均F1分数（Macro-average F1）
    :param labels: 真实标签张量（一维）
    :param preds: 预测标签张量（一维）
    :param num_classes: 总类别数
    :return: 宏平均F1分数
    """
    f1_sum = 0.0
    for cls in range(num_classes):
        # 计算TP/FP/FN
        tp = ((preds == cls) & (labels == cls)).sum().float()
        fp = ((preds == cls) & (labels != cls)).sum().float()
        fn = ((labels == cls) & (preds != cls)).sum().float()

        # 计算精确率和召回率
        precision_denominator = tp + fp
        precision = tp / precision_denominator if precision_denominator > 0 else 0.0

        recall_denominator = tp + fn
        recall = tp / recall_denominator if recall_denominator > 0 else 0.0

        # 计算F1分数
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        f1_sum += f1

    return (f1_sum / num_classes).item()

def print_eval(val_metrics):
    print(
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Macro-F1: {val_metrics['macro_f1']:.4f} | "
        f"Precision: {val_metrics['macro_precision']:.4f} | "
        f"Recall: {val_metrics['macro_recall']:.4f} | "
        f"RMSE: {val_metrics['rmse']:.4f}"
    )