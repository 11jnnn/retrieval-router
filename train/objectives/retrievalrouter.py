"""RetrievalRouter's reward-vector training objective.

Target: the full per-query reward vector r_i = (1-L)*nDCG_i + L*(1 - normalised_latency_i),
turned into a target distribution with softmax(r / T). The model therefore learns *how much* better
each arm is, not merely which one wins.

Loss: KL(target || pi), computed as cross-entropy against the target distribution. Samples with
an all-zero reward vector are masked out. At lambda greater than zero, the efficiency term keeps
all-fail queries in training by providing a nonzero signal.
"""
import numpy as np
import torch
import torch.nn.functional as F

from train.config import REWARD_COLS, TEMPERATURE
from train.objectives import Objective

def retrievalrouter_loss(logits, labels):
    """
    将奖励转换为目标分布，计算软标签交叉熵。

    对模型参数而言，这与最小化
    KL(目标分布 || 模型预测分布) 的梯度相同。

    最后只对具有有效奖励信号的查询求平均。
    """

    # labels：每条查询对应的五个奖励。
    #
    # 先除以温度 TEMPERATURE，再做 softmax，
    # 得到五条管线的目标概率分布。
    #
    # TEMPERATURE 在仓库中设为 0.1。
    # 较小的温度会放大奖励差异，使目标分布更集中。
    #
    # dim=-1：沿最后一维，也就是五条管线的方向计算。
    #
    # 输入和输出形状均为 [B, 5]。
    target_distribution = F.softmax(
        labels / TEMPERATURE,
        dim=-1,
    )

    # 将模型输出的 logits 转换成“对数概率”。
    #
    # 相当于 log(softmax(logits))，
    # 直接使用 log_softmax 在数值计算上更稳定。
    #
    # 形状仍为 [B, 5]。
    log_probs = F.log_softmax(logits, dim=-1)

    # 对每条查询计算软标签交叉熵：
    #
    # loss = -Σ(目标概率 × 模型预测概率的对数)
    #
    # 先逐元素相乘，再对五条管线求和。
    #
    # 形状变化：
    # [B, 5] → [B]
    #
    # 结果是每条查询分别对应一个损失。
    per_sample = -(
        target_distribution * log_probs
    ).sum(dim=-1)

    # 找出每条查询的最高奖励。
    #
    # labels.max(dim=-1) 返回：
    # 1. 每行最大值
    # 2. 最大值对应的下标
    #
    # [0] 取的是“最大值”这部分，
    # 不是取第一条查询。
    #
    # 最高奖励 > 0：认为这条查询有训练信号。
    # 最高奖励 = 0：排除这条查询。
    #
    # .float() 将布尔值转成：
    # True → 1.0
    # False → 0.0
    #
    # mask 形状为 [B]。
    mask = (labels.max(dim=-1)[0] > 0).float()

    # per_sample * mask：
    # 将无效查询的损失乘以 0。
    #
    # 分子：有效查询的损失总和。
    # 分母：有效查询的数量。
    #
    # clamp(min=1)：
    # 当整个批次都没有有效查询时，避免除以 0。
    #
    # 最终返回一个标量损失。
    return (
        (per_sample * mask).sum()
        / mask.sum().clamp(min=1)
    )


# 继承 Objective 接口，定义 RetrievalRouter 使用的训练目标。
class RetrievalRouterObjective(Objective):

    # 训练目标的名称，用于配置和选择。
    name = "retrievalrouter"

    # labels 保存的是小数奖励，因此使用 float32。
    label_dtype = torch.float32

    # 在原始数据表中添加训练目标。
    def add_targets(self, df):

        # 将导入延迟到调用这个函数时，
        # 用于处理模块之间的循环依赖。
        from train.prepare_data import calculate_rewards

        # 计算五条管线的奖励，并写入表格。
        return calculate_rewards(df)

    # 从表格中取出训练标签。
    def targets(self, df):

        # df[REWARD_COLS]：
        # 取出五条管线的奖励列。
        #
        # .to_numpy()：
        # 转成 [N, 5] 的 NumPy 数组。
        #
        # 遍历每一行，并转成 Python 列表。
        #
        # 返回形式：
        # [
        #     [查询1的五个奖励],
        #     [查询2的五个奖励],
        #     ...
        # ]
        return [
            list(r)
            for r in df[REWARD_COLS].to_numpy()
        ]

    # 训练器通过这个方法计算损失。
    def loss(self, logits, labels):

        # 调用上面定义的损失函数。
        return retrievalrouter_loss(logits, labels)

    # 返回验证模型时使用的评分矩阵。
    # “-> np.ndarray” 是返回类型提示。
    def eval_matrix(self, df) -> np.ndarray:

        # 返回每条查询、每条管线对应的原始奖励。
        #
        # 形状为 [N, 5]。
        # 验证时，根据模型选中的管线，
        # 读取对应奖励，计算平均值。
        return df[REWARD_COLS].to_numpy()


def retrievalrouter_loss(logits, labels):
    """KL onto softmax(reward / T), averaged over queries that have any signal."""
    target_distribution = F.softmax(labels / TEMPERATURE, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -(target_distribution * log_probs).sum(dim=-1)

    # Exclude only all-zero reward vectors. At lambda > 0, efficiency keeps rewards nonzero.
    mask = (labels.max(dim=-1)[0] > 0).float()
    return (per_sample * mask).sum() / mask.sum().clamp(min=1)


class RetrievalRouterObjective(Objective):
    name = "retrievalrouter"
    label_dtype = torch.float32

    def add_targets(self, df):
        from train.prepare_data import calculate_rewards   # local: avoids a circular import
        return calculate_rewards(df)

    def targets(self, df):
        return [list(r) for r in df[REWARD_COLS].to_numpy()]

    def loss(self, logits, labels):
        return retrievalrouter_loss(logits, labels)

    def eval_matrix(self, df) -> np.ndarray:
        # Best epoch = highest reward-of-chosen-arm, i.e. the quantity we actually optimise.
        return df[REWARD_COLS].to_numpy()
