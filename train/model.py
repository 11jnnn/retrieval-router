import torch
import torch.nn as nn
# num_labels：分类数量。
# 在 RetrievalRouter 中为 5，对应五条可选检索管线。
# ": int" 是类型提示，表示预期传入整数

def build_model(num_labels: int):
    """
    构建路由模型：
    Qwen3 + LoRA + 平均池化 + 线性分类层。
    """

    # LoraConfig：配置 LoRA 的参数。
    # get_peft_model：根据配置，给基础模型加入 LoRA 模块。
    # TaskType：指定 PEFT 包装模型时使用的任务类型。
    from peft import LoraConfig, get_peft_model, TaskType

    # AutoModel：根据模型名称，加载对应的预训练模型主干。
    from transformers import AutoModel

    # 从配置文件读取模型和 LoRA 设置。
    #
    # MODEL_NAME：基础模型名称，这里是 Qwen/Qwen3-0.6B-Base。
    # DTYPE：模型参数的数据类型，例如 bfloat16 或 float32。
    # device：模型运行的设备，例如 cuda 或 cpu。
    #
    # LORA_R：LoRA 的秩，这里为 16。
    # LORA_ALPHA：LoRA 的缩放参数，这里为 32。
    # LORA_DROPOUT：LoRA 分支的 dropout，这里为 0.05。
    # TARGET_MODULES：需要加入 LoRA 的模块名称列表。
    from train.config import (
        MODEL_NAME,
        DTYPE,
        device,
        LORA_R,
        LORA_ALPHA,
        LORA_DROPOUT,
        TARGET_MODULES,
    )

    # 加载 Qwen3 预训练主干。
    #
    # MODEL_NAME：指定加载哪个模型。
    # trust_remote_code=True：允许使用模型仓库提供的自定义代码。
    # dtype=DTYPE：指定加载参数时使用的数据类型。
    # .to(device)：将模型移动到指定设备。
    base = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=DTYPE,
    ).to(device)

    # 给 Qwen3 主干加入 LoRA。
    #
    # get_peft_model 会包装基础模型，加入可训练的 LoRA 参数，
    # 并冻结原始主干参数。
    #
    # 此处重新赋值给 base，后面的 base 就是加入 LoRA 后的模型。
    base = get_peft_model(
        base,
        LoraConfig(
            # LoRA 使用的低秩维度。
            r=LORA_R,

            # 控制 LoRA 权重增量的缩放程度。
            lora_alpha=LORA_ALPHA,

            # LoRA 分支中的 dropout 概率。
            lora_dropout=LORA_DROPOUT,

            # 指定在哪些模块中加入 LoRA。
            target_modules=TARGET_MODULES,

            # 不额外训练基础模型中的 bias 参数。
            bias="none",

            # 按“特征提取”任务包装基础模型。
            # 后续使用 Qwen3 的隐藏状态进行分类。
            task_type=TaskType.FEATURE_EXTRACTION,
        ),
    )

    # 使用作者定义的分类模型类，组装完整路由器。
    #
    # 这个类内部负责：
    # 1. 调用加入 LoRA 的 Qwen3，获得 token 隐藏状态。
    # 2. 对真实 token 的隐藏状态进行平均池化。
    # 3. 通过线性层得到各条管线的分类分数。
    model = QwenForSoftClassification(
        # 加入 LoRA 后的 Qwen3 主干。
        base_model=base,

        # 分类层输出的分数数量，这里为 5。
        num_labels=num_labels,

        # 从模型配置中读取隐藏状态维度。
        # 这里的 Qwen3-0.6B 对应 1024 维。
        hidden_size=base.config.hidden_size,

        # 平均池化之后、分类层之前的 dropout。
        dropout=0.1,
    ).to(device)

    # 返回完整模型对象，供后续训练或推理调用。
    return model

# 继承 PyTorch 的 nn.Module，定义一个模型类。
class QwenForSoftClassification(nn.Module):
    """
    使用 Qwen3 提取查询特征，再通过平均池化和线性层，
    得到各条检索管线的分类分数。

    这个类负责输出 logits。
    训练损失由外部的 Objective 计算。
    """

    # 创建模型对象时执行 __init__。
    #
    # base_model：
    # 上一段代码传入的、已经加入 LoRA 的 Qwen3。
    #
    # num_labels：
    # 分类数量。RetrievalRouter 中为 5。
    #
    # hidden_size：
    # 每个 token 的隐藏状态维度，这里为 1024。
    #
    # dropout：
    # 分类前的 dropout 概率，默认 0.1。
    def __init__(
        self,
        base_model,
        num_labels: int,
        hidden_size: int = 1024,
        dropout: float = 0.1,
    ):
        # 初始化父类 nn.Module。
        # 使 PyTorch 可以管理这个模型的子模块和参数。
        super().__init__()

        # 保存 Qwen3 + LoRA，后面用它处理查询。
        self.base_model = base_model

        # 创建 Dropout 层。
        # 训练时随机将约 10% 的输入分量置零，
        # 用于降低过拟合。
        self.dropout = nn.Dropout(dropout)

        # 创建线性分类层：
        # 输入：1024 维查询向量。
        # 输出：5 个管线分数。
        self.classifier = nn.Linear(hidden_size, num_labels)

        # 初始化分类层权重。
        # 从均值为 0、标准差为 0.02 的正态分布中取值。
        nn.init.normal_(self.classifier.weight, std=0.02)

        # 将分类层偏置初始化为 0。
        nn.init.zeros_(self.classifier.bias)

    # @property 让调用者可以通过 model.device 读取设备，
    # 而不需要写 model.device()。
    @property
    def device(self):
        # self.parameters() 返回模型参数的迭代器。
        # next(...) 取出第一个参数。
        # .device 查看这个参数位于 CPU 还是 GPU。
        return next(self.parameters()).device

    # 每次调用 model(...) 时，PyTorch 会执行 forward。
    #
    # input_ids：
    # 查询的 token 编号，形状为 [B, L]。
    #
    # attention_mask：
    # 标记真实 token 和补齐位置，形状为 [B, L]。
    # 真实 token 为 1，补齐位置为 0。
    #
    # B：批次中的查询数量。
    # L：该批次补齐后的序列长度。
    def forward(self, input_ids, attention_mask):

        # 将查询交给 Qwen3 处理。
        # attention_mask 让模型识别哪些位置是补齐位置。
        # output_hidden_states=True 要求返回各层隐藏状态。
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # 取最后一层的隐藏状态。
        #
        # 形状：[B, L, 1024]
        # 每条查询的每个 token 都对应一个 1024 维向量。
        hidden_states = outputs.last_hidden_state

        # 将 attention_mask 扩展为与隐藏状态相同的形状。
        #
        # unsqueeze(-1)：
        # [B, L] → [B, L, 1]
        #
        # expand(hidden_states.size())：
        # [B, L, 1] → [B, L, 1024]
        #
        # float()：
        # 将掩码转换为浮点数，方便参与乘法和除法。
        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(hidden_states.size())
            .float()
        )

        # 进行忽略补齐位置的平均池化。
        #
        # hidden_states * mask：
        # 将补齐位置的向量乘以 0。
        #
        # torch.sum(..., dim=1)：
        # 沿 token 维度求和，将 [B, L, 1024]
        # 压缩为 [B, 1024]。
        #
        # mask.sum(dim=1)：
        # 得到有效 token 数量，在每个特征维度上重复。
        #
        # torch.clamp(..., min=1e-9)：
        # 保证分母至少为 1e-9，避免除以 0。
        pooled = (
            torch.sum(hidden_states * mask, dim=1)
            / torch.clamp(mask.sum(dim=1), min=1e-9)
        )

        # 先对查询向量执行 Dropout，
        # 再通过线性分类层得到五个分数。
        #
        # [B, 1024] → [B, 5]
        logits = self.classifier(self.dropout(pooled))

        # 返回字典，供外部训练或推理代码使用。
        return {"logits": logits}

