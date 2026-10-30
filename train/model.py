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



class QwenForSoftClassification(nn.Module):
    """Qwen3 + mean pooling + a linear head over the arms.

    Deliberately logits-only: the loss lives in an Objective, not in the model. That is what
    lets RetrievalRouter and the baselines share this exact architecture while
    differing solely in their training signal.
    """

    def __init__(self, base_model, num_labels: int, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Mean pooling over the sequence, ignoring padding.
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled = torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits}
