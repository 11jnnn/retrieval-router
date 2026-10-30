import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
from datasets import Dataset
from tqdm import tqdm

from train.config import (
    MAX_LENGTH,
    LOG_FILE,
    PREDICTIONS_PATH,
    TEST_DATA_PATH,
    ARM_NAMES,
    EPS,
    device,
)

INFERENCE_BATCH_SIZE = 32

# HuggingFace Hub model ID (override via env for a lambda sweep)
HF_MODEL_ID: str | None = os.getenv("ROUTER_MODEL_ID")


def load_model(model_id: str | None = None):
    """
    Load the RetrievalRouter model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Tuple of (model, tokenizer)
    """
    model_id = model_id or HF_MODEL_ID
    if not model_id:
        raise ValueError("Set ROUTER_MODEL_ID or pass model_id explicitly")

    print("=" * 80)
    print("LOADING MODEL FROM HUGGINGFACE HUB")
    print("=" * 80)
    print(f"Model ID: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)

    model.eval()
    print(f"Model loaded from HuggingFace Hub: {model_id}")

    return model, tokenizer


def load_test_data(test_path: str = None):
    """Load test data from Excel file."""
    if test_path is None:
        test_path = TEST_DATA_PATH

    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)

    test_df = pd.read_excel(test_path)
    print(f"Test samples: {len(test_df)}")

    return test_df


def create_test_dataset(test_df: pd.DataFrame, tokenizer):
    """Tokenize the queries. No targets: inference only needs logits, and requiring a target
    here would tie inference to one objective's columns."""
    ds = Dataset.from_dict({"text": test_df["query"].tolist()})

    ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        ),
        batched=True,
        desc="Tokenizing"
    )

    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds

def compute_metrics(eval_pred) -> dict:
    """
    评估路由器的管线选择结果。

    返回：
    选择最优管线的准确率、分布相似度、
    所选管线的平均评分、理想平均评分、
    以及距离理想结果的差距。
    """

    # 拆出模型预测分数和各管线的真实评分。
    # 两者形状均为 [N, 5]。
    logits, true_ndcgs = eval_pred

    # 将 logits 转成预测概率。
    #
    # dim=-1：对每条查询的五条管线进行 softmax。
    # .numpy()：转换为 NumPy 数组，供后面计算指标。
    pred_probs = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()

    # 将每条查询的真实评分归一化，使其总和约为 1。
    #
    # 对于每一行 r：
    # 如果评分总和 > 0，就让每个评分除以总和。
    # 如果评分全为 0，就使用均匀分布。
    #
    # 这里的归一化只用于计算分布相似度。
    # 它和训练时的 softmax(reward / temperature) 不同。
    label_probs = np.array([
        r / (np.sum(r) + EPS)
        if np.sum(r) > 0
        else np.ones(len(r)) / len(r)
        for r in true_ndcgs
    ])

    # 找到每条查询预测概率最大的管线编号。
    #
    # [N, 5] → [N]
    pred_labels = np.argmax(pred_probs, axis=1)

    # 找到每条查询真实评分最高的管线编号。
    #
    # [N, 5] → [N]
    true_labels = np.argmax(true_ndcgs, axis=1)

    # 比较预测管线和最优管线的编号是否一致。
    # 计算选对最优管线的查询比例。
    accuracy = accuracy_score(true_labels, pred_labels)

    # 逐条查询计算：
    # 模型预测概率 p 与归一化真实评分 l 的余弦相似度。
    #
    # np.dot(p, l)：两个向量的点积。
    # np.linalg.norm(...)：向量的长度，即 L2 范数。
    #
    # 最后对所有查询的余弦相似度求平均。
    cosine = np.mean([
        np.dot(p, l)
        / (
            np.linalg.norm(p) * np.linalg.norm(l)
            + EPS
        )
        for p, l in zip(pred_probs, label_probs)
    ])

    # 对每条查询，读取模型选中管线的真实评分。
    #
    # true_ndcgs[i, pred_labels[i]]：
    # 第 i 条查询，模型所选管线对应的评分。
    #
    # 最后对所有查询求平均。
    predicted_ndcg = np.mean([
        true_ndcgs[i, pred_labels[i]]
        for i in range(len(pred_labels))
    ])

    # 对每条查询，取五条管线中的最高真实评分。
    # 再对所有查询求平均。
    #
    # 代表“每道查询都选中最优管线”的理想结果。
    oracle_ndcg = np.mean(
        np.max(true_ndcgs, axis=1)
    )

    # 理想平均评分减去模型实际选择的平均评分。
    # 表示路由决策造成的平均评分损失。
    regret = oracle_ndcg - predicted_ndcg

    # 将上述差距除以理想平均评分。
    # 得到相对差距。
    normalized_regret = regret / (oracle_ndcg + EPS)

    # 返回所有评估指标。
    return {
        "accuracy": accuracy,
        "cosine_similarity": cosine,
        "predicted_ndcg": predicted_ndcg,
        "oracle_ndcg": oracle_ndcg,
        "regret": regret,
        "normalized_regret": normalized_regret,
    }

@torch.no_grad()
def run_inference(model, tokenizer, test_df: pd.DataFrame, batch_size: int = INFERENCE_BATCH_SIZE) -> np.ndarray:
    """
    Run batched inference on test data.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_df: Test dataframe
        batch_size: Batch size for inference

    Returns:
        Prediction probabilities, shape (num_samples, num_classes)
    """
    print("=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)

    test_dataset = create_test_dataset(test_df, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_logits = []

    for batch in tqdm(dataloader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs["logits"].cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    pred_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    return pred_probs


def evaluate_model(model, tokenizer, test_df: pd.DataFrame, save_predictions: bool = True,
                   objective=None):
    """
    Evaluate model on test set and optionally save predictions.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_df: Test dataframe
        save_predictions: Whether to save predictions to file

    Returns:
        Dictionary of metrics
    """
    from train.objectives import get_objective
    from train.config import OBJECTIVE
    objective = objective or get_objective(OBJECTIVE)

    pred_probs = run_inference(model, tokenizer, test_df)
    # The matrix the method is judged on: rewards for RetrievalRouter, raw nDCG for hard-label
    # baselines that have no reward.
    true_ndcgs = objective.eval_matrix(test_df)

    # Compute metrics
    logits = np.log(pred_probs + EPS)  # Convert back for compute_metrics
    metrics = compute_metrics((logits, true_ndcgs))

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Value':<15}")
    print("-" * 45)
    print(f"{'Accuracy (Pick Best)':<30} {metrics['accuracy']:.4f}")
    print(f"{'Predicted NDCG (Mean)':<30} {metrics['predicted_ndcg']:.4f}")
    print(f"{'Oracle NDCG (Mean)':<30} {metrics['oracle_ndcg']:.4f}")
    print(f"{'Regret (Absolute)':<30} {metrics['regret']:.4f}")
    print(f"{'Regret (Normalized %)':<30} {metrics['normalized_regret']*100:.2f}%")

    if save_predictions:
        # Add predictions to dataframe
        results_df = test_df.copy()
        for i, arm in enumerate(ARM_NAMES):
            results_df[f"pred_prob_{arm}"] = pred_probs[:, i]
            results_df[f"true_ndcg_{arm}"] = true_ndcgs[:, i]

        # Predicted vs True logic
        results_df["predicted_label"] = np.argmax(pred_probs, axis=1)
        results_df["predicted_pipeline"] = [ARM_NAMES[i] for i in results_df["predicted_label"]]

        results_df["true_label"] = np.argmax(true_ndcgs, axis=1)
        results_df["true_pipeline"] = [ARM_NAMES[i] for i in results_df["true_label"]]

        results_df["predicted_ndcg"] = [
            true_ndcgs[i, results_df["predicted_label"].iloc[i]]
            for i in range(len(results_df))
        ]

        results_df["oracle_ndcg"] = np.max(true_ndcgs, axis=1)
        results_df["regret"] = results_df["oracle_ndcg"] - results_df["predicted_ndcg"]

        # Save predictions
        os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
        results_df.to_excel(PREDICTIONS_PATH, index=False)
        print(f"\nPredictions saved to {PREDICTIONS_PATH}")

        # Save metrics log
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics log saved to {LOG_FILE}")

    return metrics


def main():
    """Main inference pipeline."""
    print("\n" + "=" * 80)
    print("RETRIEVALROUTER - INFERENCE")
    print("=" * 80)
    print(f"Model: {HF_MODEL_ID}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load model from HuggingFace Hub
    model, tokenizer = load_model()

    # Load test data
    test_df = load_test_data()

    # Evaluate
    metrics = evaluate_model(model, tokenizer, test_df)

    return metrics


if __name__ == "__main__":
    main()
