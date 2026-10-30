import pandas as pd
from train.config import (
    DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    SAMPLE_RATE, RANDOM_SEED, LAMBDA, OBJECTIVE,
    NDCG_COLS, LATENCY_COLS, REWARD_COLS
)

def load_and_split_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH).dropna()
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "query"}, inplace=True)

    print(f"Original dataset shape: {df.shape}")
    print(f"Datasets present: {df['DATASET'].unique().tolist()}")

    train_dfs = []
    val_dfs = []
    test_dfs = []

    for dataset_name in df['DATASET'].unique():
        dataset_df = df[df['DATASET'] == dataset_name]
        total_len = len(dataset_df)
        
        # Calculate sizes
        n_test = int(total_len * SAMPLE_RATE)
        n_val = int(total_len * SAMPLE_RATE)

        # Safety check for small datasets
        if n_test == 0 or n_val == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' has {total_len} samples. "
                f"Sampling {SAMPLE_RATE*100}% results in 0 samples for test/val. "
                f"Increase SAMPLE_RATE or ensure dataset has enough samples."
            )
        
        if (n_test + n_val) >= total_len:
             raise ValueError(
                f"Dataset '{dataset_name}' is too small for current split rates. "
                f"Total: {total_len}, Test+Val required: {n_test + n_val}"
            )

        # 1. Sample Test set from full data
        test_df = dataset_df.sample(n=n_test, random_state=RANDOM_SEED)
        
        # 2. Drop Test samples to get remaining pool
        remaining_df = dataset_df.drop(test_df.index)
        
        # 3. Sample Validation set from remaining pool
        val_df = remaining_df.sample(n=n_val, random_state=RANDOM_SEED)
        
        # 4. Remaining samples go to Train
        train_df = remaining_df.drop(val_df.index)

        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

        print(f"  {dataset_name}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    train_data = pd.concat(train_dfs, ignore_index=True)
    val_data = pd.concat(val_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)

    print(f"\nTrain dataset shape: {train_data.shape}")
    print(f"Val dataset shape:   {val_data.shape}")
    print(f"Test dataset shape:  {test_data.shape}")

    return train_data, val_data, test_data


def calculate_rewards(df):
    """
    为每条查询计算五条检索管线的奖励。

    奖励由两部分组成：
    1. 检索质量：ndcg
    2. 检索效率：1 - normalized_latency

    LAMBDA 控制两部分的权重。
    """

    # NDCG_COLS 是五条管线的 nDCG 列名列表。
    # df[NDCG_COLS]：从表格中选出这五列。
    # .values：将选出的表格转换为 NumPy 数组。
    #
    # 假设有 N 条查询，得到的形状是 [N, 5]。
    # 每一行：一道查询。
    # 每一列：一条检索管线。
    ndcg_values = df[NDCG_COLS].values

    # 取出五条管线的检索耗时，同样得到 [N, 5]。
    #
    # 注意：耗时列与 nDCG 列必须按照同样的管线顺序排列。
    # 这样，第 i 列的质量和耗时才属于同一条管线。
    latency_values = df[LATENCY_COLS].values

    # 对每一行的五个耗时求和。
    #
    # axis=1：沿着列的方向求和，即分别处理每一道查询。
    # keepdims=True：保留二维形状。
    #
    # 形状变化：
    # [N, 5] → [N, 1]
    latency_sum = latency_values.sum(axis=1, keepdims=True)

    # 每条管线的耗时，除以这道查询对应的五条管线耗时总和。
    #
    # NumPy 会自动把每行的总耗时应用到该行的五个元素上，
    # 这个过程叫作“广播”。
    #
    # 1e-9 = 0.000000001，用于避免分母为 0。
    #
    # 结果形状仍为 [N, 5]。
    normalized_latency = latency_values / (latency_sum + 1e-9)

    # 逐个元素计算奖励。
    #
    # (1 - LAMBDA) * ndcg_values：
    # 检索质量贡献。nDCG 越高，这部分越大。
    #
    # LAMBDA * (1 - normalized_latency)：
    # 检索效率贡献。耗时越短，这部分越大。
    #
    # 最终得到 [N, 5] 的奖励数组。
    rewards = (
        (1 - LAMBDA) * ndcg_values
        + LAMBDA * (1 - normalized_latency)
    )

    # 将五列奖励写回原始表格。
    #
    # enumerate(REWARD_COLS) 会依次得到：
    # i=0，第一个奖励列名；
    # i=1，第二个奖励列名；
    # ...
    #
    # rewards[:, i]：
    # 取出所有查询在第 i 条管线上的奖励。
    for i, col in enumerate(REWARD_COLS):
        df[col] = rewards[:, i]

    # 建立“奖励列名 → 管线编号”的字典。
    #
    # 例如：
    # {
    #     "MULTIMODAL_RERANK_reward": 0,
    #     "MULTIMODAL-SINGLE_reward": 1,
    #     "TEXT_RERANK_reward": 2,
    #     "TEXT-SINGLE_reward": 3,
    #     "BM25_reward": 4,
    # }
    arm_index = {
        col: i for i, col in enumerate(REWARD_COLS)
    }

    # 对每一道查询，取五个奖励中的最大值。
    #
    # 例如奖励为：
    # [0.92, 0.808, 0.864, 0.736, 0.512]
    #
    # 这一行的 optimal_reward 就是 0.92。
    df["optimal_reward"] = df[REWARD_COLS].max(axis=1)

    # idxmax(axis=1)：
    # 找出每行最大奖励对应的“列名”。
    #
    # .map(arm_index)：
    # 将这个列名转换成管线编号。
    #
    # 例如：
    # "MULTIMODAL_RERANK_reward" → 0
    df["optimal_arm"] = (
        df[REWARD_COLS]
        .idxmax(axis=1)
        .map(arm_index)
    )

    # 返回添加了奖励列、最优奖励和最优管线编号的表格。
    # 这个函数也直接修改了传入的 df。
    return df


def save_datasets(train_df, val_df, test_df, paths=None):
    train_path, val_path, test_path = paths or (TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH)
    print(f"\nSaving datasets...")
    train_df.to_excel(train_path, index=False)
    print(f"Saved training data to {train_path}")

    val_df.to_excel(val_path, index=False)
    print(f"Saved validation data to {val_path}")

    test_df.to_excel(test_path, index=False)
    print(f"Saved test data to {test_path}")


def main(objective=None, paths=None):
    """Split the data, then let the objective attach its own training target.

    The split is seeded and objective-independent, so every method -- ours and any baseline --
    trains and is evaluated on exactly the same queries.
    """
    from train.objectives import get_objective   # local: prepare_data <-> objectives cycle
    objective = objective or get_objective(OBJECTIVE)

    print(f"=== Dataset Preparation (objective: {objective.name}) ===\n")

    # Step 1: Split (shared, seeded, identical for every objective)
    train_df, val_df, test_df = load_and_split_data()

    # Step 2: Attach this objective's target
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\nBuilding targets for {name} set...")
        objective.add_targets(df)

    # Step 3: Save
    save_datasets(train_df, val_df, test_df, paths)

    print("\n=== Dataset Preparation Complete ===")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")


if __name__ == "__main__":
    main()