"""特征工程模块"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


def build_user_features(df: pd.DataFrame, cache_dir: str = "./data/cache") -> tuple[pd.DataFrame, dict]:
    """构建用户特征

    Args:
        df: 原始数据
        cache_dir: 缓存目录

    Returns:
        用户特征DataFrame
    """
    cache_path = os.path.join(cache_dir, "user_features.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 读取用户信息
    user_info = pd.read_csv("./data/format1/data_format1/user_info_format1.csv")

    # 展开日志
    df_explode = df.copy()
    df_explode["log_list"] = df_explode["activity_log"].str.split("#")
    df_explode = df_explode.explode("log_list")
    df_explode[["item_id", "cate_id", "brand_id", "time_str", "action_type"]] = df_explode["log_list"].str.split(
        ":", expand=True
    )

    # 用户基础统计特征
    user_stats = (
        df_explode.groupby("user_id")
        .agg(
            {
                "merchant_id": ["nunique"],  # 交互过的商户数
                "item_id": ["nunique"],  # 交互过的商品数
                "action_type": ["count"],  # 总行为数
                "cate_id": ["nunique"],  # 交互过的类目数
                "brand_id": ["nunique"],  # 交互过的品牌数
            }
        )
        .reset_index()
    )
    user_stats.columns = ["user_id", "merchant_cnt", "item_cnt", "action_cnt", "cate_cnt", "brand_cnt"]

    # 用户行为类型分布
    action_dist = (
        pd.crosstab(
            df_explode["user_id"],
            df_explode["action_type"],
            normalize="index",
        )
        .add_prefix("user_action_")
        .reset_index()
    )

    # 合并人口统计学特征
    user_info = pd.read_csv("./data/format1/data_format1/user_info_format1.csv")

    # 使用LabelEncoder编码age_range和gender
    age_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()

    # 处理缺失值并编码
    user_info["age_range"] = user_info["age_range"].astype(str).fillna("unknown")
    user_info["gender"] = user_info["gender"].astype(str).fillna("unknown")

    user_info["age_range_enc"] = age_encoder.fit_transform(user_info["age_range"])
    user_info["gender_enc"] = gender_encoder.fit_transform(user_info["gender"])

    # 分开处理数值特征和类别特征
    # 1. 数值特征部分
    # 1.1 基础统计特征
    num_stats_cols = ["merchant_cnt", "item_cnt", "action_cnt", "cate_cnt", "brand_cnt"]
    # 1.2 行为分布特征
    action_cols = action_dist.columns.tolist()[1:]  # 除了user_id外的所有行为分布列
    # 2. 类别特征
    cat_cols = ["age_range_enc", "gender_enc"]

    # 合并所有特征
    user_features = user_stats.merge(action_dist, on="user_id", how="left").merge(
        user_info[["user_id"] + cat_cols], on="user_id", how="left"
    )
    user_features = user_features.fillna(0)

    # 标准化数值特征
    scaler = StandardScaler()
    user_features[num_stats_cols + action_cols] = scaler.fit_transform(user_features[num_stats_cols + action_cols])

    # 记录特征维度信息
    feature_dims = {
        "num_stats_cols": num_stats_cols,  # 基础统计特征
        "action_cols": action_cols,  # 行为分布特征
        "cat_cols": cat_cols,  # 类别特征
        "num_feats": len(num_stats_cols) + len(action_cols),  # 数值特征总数
        "cat_feats": len(cat_cols),  # 类别特征总数
    }

    print("✅ 用户特征构建完成:")
    print(f"  - 统计特征: {num_stats_cols}")
    print(f"  - 行为特征: {action_cols}")
    print(f"  - 类别特征: {cat_cols}")
    print(f"  - 年龄类别: {list(age_encoder.classes_)}")
    print(f"  - 性别类别: {list(gender_encoder.classes_)}")

    # 缓存
    with open(cache_path, "wb") as f:
        pickle.dump((user_features, feature_dims), f)

    return user_features, feature_dims


def build_merchant_features(df: pd.DataFrame, cache_dir: str = "./data/cache") -> tuple[pd.DataFrame, dict]:
    """构建商户特征

    Args:
        df: 原始数据
        cache_dir: 缓存目录

    Returns:
        tuple: (merchant_features, feature_dims)
            - merchant_features: 商户特征DataFrame
            - feature_dims: 特征维度信息字典，包含数值特征和类别特征的列名
    """
    cache_path = os.path.join(cache_dir, "merchant_features.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 展开日志
    df_explode = df.copy()
    df_explode["log_list"] = df_explode["activity_log"].str.split("#")
    df_explode = df_explode.explode("log_list")
    record_columns = ["item_id", "cate_id", "brand_id", "time_str", "action_type"]
    df_explode[record_columns] = df_explode["log_list"].str.split(":", expand=True)

    # 验证数据
    print("✓ 数据验证:")
    print(f"  - 总行数: {len(df_explode)}")
    print("  - 年龄段分布:\n", df_explode["age_range"].value_counts(normalize=True))

    # 商户基础统计特征
    merchant_stats = (
        df_explode.groupby("merchant_id")
        .agg(
            {
                "user_id": ["nunique"],  # 交互用户数
                "item_id": ["nunique"],  # 商品数
                "action_type": ["count"],  # 总行为数
                "cate_id": ["nunique"],  # 类目数
                "brand_id": ["nunique"],  # 品牌数
            }
        )
        .reset_index()
    )
    merchant_stats.columns = ["merchant_id", "user_cnt", "item_cnt", "action_cnt", "cate_cnt", "brand_cnt"]

    # 商户行为类型分布
    action_dist = (
        pd.crosstab(
            df_explode["merchant_id"],
            df_explode["action_type"],
            normalize="index",
        )
        .add_prefix("merchant_action_")
        .reset_index()
    )

    print("  - 性别分布:\n", df_explode["gender"].value_counts(normalize=True))

    # 计算商户的用户人口统计特征
    print("\n计算商户的用户人口统计特征...")


    # print(df_with_demo.head(10))
    # 计算每个商户的用户人口统计特征的均值和标准差
    demo_stats = (
        df_explode.groupby("merchant_id")
        .agg(
            {
                "age_range": ["mean", "std"],  # 平均年龄段和标准差
                "gender": ["mean", "std"],  # 平均性别和标准差
            }
        )
        .reset_index()
    )
    demo_stats.columns = ["merchant_id", "age_mean", "age_std", "gender_mean", "gender_std"]

    # 合并所有特征
    merchant_features = merchant_stats.merge(action_dist, on="merchant_id", how="left").merge(
        demo_stats, on="merchant_id", how="left"
    )
    merchant_features = merchant_features.fillna(0)

    # 分类整理特征
    # 1. 数值特征部分
    # 1.1 基础统计特征
    num_stats_cols = ["user_cnt", "item_cnt", "action_cnt", "cate_cnt", "brand_cnt"]
    # 1.2 行为分布特征
    action_cols = action_dist.columns.tolist()[1:]  # 除了merchant_id外的所有行为分布列
    # 1.3 人口统计学特征
    demo_stats_cols = ["age_mean", "age_std", "gender_mean", "gender_std"]

    # 所有数值特征列表
    num_cols = num_stats_cols + action_cols + demo_stats_cols

    # 标准化数值特征
    scaler = StandardScaler()
    merchant_features[num_cols] = scaler.fit_transform(merchant_features[num_cols])

    # 记录特征维度信息
    feature_dims = {
        "num_stats_cols": num_stats_cols,  # 基础统计特征
        "action_cols": action_cols,  # 行为分布特征
        "demo_stats_cols": demo_stats_cols,  # 人口统计学特征
        "num_feats": len(num_cols),  # 数值特征总数
        "cat_feats": 0,  # 商户特征中没有需要embedding的类别特征
    }

    print("✅ 商户特征构建完成:")
    print(f"  - 统计特征: {num_stats_cols}")
    print(f"  - 行为特征: {action_cols}")
    print(f"  - 人口统计: {demo_stats_cols}")

    # 缓存
    with open(cache_path, "wb") as f:
        pickle.dump((merchant_features, feature_dims), f)

    return merchant_features, feature_dims


def build_sequence_features(
    df: pd.DataFrame,
    user_features: pd.DataFrame,
    merchant_features: pd.DataFrame,
    user_feature_dims: dict,
    merchant_feature_dims: dict,
    seq_len: int = 10,
    cache_dir: str = "./data/cache",
) -> tuple[dict, dict]:
    """构建序列特征和编码器

    Args:
        df: 原始数据
        user_features: 用户特征
        merchant_features: 商户特征
        feature_dims: 用户特征维度信息字典，包含数值特征和类别特征的列名
        merchant_feature_dims: 商户特征维度信息字典
        seq_len: 序列长度
        cache_dir: 缓存目录

    Returns:
        序列特征字典和编码器字典
    """

    cache_path = os.path.join(cache_dir, f"sequence_features_len_{seq_len}.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 只保留发生过首次购买的用户-商户对
    df = df[(df["label"].isin([0, 1])) | (df["label"].isnull())].copy()
    print("标签分布:", df["label"].value_counts().to_dict())

    # 展开日志
    df_explode = df.copy()
    df_explode["log_list"] = df_explode["activity_log"].str.split("#")
    df_explode = df_explode.explode("log_list")
    record_columns = ["item_id", "cate_id", "brand_id", "time_str", "action_type"]
    df_explode[record_columns] = df_explode["log_list"].str.split(":", expand=True)

    # 编码类别特征
    encoders = {}
    for col in ["item_id", "cate_id", "brand_id", "merchant_id", "action_type"]:
        encoders[col] = LabelEncoder()
        df_explode[col] = df_explode[col].fillna("UNK")
        encoders[col].fit(df_explode[col].astype(str))
        df_explode[f"{col}_enc"] = encoders[col].transform(df_explode[col].astype(str))

    # 聚合序列特征
    sequence_features = {
        "seqs": [],
        "time_gaps": [],
        "labels": [],
        "user_num_feats": [],
        "user_cat_feats": [],
        "merchant_feats": [],
        "user_ids": [],
        "merchant_ids": [],
    }

    grouped = df_explode.groupby(["user_id", "merchant_id"], sort=False)
    for (user_id, merchant_id), group in tqdm(grouped, total=len(grouped), desc="构建序列特征"):
        # 行为序列
        seq = group[["item_id_enc", "cate_id_enc", "brand_id_enc", "merchant_id_enc", "action_type_enc"]].values

        # 时间间隔
        # 将时间字符串转换为时间戳
        time_strs = group["time_str"].fillna("0101").values
        times = []
        for t in time_strs:
            try:
                dt = pd.to_datetime("2024" + t, format="%Y%m%d")
                times.append(dt.timestamp())
            except (ValueError, TypeError):
                times.append(None)

        # 过滤掉无效的时间戳并转换为numpy数组
        valid_times = [t for t in times if t is not None]
        if len(valid_times) > 1:
            # 计算时间差（以天为单位）
            time_gap = np.diff(np.array(valid_times, dtype=np.float64)) / (24 * 60 * 60)
        else:
            time_gap = np.zeros(1, dtype=np.float64)
        time_gap = np.pad(time_gap, (0, max(0, seq_len - len(time_gap))), "constant")

        # 补齐/截断序列
        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), seq.shape[1]))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[-seq_len:]
            time_gap = time_gap[-seq_len:]

        # 添加用户和商户特征
        user_data = user_features[user_features["user_id"] == user_id]
        merchant_data = merchant_features[merchant_features["merchant_id"] == merchant_id]

        # 如果找不到用户或商户特征，跳过该样本
        if len(user_data) == 0 or len(merchant_data) == 0:
            continue

        # 1. 处理用户特征
        # 1.1 获取用户数值特征（统计特征 + 行为特征）
        user_num_cols = user_feature_dims["num_stats_cols"] + user_feature_dims["action_cols"]
        user_num_feat = user_data[user_num_cols].iloc[0].values

        # 1.2 获取用户类别特征（需要embedding的特征）
        user_cat_cols = user_feature_dims["cat_cols"]
        user_cat_feat = user_data[user_cat_cols].iloc[0].values

        # 2. 处理商户特征
        # 2.1 获取商户数值特征（统计特征 + 行为特征 + 人口统计特征）
        merchant_num_cols = (
            merchant_feature_dims["num_stats_cols"]
            + merchant_feature_dims["action_cols"]
            + merchant_feature_dims["demo_stats_cols"]
        )
        merchant_feat = merchant_data[merchant_num_cols].iloc[0].values

        # 只有在特征完整的情况下才添加样本
        sequence_features["seqs"].append(seq)
        sequence_features["time_gaps"].append(time_gap)
        sequence_features["labels"].append(group["label"].iloc[0])
        sequence_features["user_num_feats"].append(user_num_feat)
        sequence_features["user_cat_feats"].append(user_cat_feat)
        sequence_features["merchant_feats"].append(merchant_feat)
        sequence_features["user_ids"].append(user_id)
        sequence_features["merchant_ids"].append(merchant_id)

    print("✅ 序列特征构建完成:")
    print(f"  - 样本数: {len(sequence_features['seqs'])}")
    print(f"  - 序列长度: {seq_len}")

    print(f"  - 训练集样本数: {len([label for label in sequence_features['labels'] if label in [0, 1]])}")
    print(f"  - 测试集样本数: {len([label for label in sequence_features['labels'] if pd.isna(label)])}")

    # 缓存
    with open(cache_path, "wb") as f:
        pickle.dump((sequence_features, encoders), f)

    return sequence_features, encoders
