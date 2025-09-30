"""特征工程模块"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm


def create_global_labelencoder(df_explode: pd.DataFrame, columns: list, cache_dir: str) -> LabelEncoder:
    """为指定的列创建全局LabelEncoder

    Args:
        df_explode: 原始数据
        columns: 需要编码的列名列表
        cache_dir: 缓存目录

    Returns:
        LabelEncoder对象
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "global_label_encoder.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 使用列表而不是集合，因为最终还是要转换为列表
    all_values = []

    # 预分配足够大的列表空间
    # total_size = sum(df_explode[col].nunique() for col in columns)
    all_values = []
    # all_values.reserve(total_size + len(columns))  # 为UNK预留空间

    # 使用更高效的字符串连接方式
    for col in columns:
        # 使用nunique先获取唯一值数量
        unique_count = df_explode[col].nunique()
        print(f"列 {col} 的唯一值数量: {unique_count}")

        # 使用numpy的unique函数，比pandas的unique更快
        col_unique = pd.Series(df_explode[col].dropna().unique())

        # 使用vectorized操作替代列表推导
        col_values = col + ":" + col_unique.astype(str)

        # 直接extend而不是update
        all_values.extend(col_values)
        # 添加UNK值
        all_values.append(f"{col}:UNK")

        # 只打印少量示例值
        print(f"示例值: {col_values[:5]}")

    label_encoder = LabelEncoder()
    # 直接使用列表训练，避免转换
    label_encoder.fit(all_values)

    # 使用更高效的pickle协议
    with open(cache_path, "wb") as f:
        pickle.dump(label_encoder, f, protocol=4)

    return label_encoder


def build_user_conversion_rates(df_explode: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个用户的点击、收藏、购买、复购转化率
    action_type: 0=点击, 1=加购, 2=购买, 3=收藏
    输出: user_id, click_fav_rate, click_buy_rate, fav_buy_rate, rebuy_rate
    """
    df = df_explode[["user_id", "merchant_id", "action_type", "time_str"]].copy()
    df = df.sort_values(["user_id", "merchant_id", "time_str"])

    user_stats = {}

    for user_id, group in df.groupby("user_id"):
        click_cnt = (group["action_type"] == "0").sum()
        fav_cnt = (group["action_type"] == "3").sum()
        buy_cnt = (group["action_type"] == "2").sum()
        click_fav_cnt = 0
        click_buy_cnt = 0
        fav_buy_cnt = 0
        rebuy_merchant_cnt = 0
        buy_merchant_cnt = 0

        # 按商户分组，统计转化次数和复购
        for _, merchant_group in group.groupby("merchant_id"):
            acts = merchant_group["action_type"].values
            # 点击后有收藏
            if "0" in acts and "3" in acts:
                click_fav_cnt += 1
            # 点击后有购买
            if "0" in acts and "2" in acts:
                click_buy_cnt += 1
            # 收藏后有购买
            if "3" in acts and "2" in acts:
                fav_buy_cnt += 1
            # 复购：购买次数>=2
            buy_times = (acts == "2").sum()
            if buy_times >= 2:
                rebuy_merchant_cnt += 1
            if buy_times >= 1:
                buy_merchant_cnt += 1

        user_stats[user_id] = {
            "click_cnt": click_cnt,
            "fav_cnt": fav_cnt,
            "buy_cnt": buy_cnt,
            "click_fav_cnt": click_fav_cnt,
            "click_buy_cnt": click_buy_cnt,
            "fav_buy_cnt": fav_buy_cnt,
            "rebuy_merchant_cnt": rebuy_merchant_cnt,
            "buy_merchant_cnt": buy_merchant_cnt,
        }

    user_df = pd.DataFrame.from_dict(user_stats, orient="index")
    user_df["click_fav_rate"] = user_df["click_fav_cnt"] / (user_df["click_cnt"] + 1e-6)
    user_df["click_buy_rate"] = user_df["click_buy_cnt"] / (user_df["click_cnt"] + 1e-6)
    user_df["fav_buy_rate"] = user_df["fav_buy_cnt"] / (user_df["fav_cnt"] + 1e-6)
    user_df["rebuy_rate"] = user_df["rebuy_merchant_cnt"] / (user_df["buy_merchant_cnt"] + 1e-6)
    user_df = user_df.reset_index().rename(columns={"index": "user_id"})

    stand_scaler = MinMaxScaler()
    cnt_cols = [
        "click_cnt",
        "fav_cnt",
        "buy_cnt",
        "click_fav_cnt",
        "click_buy_cnt",
        "fav_buy_cnt",
        "rebuy_merchant_cnt",
        "buy_merchant_cnt",
    ]
    user_df[cnt_cols] = stand_scaler.fit_transform(user_df[cnt_cols])
    print("✅ 用户转化率特征构建完成")
    return user_df


def build_high_freq_cate_user_action_dist_v1(
    df_explode: pd.DataFrame,
    cate_col: str = "cate_id",
    user_col: str = "user_id",
    action_col: str = "action_type",
    freq_threshold: int = 100000,
) -> pd.DataFrame:
    """
    统计高频类别（如cate_id），并计算每个用户在这些类别上的行为分布（占比）
    输出形式: user_id, action_{action_type}_{cate_col}_{cate}
    """
    # 1. 统计高频类别
    cate_counts = df_explode[cate_col].value_counts()
    high_freq_cates = cate_counts[cate_counts >= freq_threshold].index.tolist()
    print(f"高频{cate_col}数量: {len(high_freq_cates)}")

    # 2. 过滤相关字段
    df_high_cate = df_explode[df_explode[cate_col].isin(high_freq_cates)][[user_col, cate_col, action_col]]

    # 3. 计算每个用户在这些类别上的行为分布（占比）
    counts = df_high_cate.groupby([user_col, cate_col, action_col]).size().reset_index(name="count")

    # 2. 计算总和
    totals = counts.groupby([user_col, cate_col])["count"].sum().reset_index()

    # 3. 使用merge和向量化操作计算比例
    user_cate_action = counts.merge(totals, on=[user_col, cate_col], suffixes=("", "_total"))
    user_cate_action["ratio"] = user_cate_action["count"] / user_cate_action["count_total"]

    # 4. 删除中间列
    user_cate_action = user_cate_action.drop(["count", "count_total"], axis=1)
    print(user_cate_action.head(10))

    # 4. pivot为宽表
    user_cate_action["col_name"] = user_cate_action.apply(
        lambda x: f"action_{x[action_col]}_{cate_col}_{x[cate_col]}", axis=1
    )
    result = user_cate_action.pivot_table(
        index=user_col, columns="col_name", values="ratio", fill_value=0
    ).reset_index()

    print(f"✅ 高频{cate_col}用户行为分布特征构建完成，输出形状: {result.shape}")
    print(result.head(10))
    return result


def build_high_freq_cate_user_action_dist(
    df_explode: pd.DataFrame,
    cate_col: str = "cate_id",
    user_col: str = "user_id",
    action_col: str = "action_type",
    freq_threshold: int = 100000,
) -> pd.DataFrame:
    """统计高频类别的用户行为分布

    Args:
        df_explode: 展开后的数据框
        cate_col: 类别列名（如cate_id, brand_id等）
        user_col: 用户ID列名
        action_col: 行为类型列名
        freq_threshold: 高频类别的阈值

    Returns:
        DataFrame: 用户在各个类别上的行为分布特征
    """
    # 1. 统计高频类别（使用value_counts的高效实现）
    cate_counts = df_explode[cate_col].value_counts()
    high_freq_cates = cate_counts[cate_counts >= freq_threshold].index
    print(f"高频{cate_col}数量: {len(high_freq_cates)}")

    # 2. 过滤数据（使用isin的向量化操作）
    df_high_cate = df_explode[df_explode[cate_col].isin(high_freq_cates)][[user_col, cate_col, action_col]]

    # 3. 一次性计算所有计数（避免多次groupby）
    counts = df_high_cate.groupby([user_col, cate_col, action_col]).size().reset_index(name="count")

    # 4. 计算分组总和（使用transform的向量化操作）
    totals = counts.groupby([user_col, cate_col])["count"].transform("sum")

    # 5. 计算比例（向量化操作）
    counts["ratio"] = counts["count"] / totals

    # 6. 构建特征名称（向量化操作）
    counts["col_name"] = (
        "action_" + counts[action_col].astype(str) + "_" + cate_col + "_" + counts[cate_col].astype(str)
    )

    # 7. 转换为宽表格式（一次性操作）
    result = counts.pivot_table(index=user_col, columns="col_name", values="ratio", fill_value=0).reset_index()

    print(f"✅ 高频{cate_col}用户行为分布特征构建完成，输出形状: {result.shape}")
    return result


def build_user_stat_features(df_explode: pd.DataFrame) -> pd.DataFrame:
    """构建用户基础统计特征

    Args:
        df_explode: 展开后的日志数据

    Returns:
        用户基础统计特征DataFrame
    """
    basic_stats_cols = ["merchant_cnt", "item_cnt", "action_cnt", "cate_cnt", "brand_cnt"]
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
    user_stats.columns = ["user_id"] + basic_stats_cols
    # 标准化数值特征
    scaler = MinMaxScaler()
    user_stats[basic_stats_cols] = scaler.fit_transform(user_stats[basic_stats_cols])

    action_dist = (
        pd.crosstab(
            df_explode["user_id"],
            df_explode["action_type"],
            normalize="index",
        )
        .add_prefix("user_action_")
        .reset_index()
    )
    user_stats = user_stats.merge(action_dist, on="user_id", how="left").fillna(0)
    return user_stats


def build_user_features(
    df_explode: pd.DataFrame,
    global_le_encoder: LabelEncoder = None,
    high_freq_threshold: int = 300000,
    cache_dir: str = "./data/cache",
) -> tuple[pd.DataFrame, dict]:
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

    # 展开日志
    # df_explode = df.copy()
    # df_explode["log_list"] = df_explode["activity_log"].str.split("#")
    # df_explode = df_explode.explode("log_list")
    # df_explode[["item_id", "cate_id", "brand_id", "time_str", "action_type"]] = df_explode["log_list"].str.split(
    #     ":", expand=True
    # )

    # 用户行为类型分布
    user_action_stats = build_user_stat_features(df_explode)

    # 用户基础属性特征
    user_basic = df_explode[["user_id", "age_range", "gender"]].drop_duplicates()
    user_basic["age_range_enc"] = global_le_encoder.transform(
        user_basic["age_range"].apply(lambda x: f"age_range:{x}" if pd.notna(x) else "age_range:UNK")
    )
    user_basic["gender_enc"] = global_le_encoder.transform(
        user_basic["gender"].apply(lambda x: f"gender:{x}" if pd.notna(x) else "gender:UNK")
    )
    cat_cols = ["age_range_enc", "gender_enc"]
    user_basic = user_basic[["user_id"] + cat_cols]

    # 计算用户在高频类别上的行为分布特征
    user_cate_interaction_dist = build_high_freq_cate_user_action_dist(
        df_explode,
        cate_col="cate_id",
        user_col="user_id",
        action_col="action_type",
        freq_threshold=high_freq_threshold,
    )
    user_brand_interaction_dist = build_high_freq_cate_user_action_dist(
        df_explode,
        cate_col="brand_id",
        user_col="user_id",
        action_col="action_type",
        freq_threshold=high_freq_threshold,
    )
    user_item_interaction_dist = build_high_freq_cate_user_action_dist(
        df_explode,
        cate_col="item_id",
        user_col="user_id",
        action_col="action_type",
        freq_threshold=high_freq_threshold,
    )

    # 计算用户转化率
    user_conv_feat = build_user_conversion_rates(df_explode)

    # 合并所有特征
    user_features = (
        user_action_stats.merge(user_basic, on="user_id", how="left")
        .merge(user_cate_interaction_dist, on="user_id", how="left")
        .merge(user_brand_interaction_dist, on="user_id", how="left")
        .merge(user_item_interaction_dist, on="user_id", how="left")
        .merge(user_conv_feat, on="user_id", how="left")
    )
    user_features = user_features.fillna(0)

    num_stats_cols = [col for col in user_features.columns if col not in ["user_id"] + cat_cols]
    # 记录特征维度信息
    feature_dims = {
        "num_stats_cols": num_stats_cols,  # 数值特征
        "cat_cols": cat_cols,  # 类别特征
        "num_feats": len(num_stats_cols),  # 数值特征总数
        "cat_feats": len(cat_cols),  # 类别特征总数
    }

    print("✅ 用户特征构建完成:")
    print("用户特征示例: \n", user_features.iloc[0].to_frame())

    # 缓存
    with open(cache_path, "wb") as f:
        pickle.dump((user_features, feature_dims), f)

    return user_features, feature_dims


def build_merchant_conversion_features_v2(df_explode: pd.DataFrame) -> pd.DataFrame:
    """
    统计商户的行为级别转化率（基于行为次数，复购率基于购买行为序列）
    """
    df = df_explode[["merchant_id", "user_id", "action_type", "time_str"]].copy()
    df = df.sort_values(["merchant_id", "user_id", "time_str"])

    merchant_stats = {}

    for merchant_id, group in df.groupby("merchant_id"):
        click_cnt = (group["action_type"] == "0").sum()
        cart_cnt = (group["action_type"] == "1").sum()
        buy_cnt = (group["action_type"] == "2").sum()
        fav_cnt = (group["action_type"] == "3").sum()
        click_cart_cnt = 0
        click_fav_cnt = 0
        click_buy_cnt = 0
        fav_buy_cnt = 0
        rebuy_user_cnt = 0
        buy_user_cnt = 0

        # 按用户分组，统计转化次数和复购
        for _, user_group in group.groupby("user_id"):
            acts = user_group["action_type"].values
            # 点击后有加购
            if "0" in acts and "1" in acts:
                click_cart_cnt += 1
            # 点击后有收藏
            if "0" in acts and "3" in acts:
                click_fav_cnt += 1
            # 点击后有购买
            if "0" in acts and "2" in acts:
                click_buy_cnt += 1
            # 收藏后有购买
            if "3" in acts and "2" in acts:
                fav_buy_cnt += 1
            # 复购：购买次数>=2
            buy_times = (acts == "2").sum()
            if buy_times >= 2:
                rebuy_user_cnt += 1
            if buy_times >= 1:
                buy_user_cnt += 1

        merchant_stats[merchant_id] = {
            "click_cnt": click_cnt,
            "cart_cnt": cart_cnt,
            "buy_cnt": buy_cnt,
            "fav_cnt": fav_cnt,
            "click_cart_cnt": click_cart_cnt,
            "click_fav_cnt": click_fav_cnt,
            "click_buy_cnt": click_buy_cnt,
            "fav_buy_cnt": fav_buy_cnt,
            "rebuy_user_cnt": rebuy_user_cnt,
            "buy_user_cnt": buy_user_cnt,
        }

    merchant_df = pd.DataFrame.from_dict(merchant_stats, orient="index")
    merchant_df["click_cart_rate"] = merchant_df["click_cart_cnt"] / (merchant_df["click_cnt"] + 1e-6)
    merchant_df["click_fav_rate"] = merchant_df["click_fav_cnt"] / (merchant_df["click_cnt"] + 1e-6)
    merchant_df["click_buy_rate"] = merchant_df["click_buy_cnt"] / (merchant_df["click_cnt"] + 1e-6)
    merchant_df["fav_buy_rate"] = merchant_df["fav_buy_cnt"] / (merchant_df["fav_cnt"] + 1e-6)
    merchant_df["rebuy_rate"] = merchant_df["rebuy_user_cnt"] / (merchant_df["buy_user_cnt"] + 1e-6)
    merchant_df = merchant_df.reset_index().rename(columns={"index": "merchant_id"})
    stand_scaler = MinMaxScaler()
    cnt_cols = [
        "click_cnt",
        "cart_cnt",
        "buy_cnt",
        "fav_cnt",
        "click_cart_cnt",
        "click_fav_cnt",
        "click_buy_cnt",
        "fav_buy_cnt",
        "rebuy_user_cnt",
        "buy_user_cnt",
    ]
    merchant_df[cnt_cols] = stand_scaler.fit_transform(merchant_df[cnt_cols])
    print("✅ 商户行为级别转化率特征构建完成")
    print(merchant_df.head())
    return merchant_df


def build_merchant_features(
    # df: pd.DataFrame,
    df_explode: pd.DataFrame = None,
    global_le_encoder: LabelEncoder = None,
    cache_dir: str = "./data/cache",
) -> tuple[pd.DataFrame, dict]:
    """构建商户特征

    Args:
        df: 原始数据
        global_le_encoder: 全局LabelEncoder
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

    merchant_conversion_features = build_merchant_conversion_features_v2(df_explode)

    # 合并所有特征
    merchant_features = (
        merchant_stats.merge(action_dist, on="merchant_id", how="left")
        .merge(demo_stats, on="merchant_id", how="left")
        .merge(merchant_conversion_features, on="merchant_id", how="left")
    )
    merchant_features = merchant_features.fillna(0)

    # 分类整理特征
    # 1. 数值特征部分
    # 1.1 基础统计特征
    num_stats_cols = ["user_cnt", "item_cnt", "action_cnt", "cate_cnt", "brand_cnt"]
    # 1.2 行为分布特征
    # action_cols = action_dist.columns.tolist()[1:]  # 除了merchant_id外的所有行为分布列
    # 1.3 人口统计学特征
    # demo_stats_cols = ["age_mean", "age_std", "gender_mean", "gender_std"]

    # 所有数值特征列表
    # num_cols = num_stats_cols + action_cols + demo_stats_cols

    # 标准化数值特征

    scaler = MinMaxScaler()
    merchant_features[num_stats_cols] = scaler.fit_transform(merchant_features[num_stats_cols])

    # 记录特征维度信息
    feature_dims = {
        "num_stats_cols": merchant_features.columns.tolist()[1:],  # 基础统计特征
        # "num_stats_cols": num_cols,  # 基础统计特征
        # "action_cols": action_cols,  # 行为分布特征
        # "demo_stats_cols": demo_stats_cols,  # 人口统计学特征
        "num_feats": len(merchant_features.columns.to_list()) - 1,  # 数值特征总数
        "cat_feats": 0,  # 商户特征中没有需要embedding的类别特征
    }

    print("✅ 商户特征构建完成:")
    print(f"  - 统计特征: {num_stats_cols}")
    # print(f"  - 行为特征: {action_cols}")
    # print(f"  - 人口统计: {demo_stats_cols}")

    # 缓存
    with open(cache_path, "wb") as f:
        pickle.dump((merchant_features, feature_dims), f)
    print("merchant feature example: \n", merchant_features.iloc[0].to_frame())
    return merchant_features, feature_dims


# ...existing code...


def build_sequence_features_v1(
    df_explode: pd.DataFrame,
    user_features: pd.DataFrame = None,
    merchant_features: pd.DataFrame = None,
    user_feature_dims: dict = {},
    merchant_feature_dims: dict = {},
    global_le_encoder: LabelEncoder = None,
    seq_len: int = 10,
    cache_dir: str = "./data/cache",
    action_type_list: list[str] = None,
    scale_action_counts: bool = True,
) -> tuple[dict, dict]:
    """
    构建序列特征：
      - 针对 (user_id, merchant_id) 直接逐组计算总行为次数与各 action_type 次数
      - 统一结束后再做 MinMax 归一化，避免索引错位
    返回:
      sequence_features: dict
      encoders: 空（兼容接口）
    """
    if action_type_list is None:
        action_type_list = ["0", "1", "2", "3"]

    cache_path = os.path.join(cache_dir, f"sequence_features_len_{seq_len}.pkl")
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 编码类别特征（统一全局LabelEncoder）
    for col in ["item_id", "cate_id", "brand_id", "merchant_id", "action_type"]:
        df_explode[f"{col}_enc"] = global_le_encoder.transform(
            df_explode[col].apply(lambda x: f"{col}:{x}" if pd.notna(x) else f"{col}:UNK")
        )

    sequence_features = {
        "seqs": [],
        "time_gaps": [],
        "labels": [],
        "user_num_feats": [],
        "user_cat_feats": [],
        "merchant_feats": [],
        "user_ids": [],
        "merchant_ids": [],
        "pair_action_cnt": [],  # 归一化后的（或原始）总行为次数
        "pair_action_type_cnt": [],  # 归一化后的（或原始）各类型次数列表
    }

    # 暂存原始计数用于之后统一归一化
    raw_total_action_cnt = []
    raw_action_type_cnt = []

    grouped = df_explode.groupby(["user_id", "merchant_id"], sort=False)

    for (user_id, merchant_id), group in tqdm(grouped, total=len(grouped), desc="构建序列特征"):
        # 序列编码矩阵
        seq = group[["item_id_enc", "cate_id_enc", "brand_id_enc", "merchant_id_enc", "action_type_enc"]].values

        # 时间间隔
        time_strs = group["time_str"].fillna("0101").values
        ts_list = []
        for t in time_strs:
            try:
                ts_list.append(pd.to_datetime("2024" + t, format="%Y%m%d").timestamp())
            except Exception:
                ts_list.append(None)
        valid_ts = [t for t in ts_list if t is not None]
        if len(valid_ts) > 1:
            gap = np.diff(np.array(valid_ts)) / (24 * 3600)
        else:
            gap = np.zeros(1, dtype=np.float64)
        gap = np.pad(gap, (0, max(0, seq_len - len(gap))), "constant")

        # 序列裁剪/补齐
        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), seq.shape[1]))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[-seq_len:]
            gap = gap[-seq_len:]

        # 用户 / 商户特征
        user_row = user_features[user_features["user_id"] == user_id]
        merchant_row = merchant_features[merchant_features["merchant_id"] == merchant_id]
        if user_row.empty or merchant_row.empty:
            continue

        user_num_cols = user_feature_dims["num_stats_cols"]
        user_num_feat = user_row[user_num_cols].iloc[0].values
        user_cat_cols = user_feature_dims["cat_cols"]
        user_cat_feat = user_row[user_cat_cols].iloc[0].values

        merchant_num_cols = (
            merchant_feature_dims["num_stats_cols"]
            + merchant_feature_dims.get("action_cols", [])
            + merchant_feature_dims.get("demo_stats_cols", [])
        )
        merchant_feat = merchant_row[merchant_num_cols].iloc[0].values

        # 行为次数（原始）
        total_cnt = len(group)
        cnt_per_type = group["action_type"].value_counts().reindex(action_type_list, fill_value=0).astype(int).tolist()

        raw_total_action_cnt.append(total_cnt)
        raw_action_type_cnt.append(cnt_per_type)

        # 存放其它特征（计数暂不放入，稍后统一归一化后再替换）
        sequence_features["seqs"].append(seq)
        sequence_features["time_gaps"].append(gap)
        sequence_features["labels"].append(group["label"].iloc[0] if "label" in group else 0)
        sequence_features["user_num_feats"].append(user_num_feat)
        sequence_features["user_cat_feats"].append(user_cat_feat)
        sequence_features["merchant_feats"].append(merchant_feat)
        sequence_features["user_ids"].append(user_id)
        sequence_features["merchant_ids"].append(merchant_id)

    # 统一归一化行为计数
    raw_total_action_cnt = np.array(raw_total_action_cnt).reshape(-1, 1)
    raw_action_type_cnt = np.array(raw_action_type_cnt)  # [N, num_action_types]

    if scale_action_counts:
        scaler_total = MinMaxScaler()
        scaler_types = MinMaxScaler()
        total_scaled = scaler_total.fit_transform(raw_total_action_cnt).flatten()
        type_scaled = scaler_types.fit_transform(raw_action_type_cnt)
        sequence_features["pair_action_cnt"] = total_scaled.tolist()
        sequence_features["pair_action_type_cnt"] = type_scaled.tolist()
    else:
        sequence_features["pair_action_cnt"] = raw_total_action_cnt.flatten().tolist()
        sequence_features["pair_action_type_cnt"] = raw_action_type_cnt.tolist()

    print("✅ 序列特征构建完成:")
    print(f"  - 样本数: {len(sequence_features['seqs'])}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 行为计数已{'归一化' if scale_action_counts else '保留原始值'}")

    with open(cache_path, "wb") as f:
        pickle.dump((sequence_features, {}), f)

    for k, v in sequence_features.items():
        print(f"  - {k}: {np.array(v).shape}")
        print(f"    示例: {v[0]}")
    return sequence_features, {}


def build_sequence_features(
    df_explode: pd.DataFrame,
    user_features: pd.DataFrame = None,
    merchant_features: pd.DataFrame = None,
    user_feature_dims: dict = {},
    merchant_feature_dims: dict = {},
    global_le_encoder: LabelEncoder = None,
    seq_len: int = 10,
    cache_dir: str = "./data/cache",
    action_type_list: list[str] = None,
    scale_action_counts: bool = True,
    target_pairs: pd.DataFrame = None,  # 新增参数：目标user-merchant对
) -> tuple[dict, dict]:
    """构建序列特征

    Args:
        ...
        target_pairs: DataFrame，包含需要计算特征的user_id和merchant_id对
                     格式：columns=['user_id', 'merchant_id']
    """
    if action_type_list is None:
        action_type_list = ["0", "1", "2", "3"]

    cache_path = os.path.join(cache_dir, f"sequence_features_len_{seq_len}.pkl")
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 1. 过滤出目标user-merchant对的数据
    if target_pairs is not None:
        print(f"原始数据行数: {len(df_explode)}")
        df_explode = df_explode.merge(target_pairs, on=["user_id", "merchant_id"], how="inner")
        print(f"过滤后数据行数: {len(df_explode)}")
        print(f"目标pairs数量: {len(target_pairs)}")

    # 2. 编码类别特征（统一全局LabelEncoder）
    for col in ["item_id", "cate_id", "brand_id", "merchant_id", "action_type"]:
        df_explode[f"{col}_enc"] = global_le_encoder.transform(
            df_explode[col].apply(lambda x: f"{col}:{x}" if pd.notna(x) else f"{col}:UNK")
        )

    sequence_features = {
        "seqs": [],
        "time_gaps": [],
        "labels": [],
        "user_num_feats": [],
        "user_cat_feats": [],
        "merchant_feats": [],
        "user_ids": [],
        "merchant_ids": [],
        "pair_action_cnt": [],
        "pair_action_type_cnt": [],
    }

    # 3. 按用户-商户对分组计算特征
    raw_total_action_cnt = []
    raw_action_type_cnt = []

    # 优化groupby性能
    df_explode = df_explode.sort_values(["user_id", "merchant_id"])
    grouped = df_explode.groupby(["user_id", "merchant_id"])

    for (user_id, merchant_id), group in tqdm(grouped, desc="构建序列特征"):
        # 序列特征矩阵
        seq = group[["item_id_enc", "cate_id_enc", "brand_id_enc", "merchant_id_enc", "action_type_enc"]].values

        # 时间间隔计算优化
        time_strs = pd.to_datetime("2024" + group["time_str"].fillna("0101"), format="%Y%m%d")
        if len(time_strs) > 1:
            gap = np.diff(time_strs.astype(np.int64)) / (24 * 3600 * 1e9)
        else:
            gap = np.zeros(1, dtype=np.float64)
        gap = np.pad(gap, (0, max(0, seq_len - len(gap))), "constant")

        # 序列裁剪/补齐
        if len(seq) < seq_len:
            seq = np.pad(seq, ((0, seq_len - len(seq)), (0, 0)), "constant")
        else:
            seq = seq[-seq_len:]
            gap = gap[-seq_len:]

        # 用户/商户特征
        user_row = user_features[user_features["user_id"] == user_id]
        merchant_row = merchant_features[merchant_features["merchant_id"] == merchant_id]
        if user_row.empty or merchant_row.empty:
            continue

        # 提取特征
        user_num_feat = user_row[user_feature_dims["num_stats_cols"]].iloc[0].values
        user_cat_feat = user_row[user_feature_dims["cat_cols"]].iloc[0].values

        merchant_num_cols = (
            merchant_feature_dims["num_stats_cols"]
            + merchant_feature_dims.get("action_cols", [])
            + merchant_feature_dims.get("demo_stats_cols", [])
        )
        merchant_feat = merchant_row[merchant_num_cols].iloc[0].values

        # 行为计数
        total_cnt = len(group)
        cnt_per_type = group["action_type"].value_counts().reindex(action_type_list, fill_value=0).values

        # 添加特征
        sequence_features["seqs"].append(seq)
        sequence_features["time_gaps"].append(gap)
        sequence_features["labels"].append(group["label"].iloc[0])
        sequence_features["user_num_feats"].append(user_num_feat)
        sequence_features["user_cat_feats"].append(user_cat_feat)
        sequence_features["merchant_feats"].append(merchant_feat)
        sequence_features["user_ids"].append(user_id)
        sequence_features["merchant_ids"].append(merchant_id)

        raw_total_action_cnt.append(total_cnt)
        raw_action_type_cnt.append(cnt_per_type)

    # 4. 统一归一化行为计数
    raw_total_action_cnt = np.array(raw_total_action_cnt).reshape(-1, 1)
    raw_action_type_cnt = np.array(raw_action_type_cnt)

    if scale_action_counts:
        scaler_total = MinMaxScaler()
        scaler_types = MinMaxScaler()
        sequence_features["pair_action_cnt"] = scaler_total.fit_transform(raw_total_action_cnt).flatten().tolist()
        sequence_features["pair_action_type_cnt"] = scaler_types.fit_transform(raw_action_type_cnt).tolist()
    else:
        sequence_features["pair_action_cnt"] = raw_total_action_cnt.flatten().tolist()
        sequence_features["pair_action_type_cnt"] = raw_action_type_cnt.tolist()

    print("✅ 序列特征构建完成:")
    print(f"  - 样本数: {len(sequence_features['seqs'])}")
    print(f"  - 序列长度: {seq_len}")

    with open(cache_path, "wb") as f:
        pickle.dump((sequence_features, {}), f)

    return sequence_features, {}


# ...existing code...
