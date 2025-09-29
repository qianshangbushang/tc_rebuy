"""数据集和数据加载器"""

import os
import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from feature_engineering import build_merchant_features, build_sequence_features, build_user_features


class RebuyDataset(Dataset):
    """二次购买预测数据集

    直接使用feature_engineering模块处理好的特征,主要包括:
    1. 用户特征:
        - 数值特征(统计特征 + 行为特征)
        - 类别特征(age_range和gender)
    2. 商户特征:
        - 数值特征(统计特征 + 行为特征 + 人口统计)
    3. 序列特征:
        - 行为序列(item_id, cate_id等)
        - 时间间隔序列
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 30,
        enable_cache: bool = True,
        cache_dir: str = "./data/cache",
    ):
        """初始化数据集

        Args:
            df: 原始数据
            seq_len: 序列长度
            enable_cache: 是否使用缓存
            cache_dir: 缓存目录
        """
        cache_path = os.path.join(cache_dir, f"rebuy_dataset_{seq_len}_size_{len(df)}.pkl")
        os.makedirs(cache_dir, exist_ok=True)

        if enable_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            for k, v in data.items():
                setattr(self, k, v)
            print(f"✅ 从缓存加载数据: {cache_path}")
            return

        # 1. 构建用户和商户特征
        user_features, user_feature_dims = build_user_features(df, cache_dir=cache_dir)
        merchant_features, merchant_feature_dims = build_merchant_features(df, cache_dir=cache_dir)

        # 2. 构建序列特征
        sequence_features, encoders = build_sequence_features(
            df=df,
            user_features=user_features,
            merchant_features=merchant_features,
            user_feature_dims=user_feature_dims,
            merchant_feature_dims=merchant_feature_dims,
            seq_len=seq_len,
            cache_dir=cache_dir,
        )

        # 3. 保存特征和维度信息
        self.seq_len = seq_len
        self.seqs = sequence_features["seqs"]  # [batch_size, seq_len, num_features]
        self.time_gaps = sequence_features["time_gaps"]  # [batch_size, seq_len]
        self.labels = sequence_features["labels"]  # [batch_size]
        self.user_num_feats = sequence_features["user_num_feats"]  # [batch_size, num_feat_dim]
        self.user_cat_feats = sequence_features["user_cat_feats"]  # [batch_size, cat_feat_dim]
        self.merchant_feats = sequence_features["merchant_feats"]  # [batch_size, merchant_feat_dim]
        self.user_ids = sequence_features["user_ids"]
        self.merchant_ids = sequence_features["merchant_ids"]
        self.encoders = encoders
        self.user_feature_dims = user_feature_dims
        self.merchant_feature_dims = merchant_feature_dims

        # 4. 缓存数据
        if enable_cache:
            cache_data = {
                "seq_len": self.seq_len,
                "seqs": self.seqs,
                "time_gaps": self.time_gaps,
                "labels": self.labels,
                "user_num_feats": self.user_num_feats,
                "user_cat_feats": self.user_cat_feats,
                "merchant_feats": self.merchant_feats,
                "user_ids": self.user_ids,
                "merchant_ids": self.merchant_ids,
                "encoders": self.encoders,
                "user_feature_dims": self.user_feature_dims,
                "merchant_feature_dims": self.merchant_feature_dims,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"✅ 数据已缓存: {cache_path}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.seqs)

    def __getitem__(self, idx):
        """返回一个样本

        Returns:
            tuple:
                - seqs: 序列特征 [seq_len, num_features]
                - time_gaps: 时间间隔 [seq_len]
                - user_num_feats: 用户数值特征 [num_feat_dim]
                - user_cat_feats: 用户类别特征 [cat_feat_dim]
                - merchant_feats: 商户特征 [merchant_feat_dim]
                - labels: 标签 [1]
        """
        return (
            torch.tensor(self.seqs[idx], dtype=torch.long),
            torch.tensor(self.time_gaps[idx], dtype=torch.float32),
            torch.tensor(self.user_num_feats[idx], dtype=torch.float32),
            torch.tensor(self.user_cat_feats[idx], dtype=torch.long),
            torch.tensor(self.merchant_feats[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

    @property
    def total_items(self) -> int:
        """所有类别特征的总维度"""
        return sum(len(enc.classes_) for enc in self.encoders.values())

    @property
    def user_num_feat_dim(self) -> int:
        """用户数值特征维度"""
        return len(self.user_num_feats[0])

    @property
    def user_cat_feat_dim(self) -> int:
        """用户类别特征维度"""
        return len(self.user_cat_feats[0])

    @property
    def merchant_feat_dim(self) -> int:
        """商户特征维度"""
        return len(self.merchant_feats[0])


def build_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_info_df: pd.DataFrame,  # 添加用户信息数据
    total_items: int,
    user_feat_dim: int,
    seq_len: int = 30,
    batch_size: int = 128,
    val_ratio: float = 0.2,
    enable_cache: bool = True,
    cache_dir: str = "./data/cache",
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int, int]:  # 添加merchant_feat_dim
    """构建训练、验证、测试数据加载器，同时返回特征维度信息

    Args:
        train_df: 训练数据DataFrame
        test_df: 测试数据DataFrame
        user_info_df: 用户信息数据DataFrame
        total_items: 所有类别特征的总维度
        user_feat_dim: 用户特征维度
        seq_len: 序列长度
        batch_size: 批次大小
        val_ratio: 验证集比例
        enable_cache: 是否启用缓存
        cache_dir: 缓存目录
        pin_memory: 是否将数据固定在内存中(GPU训练时建议启用)

    Returns:
        tuple:
            - train_loader: 训练集数据加载器
            - val_loader: 验证集数据加载器
            - test_loader: 测试集数据加载器
            - total_items: 所有类别特征的总维度
            - user_feat_dim: 用户特征维度
            - merchant_feat_dim: 商户特征维度
    """

    # 训练集
    train_size = int(len(train_df) * (1 - val_ratio))
    train_df = train_df.sample(frac=1, random_state=42)  # 随机打乱

    train_dataset = RebuyDataset(
        train_df.iloc[:train_size],
        user_info_df,
        seq_len=seq_len,
        enable_cache=enable_cache,
        cache_dir=cache_dir,
    )

    # 验证集
    val_dataset = RebuyDataset(
        train_df.iloc[train_size:],
        user_info_df,
        seq_len=seq_len,
        enable_cache=enable_cache,
        cache_dir=cache_dir,
    )

    # 测试集
    test_dataset = RebuyDataset(
        test_df,
        user_info_df,
        seq_len=seq_len,
        enable_cache=enable_cache,
        cache_dir=cache_dir,
    )

    # 计算商户特征维度
    merchant_feat_dim = len(train_dataset.merchant_feats[0])

    # 构建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, total_items, user_feat_dim, merchant_feat_dim


# class RebuyDataset(Dataset):
#     """二次购买预测数据集"""

#     def __init__(self, sequence_features: dict, subset_indices: list = None):
#         """初始化数据集

#         Args:
#             sequence_features: 序列特征字典
#             subset_indices: 子集索引，用于划分训练/验证/测试集
#         """
#         self.seqs = sequence_features["seqs"]
#         self.time_gaps = sequence_features["time_gaps"]
#         self.user_num_feats = sequence_features["user_num_feats"]
#         self.user_cat_feats = sequence_features["user_cat_feats"]
#         self.merchant_feats = sequence_features["merchant_feats"]
#         self.labels = sequence_features["labels"]
#         self.user_ids = sequence_features["user_ids"]
#         self.merchant_ids = sequence_features["merchant_ids"]

#         if subset_indices is not None:
#             self.seqs = [self.seqs[i] for i in subset_indices]
#             self.time_gaps = [self.time_gaps[i] for i in subset_indices]
#             self.user_num_feats = [self.user_num_feats[i] for i in subset_indices]
#             self.user_cat_feats = [self.user_cat_feats[i] for i in subset_indices]
#             self.merchant_feats = [self.merchant_feats[i] for i in subset_indices]
#             self.labels = [self.labels[i] for i in subset_indices]
#             self.user_ids = [self.user_ids[i] for i in subset_indices]
#             self.merchant_ids = [self.merchant_ids[i] for i in subset_indices]

#     def __len__(self):
#         return len(self.seqs)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.seqs[idx], dtype=torch.long),
#             torch.tensor(self.time_gaps[idx], dtype=torch.float32),
#             torch.tensor(self.user_num_feats[idx], dtype=torch.float32),
#             torch.tensor(self.user_cat_feats[idx], dtype=torch.long),
#             torch.tensor(self.merchant_feats[idx], dtype=torch.float32),
#             torch.tensor(self.labels[idx], dtype=torch.float32),
#             self.user_ids[idx],
#             self.merchant_ids[idx],
#         )


def split_train_val_test(
    sequence_features: dict,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[dict, dict, dict]:
    """划分训练集、验证集和测试集

    Args:
        df_train: 训练数据
        df_test: 测试数据
        sequence_features: 序列特征字典
        val_ratio: 验证集比例
        random_state: 随机种子

    Returns:
        训练集、验证集和测试集的序列特征字典
    """
    labels = sequence_features["labels"]
    # 训练集中的样本索引
    train_indices = [i for i in range(len(labels)) if labels[i] in [0, 1]]
    # 测试集的样本索引
    test_idx = [i for i in range(len(labels)) if pd.isna(labels[i])]

    # 划分训练集和验证集
    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=val_ratio,
        random_state=random_state,
        stratify=[labels[i] for i in train_indices],
    )

    # 构建子集特征
    def subset_features(features: dict, indices: list) -> dict:
        return {k: [features[k][i] for i in indices] for k in features}

    train_features = subset_features(sequence_features, train_idx)
    val_features = subset_features(sequence_features, val_idx)
    test_features = subset_features(sequence_features, test_idx)

    return train_features, val_features, test_features


# def build_dataloaders(
#     train_features: dict, val_features: dict, test_features: dict, batch_size: int = 128, num_workers: int = 0
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """构建数据加载器

#     Args:
#         train_features: 训练集特征
#         val_features: 验证集特征
#         test_features: 测试集特征
#         batch_size: 批次大小
#         num_workers: 数据加载线程数

#     Returns:
#         训练集、验证集和测试集的数据加载器
#     """
#     train_dataset = RebuyDataset(train_features)
#     val_dataset = RebuyDataset(val_features)
#     test_dataset = RebuyDataset(test_features)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#     )

#     val_loader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#     )

#     return train_loader, val_loader, test_loader
