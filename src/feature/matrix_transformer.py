import gc
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder


class MatrixTransformer(BaseEstimator, TransformerMixin):
    """基于矩阵分解的embedding特征转换器"""

    def __init__(
        self,
        n_components: int = 50,
        sample_users: int = 100000,
        sample_merchants: int = 5000,
        sample_categories: int = 1000,
        sample_brands: int = 3000,
        min_interactions: int = 2,
        target_date: str = "1111",  # 目标购买行为日期
        enable_cache: bool = True,
        cache_path: str = "../output/matrix_features.pkl",
        random_state: int = 42,
    ):
        """
        Args:
            n_components: embedding维度
            sample_users: 采样用户数量
            sample_merchants/categories/brands: 采样实体数量
            min_interactions: 最小交互次数
            target_date: 目标购买行为日期（用于cat/brand向量关联）
            enable_cache: 是否启用缓存
            cache_path: 缓存路径
            random_state: 随机种子
        """
        self.n_components = n_components
        self.sample_users = sample_users
        self.sample_merchants = sample_merchants
        self.sample_categories = sample_categories
        self.sample_brands = sample_brands
        self.min_interactions = min_interactions
        self.target_date = target_date
        self.enable_cache = enable_cache
        self.cache_path = cache_path
        self.random_state = random_state

        self.cache_prepare_data = True
        self.cache_prepare_data_path = "./output/matrix_prepare_data.pkl"
        # NMF模型
        self.nmf_models = {}

        # 标签编码器
        self.label_encoders = {}

        # 采样后的实体集合
        self.sampled_entities = {}

        # Embedding结果
        self.merchant_embeddings = None  # 商户embedding
        self.category_embeddings = None  # 类别embedding
        self.brand_embeddings = None  # 品牌embedding

        # 样本级别的特征（用于transform）
        self.sample_features = None

        self.is_fitted = False

    def fit(self, X, y=None):
        """训练矩阵分解模型"""

        # 检查缓存
        if self.enable_cache and os.path.exists(self.cache_path):
            print(f"♻️ 从缓存加载矩阵分解特征: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                cached_data = pickle.load(f)

            self._load_from_cache(cached_data)
            print("✅ 矩阵分解特征加载完成")
            self.is_fitted = True
            return self

        print("🔄 开始训练矩阵分解特征...")

        # 1. 数据预处理
        df = self._prepare_interaction_data(X)

        # 2. 采样策略
        sampled_data = self._apply_sampling_strategy(df, y)

        # 3. 构建交互矩阵并训练NMF
        self._build_and_train_nmf(sampled_data)

        # 4. 计算样本级别的特征
        self._compute_sample_features(X, y)

        # 5. 缓存结果
        self._cache_results()

        self.is_fitted = True
        print("✅ 矩阵分解特征训练完成")
        return self

    def transform(self, X):
        """转换数据为矩阵分解特征"""
        if not self.is_fitted:
            raise ValueError("MatrixTransformer has not been fitted yet.")

        print("🔄 开始矩阵分解特征转换...")

        result = X.copy()

        # 关联商户embedding（直接关联）
        if self.merchant_embeddings is not None:
            result = result.merge(self.merchant_embeddings, on="merchant_id", how="left")
            print(f"  ✅ 关联商户embedding: {self.n_components}维")

        # 关联样本级别的cat/brand特征
        if self.sample_features is not None:
            result = result.merge(self.sample_features, on=["user_id", "merchant_id"], how="left")
            cat_dims = len([col for col in self.sample_features.columns if col.startswith("cat_emb")])
            brand_dims = len([col for col in self.sample_features.columns if col.startswith("brand_emb")])
            print(f"  ✅ 关联类别embedding: {cat_dims}维")
            print(f"  ✅ 关联品牌embedding: {brand_dims}维")

        print(f"✅ 矩阵分解特征转换完成，新增维度: {result.shape[1] - X.shape[1]}")
        return result

    def _prepare_interaction_data(self, X):
        """准备交互数据"""
        if self.cache_prepare_data and os.path.exists(self.cache_prepare_data_path):
            print(f"♻️ 从缓存加载交互数据: {self.cache_prepare_data_path}")
            with open(self.cache_prepare_data_path, "rb") as f:
                df = pickle.load(f)
            print("✅ 交互数据加载完成")
            return df

        print("📊 准备交互数据...")

        df = X.copy()
        df = df[df["activity_log"].notnull()]

        # 展开活动日志
        df = df.assign(activity_log=df["activity_log"].str.split("#")).explode("activity_log")
        split_columns = ["item_id", "cate_id", "brand_id", "time", "action_type"]
        df[split_columns] = df["activity_log"].str.split(":", expand=True)

        # 过滤有效数据
        df = df.dropna(subset=["cate_id", "brand_id", "action_type"])

        print(f"✅ 交互数据准备完成: {len(df)} 条记录")
        if self.cache_prepare_data:
            with open(self.cache_prepare_data_path, "wb") as f:
                pickle.dump(df, f)
            print(f"💾 交互数据已缓存: {self.cache_prepare_data_path}")
        return df

    def _apply_sampling_strategy(self, df, y):
        """应用采样策略"""
        print("🎲 应用采样策略...")

        # # 确定训练数据范围
        # if y is not None:
        #     train_mask = y != -1
        #     train_user_merchant = df[df.index.isin(df.index[train_mask])][["user_id", "merchant_id"]].drop_duplicates()
        # else:
        #     train_user_merchant = df[["user_id", "merchant_id"]].drop_duplicates()

        # 统计交互频次
        user_interactions = df.groupby("user_id").size()
        merchant_interactions = df.groupby("merchant_id").size()
        category_interactions = df.groupby("cate_id").size()
        brand_interactions = df.groupby("brand_id").size()

        # 采样活跃用户
        active_users = user_interactions[user_interactions >= self.min_interactions]
        if len(active_users) > self.sample_users:
            # 按活跃度加权采样
            user_probs = active_users / active_users.sum()
            np.random.seed(self.random_state)
            sampled_users = np.random.choice(active_users.index, size=self.sample_users, replace=False, p=user_probs)
        else:
            sampled_users = active_users.index

        # 采样热门实体
        popular_merchants = merchant_interactions.nlargest(self.sample_merchants).index
        popular_categories = category_interactions.nlargest(self.sample_categories).index
        popular_brands = brand_interactions.nlargest(self.sample_brands).index

        # 过滤采样后的数据
        sampled_df = df[
            (df["user_id"].isin(sampled_users))
            & (df["merchant_id"].isin(popular_merchants))
            & (df["cate_id"].isin(popular_categories))
            & (df["brand_id"].isin(popular_brands))
        ].copy()

        # 存储采样结果
        self.sampled_entities = {
            "users": set(sampled_users),
            "merchants": set(popular_merchants),
            "categories": set(popular_categories),
            "brands": set(popular_brands),
        }

        print(f"  ✅ 采样结果: {len(sampled_users)} 用户, {len(popular_merchants)} 商户")
        print(f"           {len(popular_categories)} 类别, {len(popular_brands)} 品牌")
        print(f"  ✅ 采样后数据: {len(sampled_df)} 条交互记录")

        return sampled_df

    def _build_and_train_nmf(self, sampled_df):
        """构建交互矩阵并训练NMF模型"""
        print("🤖 构建交互矩阵并训练NMF模型...")

        # 为每种实体类型构建用户-实体交互矩阵
        entity_types = [
            ("merchant", "merchant_id"),
            ("category", "cate_id"),
            ("brand", "brand_id"),
        ]

        for entity_name, entity_column in entity_types:
            print(f"  🔄 训练{entity_name} NMF模型...")

            # 统计交互次数
            interaction_counts = sampled_df.groupby(["user_id", entity_column]).size().reset_index(name="count")

            # 创建标签编码器
            user_encoder = LabelEncoder()
            entity_encoder = LabelEncoder()

            # 编码用户和实体ID
            interaction_counts["user_idx"] = user_encoder.fit_transform(interaction_counts["user_id"])
            interaction_counts["entity_idx"] = entity_encoder.fit_transform(interaction_counts[entity_column])

            # 存储编码器
            self.label_encoders[entity_name] = {
                "user_encoder": user_encoder,
                "entity_encoder": entity_encoder,
            }

            # 构建稀疏交互矩阵
            n_users = len(user_encoder.classes_)
            n_entities = len(entity_encoder.classes_)

            interaction_matrix = coo_matrix(
                (
                    interaction_counts["count"],
                    (interaction_counts["user_idx"], interaction_counts["entity_idx"]),
                ),
                shape=(n_users, n_entities),
            ).tocsr()

            print(f"    📊 {entity_name}交互矩阵形状: {interaction_matrix.shape}")
            print(f"    📊 稀疏度: {1 - interaction_matrix.nnz / (n_users * n_entities):.4f}")

            # 训练NMF模型
            nmf_model = NMF(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=200,
                alpha_W=0.01,  # 新版本参数
                alpha_H=0.01,  # 新版本参数
                init="random",
            )

            # 矩阵分解：W (用户embedding) × H (实体embedding)
            user_embeddings = nmf_model.fit_transform(interaction_matrix)
            entity_embeddings = nmf_model.components_.T

            # 存储模型
            self.nmf_models[entity_name] = nmf_model

            # 创建实体embedding DataFrame
            if entity_name == "merchant":
                # 商户embedding直接存储
                entity_emb_columns = [f"merchant_emb_{i}" for i in range(self.n_components)]
                self.merchant_embeddings = pd.DataFrame(entity_embeddings, columns=entity_emb_columns)
                self.merchant_embeddings["merchant_id"] = entity_encoder.classes_

            elif entity_name == "category":
                # 类别embedding存储，后续用于样本关联
                entity_emb_columns = [f"cat_emb_{i}" for i in range(self.n_components)]
                self.category_embeddings = pd.DataFrame(entity_embeddings, columns=entity_emb_columns)
                self.category_embeddings["cate_id"] = entity_encoder.classes_

            elif entity_name == "brand":
                # 品牌embedding存储，后续用于样本关联
                entity_emb_columns = [f"brand_emb_{i}" for i in range(self.n_components)]
                self.brand_embeddings = pd.DataFrame(entity_embeddings, columns=entity_emb_columns)
                self.brand_embeddings["brand_id"] = entity_encoder.classes_

            print(f"    ✅ {entity_name} NMF模型训练完成")

            # 清理内存
            del interaction_matrix, user_embeddings, entity_embeddings
            gc.collect()

    def _compute_sample_features(self, X, y):
        """计算样本级别的cat/brand特征"""
        print("🔄 计算样本级别的cat/brand特征...")

        # 确定需要处理的样本
        if y is not None:
            valid_mask = y != -1
            valid_samples = X[valid_mask][["user_id", "merchant_id"]].copy()
        else:
            valid_samples = X[["user_id", "merchant_id"]].copy()

        valid_samples = valid_samples.drop_duplicates()

        # 准备目标日期的购买行为数据
        df = self._prepare_interaction_data(X)

        # 过滤目标日期的购买行为（action_type=1表示购买）
        target_purchases = df[(df["time"].str.contains(self.target_date, na=False)) & (df["action_type"] == "1")].copy()

        print(f"  📊 目标日期({self.target_date})购买行为: {len(target_purchases)} 条")

        sample_features_list = []

        for _, row in valid_samples.iterrows():
            user_id = row["user_id"]
            merchant_id = row["merchant_id"]

            # 获取该用户在目标商户在目标日期的购买行为
            user_merchant_purchases = target_purchases[
                (target_purchases["user_id"] == user_id) & (target_purchases["merchant_id"] == merchant_id)
            ]

            # 初始化特征
            sample_feature = {
                "user_id": user_id,
                "merchant_id": merchant_id,
            }

            # Cat embedding均值
            if len(user_merchant_purchases) > 0 and self.category_embeddings is not None:
                purchase_cats = user_merchant_purchases["cate_id"].unique()
                cat_embeddings = self.category_embeddings[self.category_embeddings["cate_id"].isin(purchase_cats)]

                if len(cat_embeddings) > 0:
                    # 计算均值
                    cat_emb_cols = [col for col in cat_embeddings.columns if col.startswith("cat_emb")]
                    cat_mean = cat_embeddings[cat_emb_cols].mean()
                    sample_feature.update(cat_mean.to_dict())
                else:
                    # 填充零向量
                    for i in range(self.n_components):
                        sample_feature[f"cat_emb_{i}"] = 0.0
            else:
                # 填充零向量
                for i in range(self.n_components):
                    sample_feature[f"cat_emb_{i}"] = 0.0

            # Brand embedding均值
            if len(user_merchant_purchases) > 0 and self.brand_embeddings is not None:
                purchase_brands = user_merchant_purchases["brand_id"].unique()
                brand_embeddings = self.brand_embeddings[self.brand_embeddings["brand_id"].isin(purchase_brands)]

                if len(brand_embeddings) > 0:
                    # 计算均值
                    brand_emb_cols = [col for col in brand_embeddings.columns if col.startswith("brand_emb")]
                    brand_mean = brand_embeddings[brand_emb_cols].mean()
                    sample_feature.update(brand_mean.to_dict())
                else:
                    # 填充零向量
                    for i in range(self.n_components):
                        sample_feature[f"brand_emb_{i}"] = 0.0
            else:
                # 填充零向量
                for i in range(self.n_components):
                    sample_feature[f"brand_emb_{i}"] = 0.0

            sample_features_list.append(sample_feature)

        # 转换为DataFrame
        self.sample_features = pd.DataFrame(sample_features_list)

        print(f"  ✅ 样本特征计算完成: {len(self.sample_features)} 个样本")

    def _load_from_cache(self, cached_data):
        """从缓存加载数据"""
        self.merchant_embeddings = cached_data.get("merchant_embeddings", None)
        self.category_embeddings = cached_data.get("category_embeddings", None)
        self.brand_embeddings = cached_data.get("brand_embeddings", None)
        self.sample_features = cached_data.get("sample_features", None)
        self.sampled_entities = cached_data.get("sampled_entities", {})
        self.label_encoders = cached_data.get("label_encoders", {})

    def _cache_results(self):
        """缓存结果"""
        if self.enable_cache:
            print("💾 缓存矩阵分解特征...")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

            cached_data = {
                "merchant_embeddings": self.merchant_embeddings,
                "category_embeddings": self.category_embeddings,
                "brand_embeddings": self.brand_embeddings,
                "sample_features": self.sample_features,
                "sampled_entities": self.sampled_entities,
                "label_encoders": self.label_encoders,
            }

            with open(self.cache_path, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"✅ 特征已缓存到: {self.cache_path}")

    def get_embedding_stats(self):
        """获取embedding统计信息"""
        if not self.is_fitted:
            return None

        stats = {
            "merchant_embeddings": self.merchant_embeddings.shape if self.merchant_embeddings is not None else None,
            "category_embeddings": self.category_embeddings.shape if self.category_embeddings is not None else None,
            "brand_embeddings": self.brand_embeddings.shape if self.brand_embeddings is not None else None,
            "sample_features": self.sample_features.shape if self.sample_features is not None else None,
            "sampled_entities": {k: len(v) for k, v in self.sampled_entities.items()},
        }

        return stats

    def get_feature_names(self):
        """获取所有特征名称"""
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet.")

        feature_names = []

        # 商户embedding特征名
        if self.merchant_embeddings is not None:
            feature_names.extend([col for col in self.merchant_embeddings.columns if col != "merchant_id"])

        # 样本级别的cat/brand特征名
        if self.sample_features is not None:
            feature_names.extend([col for col in self.sample_features.columns if col not in ["user_id", "merchant_id"]])

        return feature_names


if __name__ == "__main__":
    # 创建测试数据
    import numpy as np

    np.random.seed(42)
    n_samples = 1000

    test_data = {
        "user_id": np.random.randint(1, 100, n_samples),
        "merchant_id": np.random.randint(1, 50, n_samples),
        "activity_log": [
            f"item_{np.random.randint(1, 500)}:cat_{np.random.randint(1, 20)}:brand_{np.random.randint(1, 30)}:1111:1#"
            + f"item_{np.random.randint(1, 500)}:cat_{np.random.randint(1, 20)}:brand_{np.random.randint(1, 30)}:1112:2"
            for _ in range(n_samples)
        ],
    }

    df = pd.DataFrame(test_data)
    X = df
    y = np.random.randint(-1, 2, n_samples)  # -1表示测试集，0/1表示训练集标签

    print("🧪 使用模拟数据进行测试...")
    print(f"模拟数据形状: {X.shape}")

    transformer = MatrixTransformer(
        n_components=20,
        sample_users=50,
        sample_merchants=30,
        sample_categories=15,
        sample_brands=20,
        target_date="1111",
        enable_cache=False,
    )
    transformer.fit(X, y)

    transformed_df = transformer.transform(X)
    print(f"转换后的数据形状: {transformed_df.shape}")
    print(f"特征统计: {transformer.get_embedding_stats()}")
    print(f"特征名称: {transformer.get_feature_names()}")
    print(f"转换后的数据预览:\n{transformed_df.head()}")
