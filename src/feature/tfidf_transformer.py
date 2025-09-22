import gc
import os
import pickle

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeatureTransformer(BaseEstimator, TransformerMixin):
    """TF-IDF特征转换器"""

    def __init__(
        self,
        top_n_features=500,
        min_df=2,
        max_df=0.95,
        enable_cache=True,
        cache_path: str = "../output/tfidf_feature.pkl",
    ):
        """
        Args:
            top_n_features: 每个特征类型保留的最大特征数
            min_df: 最小文档频率，过滤低频特征
            max_df: 最大文档频率，过滤高频特征
        """
        self.top_n_features = top_n_features
        self.min_df = min_df
        self.max_df = max_df
        self.enable_cache = enable_cache
        self.cache_path = cache_path

        # 用户特征的TF-IDF向量化器
        self.user_vectorizers = {
            "cat": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
            "brand": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
            "item": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
            "merchant": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
        }

        # 商户特征的TF-IDF向量化器
        self.merchant_vectorizers = {
            "cat": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
            "brand": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
            "item": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
            "user": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
        }

        # 用户-商户交互特征的TF-IDF向量化器
        self.user_merchant_vectorizers = {
            "cat": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
            "brand": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
            "item": TfidfVectorizer(max_features=top_n_features, min_df=min_df, max_df=max_df),
        }

        # 存储预计算的特征矩阵
        self.user_tfidf_features = None
        self.merchant_tfidf_features = None
        self.user_merchant_tfidf_features = None

        self.is_fitted = False

    def fit(self, X, y=None):
        """训练TF-IDF向量化器并预计算所有特征"""

        # 检查缓存
        if self.enable_cache and os.path.exists(self.cache_path):
            print(f"♻️ 从缓存加载TF-IDF特征: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                cached_data = pickle.load(f)

            self.user_tfidf_features = cached_data.get("user_tfidf_features", None)
            self.merchant_tfidf_features = cached_data.get("merchant_tfidf_features", None)
            self.user_merchant_tfidf_features = cached_data.get("user_merchant_tfidf_features", None)

            print("✅ TF-IDF特征加载完成")
            self.is_fitted = True
            return self

        print("🔄 开始训练TF-IDF特征...")

        # 1. 数据预处理和展开
        df = self._prepare_data(X)

        # 2. 构建所有文档
        print("📝 构建文档...")
        user_docs = self._build_user_documents(df)
        merchant_docs = self._build_merchant_documents(df)
        user_merchant_docs = self._build_user_merchant_documents(df)

        # 3. 训练向量化器
        print("🤖 训练向量化器...")
        self._fit_vectorizers(user_docs, merchant_docs, user_merchant_docs)

        # 4. 预计算所有特征矩阵
        print("📊 预计算特征矩阵...")
        self.user_tfidf_features = self._compute_user_features(user_docs)
        self.merchant_tfidf_features = self._compute_merchant_features(merchant_docs)
        self.user_merchant_tfidf_features = self._compute_user_merchant_features(user_merchant_docs)

        self.is_fitted = True
        print("✅ TF-IDF特征训练完成")

        # 5. 缓存特征
        if self.enable_cache:
            print("💾 缓存TF-IDF特征...")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            cached_data = {
                "user_tfidf_features": self.user_tfidf_features,
                "merchant_tfidf_features": self.merchant_tfidf_features,
                "user_merchant_tfidf_features": self.user_merchant_tfidf_features,
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"✅ 特征已缓存到: {self.cache_path}")

        # 清理内存
        del df, user_docs, merchant_docs, user_merchant_docs
        gc.collect()

        return self

    def transform(self, X):
        """通过关联预计算的特征矩阵来转换数据"""
        if not self.is_fitted:
            raise ValueError("TfidfFeatureTransformer has not been fitted yet.")

        print("🔄 开始TF-IDF特征转换...")

        df = X.copy()

        # 1. 关联用户TF-IDF特征
        if self.user_tfidf_features is not None:
            df = df.merge(self.user_tfidf_features, on="user_id", how="left")
            print(f"  ✅ 关联用户TF-IDF特征: {self.user_tfidf_features.shape[1] - 1} 维")

        # 2. 关联商户TF-IDF特征
        if self.merchant_tfidf_features is not None:
            df = df.merge(self.merchant_tfidf_features, on="merchant_id", how="left")
            print(f"  ✅ 关联商户TF-IDF特征: {self.merchant_tfidf_features.shape[1] - 1} 维")

        # 3. 关联用户-商户交互TF-IDF特征
        if self.user_merchant_tfidf_features is not None:
            df = df.merge(self.user_merchant_tfidf_features, on=["user_id", "merchant_id"], how="left")
            print(f"  ✅ 关联用户-商户TF-IDF特征: {self.user_merchant_tfidf_features.shape[1] - 2} 维")

        # 4. 移除不需要的列
        drop_cols = ["activity_log"]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        print(f"✅ TF-IDF特征转换完成，最终特征维度: {df.shape[1]}")
        return df

    def _prepare_data(self, X):
        """准备和展开数据"""
        df = X.copy()
        df = df[df["activity_log"].notnull()]
        df = df.assign(activity_log=df["activity_log"].str.split("#")).explode("activity_log")

        split_columns = ["item_id", "cate_id", "brand_id", "time", "action_type"]
        df[split_columns] = df["activity_log"].str.split(":", expand=True)

        return df

    def _build_user_documents(self, df):
        """构建用户行为文档"""
        user_docs = {}

        for feature_type, column in [
            ("cat", "cate_id"),
            ("brand", "brand_id"),
            ("item", "item_id"),
            ("merchant", "merchant_id"),
        ]:
            user_grouped = df.groupby("user_id")[column].apply(lambda x: " ".join(x.astype(str).tolist())).to_dict()
            user_docs[feature_type] = user_grouped

        return user_docs

    def _build_merchant_documents(self, df):
        """构建商户交互文档"""
        merchant_docs = {}

        for feature_type, column in [
            ("cat", "cate_id"),
            ("brand", "brand_id"),
            ("item", "item_id"),
            ("user", "user_id"),
        ]:
            merchant_grouped = (
                df.groupby("merchant_id")[column].apply(lambda x: " ".join(x.astype(str).tolist())).to_dict()
            )
            merchant_docs[feature_type] = merchant_grouped

        return merchant_docs

    def _build_user_merchant_documents(self, df):
        """构建用户-商户交互文档"""
        user_merchant_docs = {}

        for feature_type, column in [("cat", "cate_id"), ("brand", "brand_id"), ("item", "item_id")]:
            user_merchant_grouped = df.groupby(["user_id", "merchant_id"])[column].apply(
                lambda x: " ".join(x.astype(str).tolist())
            )

            # 转换为字典，键为 "user_id_merchant_id"
            user_merchant_dict = {}
            for (user_id, merchant_id), doc in user_merchant_grouped.items():
                key = f"{user_id}_{merchant_id}"
                user_merchant_dict[key] = doc

            user_merchant_docs[feature_type] = user_merchant_dict

        return user_merchant_docs

    def _fit_vectorizers(self, user_docs, merchant_docs, user_merchant_docs):
        """训练所有向量化器"""
        # 训练用户向量化器
        for feature_type in ["cat", "brand", "item", "merchant"]:
            print(f"  🔤 训练用户{feature_type}向量化器...")
            docs = list(user_docs[feature_type].values())
            if docs:
                self.user_vectorizers[feature_type].fit(docs)

        # 训练商户向量化器
        for feature_type in ["cat", "brand", "item", "user"]:
            print(f"  🏪 训练商户{feature_type}向量化器...")
            docs = list(merchant_docs[feature_type].values())
            if docs:
                self.merchant_vectorizers[feature_type].fit(docs)

        # 训练用户-商户交互向量化器
        for feature_type in ["cat", "brand", "item"]:
            print(f"  🤝 训练用户-商户{feature_type}向量化器...")
            docs = list(user_merchant_docs[feature_type].values())
            if docs:
                self.user_merchant_vectorizers[feature_type].fit(docs)

    def _compute_user_features(self, user_docs):
        """预计算所有用户的TF-IDF特征"""
        print("  📊 计算用户TF-IDF特征...")

        all_user_ids = set()
        for feature_type in ["cat", "brand", "item", "merchant"]:
            all_user_ids.update(user_docs[feature_type].keys())

        all_user_ids = sorted(list(all_user_ids))

        # 为每种特征类型计算TF-IDF
        user_features = []
        feature_names = ["user_id"]

        for feature_type in ["cat", "brand", "item", "merchant"]:
            vectorizer = self.user_vectorizers[feature_type]

            # 准备文档列表
            docs = [user_docs[feature_type].get(user_id, "") for user_id in all_user_ids]

            if any(docs):
                try:
                    tfidf_matrix = vectorizer.transform(docs)
                    # 转换为DataFrame
                    n_features = tfidf_matrix.shape[1]
                    columns = [f"user_{feature_type}_tfidf_{i}" for i in range(n_features)]
                    feature_names.extend(columns)
                    user_features.append(tfidf_matrix.toarray())
                except Exception as e:
                    print(f"⚠️ 用户{feature_type}特征计算失败: {e}")

        if user_features:
            # 合并所有特征
            combined_features = pd.DataFrame(
                index=all_user_ids,
                data=hstack([csr_matrix(feat) for feat in user_features]).toarray(),
                columns=feature_names[1:],  # 排除user_id列
            )
            combined_features.reset_index(inplace=True)
            combined_features.rename(columns={"index": "user_id"}, inplace=True)
            return combined_features
        else:
            return pd.DataFrame({"user_id": all_user_ids})

    def _compute_merchant_features(self, merchant_docs):
        """预计算所有商户的TF-IDF特征"""
        print("  🏪 计算商户TF-IDF特征...")

        all_merchant_ids = set()
        for feature_type in ["cat", "brand", "item", "user"]:
            all_merchant_ids.update(merchant_docs[feature_type].keys())

        all_merchant_ids = sorted(list(all_merchant_ids))

        merchant_features = []
        feature_names = ["merchant_id"]

        for feature_type in ["cat", "brand", "item", "user"]:
            vectorizer = self.merchant_vectorizers[feature_type]

            docs = [merchant_docs[feature_type].get(merchant_id, "") for merchant_id in all_merchant_ids]

            if any(docs):
                try:
                    tfidf_matrix = vectorizer.transform(docs)
                    n_features = tfidf_matrix.shape[1]
                    columns = [f"merchant_{feature_type}_tfidf_{i}" for i in range(n_features)]
                    feature_names.extend(columns)
                    merchant_features.append(tfidf_matrix.toarray())
                except Exception as e:
                    print(f"⚠️ 商户{feature_type}特征计算失败: {e}")

        if merchant_features:
            combined_features = pd.DataFrame(
                index=all_merchant_ids,
                data=hstack([csr_matrix(feat) for feat in merchant_features]).toarray(),
                columns=feature_names[1:],
            )
            combined_features.reset_index(inplace=True)
            combined_features.rename(columns={"index": "merchant_id"}, inplace=True)
            return combined_features
        else:
            return pd.DataFrame({"merchant_id": all_merchant_ids})

    def _compute_user_merchant_features(self, user_merchant_docs):
        """预计算所有用户-商户对的TF-IDF特征"""
        print("  🤝 计算用户-商户TF-IDF特征...")

        # 收集所有用户-商户对
        all_pairs = set()
        for feature_type in ["cat", "brand", "item"]:
            for key in user_merchant_docs[feature_type].keys():
                user_id, merchant_id = key.split("_", 1)
                all_pairs.add((int(user_id), int(merchant_id)))

        all_pairs = sorted(list(all_pairs))

        user_merchant_features = []
        feature_names = ["user_id", "merchant_id"]

        for feature_type in ["cat", "brand", "item"]:
            vectorizer = self.user_merchant_vectorizers[feature_type]

            docs = []
            for user_id, merchant_id in all_pairs:
                key = f"{user_id}_{merchant_id}"
                doc = user_merchant_docs[feature_type].get(key, "")
                docs.append(doc)

            if any(docs):
                try:
                    tfidf_matrix = vectorizer.transform(docs)
                    n_features = tfidf_matrix.shape[1]
                    columns = [f"user_merchant_{feature_type}_tfidf_{i}" for i in range(n_features)]
                    feature_names.extend(columns)
                    user_merchant_features.append(tfidf_matrix.toarray())
                except Exception as e:
                    print(f"⚠️ 用户-商户{feature_type}特征计算失败: {e}")

        if user_merchant_features:
            combined_features = pd.DataFrame(
                data=hstack([csr_matrix(feat) for feat in user_merchant_features]).toarray(), columns=feature_names[2:]
            )
            # 添加用户ID和商户ID列
            combined_features.insert(0, "user_id", [pair[0] for pair in all_pairs])
            combined_features.insert(1, "merchant_id", [pair[1] for pair in all_pairs])
            return combined_features
        else:
            return pd.DataFrame(
                {"user_id": [pair[0] for pair in all_pairs], "merchant_id": [pair[1] for pair in all_pairs]}
            )

    def get_feature_names(self):
        """获取所有特征名称"""
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet.")

        feature_names = []

        if self.user_tfidf_features is not None:
            feature_names.extend([col for col in self.user_tfidf_features.columns if col != "user_id"])

        if self.merchant_tfidf_features is not None:
            feature_names.extend([col for col in self.merchant_tfidf_features.columns if col != "merchant_id"])

        if self.user_merchant_tfidf_features is not None:
            feature_names.extend(
                [col for col in self.user_merchant_tfidf_features.columns if col not in ["user_id", "merchant_id"]]
            )

        return feature_names


if __name__ == "__main__":
    # 创建一个简单的测试数据集
    import numpy as np

    # 模拟数据进行测试
    np.random.seed(42)
    n_samples = 1000

    test_data = {
        "user_id": np.random.randint(1, 100, n_samples),
        "merchant_id": np.random.randint(1, 50, n_samples),
        "activity_log": [
            f"item_{np.random.randint(1, 500)}:cat_{np.random.randint(1, 20)}:brand_{np.random.randint(1, 30)}:202401:1#"
            + f"item_{np.random.randint(1, 500)}:cat_{np.random.randint(1, 20)}:brand_{np.random.randint(1, 30)}:202402:2"
            for _ in range(n_samples)
        ],
    }

    df = pd.DataFrame(test_data)
    X = df
    y = np.random.randint(0, 2, n_samples)

    print("🧪 使用模拟数据进行测试...")
    print(f"模拟数据形状: {X.shape}")

    transformer = TfidfFeatureTransformer(top_n_features=50, min_df=1, max_df=0.95)
    transformer.fit(X, y)

    transformed_df = transformer.transform(X)
    print(f"转换后的数据形状: {transformed_df.shape}")
    print(f"特征名称前10个: {transformer.get_feature_names()[:10]}")
    print(f"转换后的数据预览:\n{transformed_df.head()}")
