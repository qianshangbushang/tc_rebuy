import gc
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .data import load_data, load_dataframe
except ImportError:
    from data import load_data, load_dataframe


class CacheFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, enable_cache: bool = True, cache_path: str = "./output/cache_feature.pkl") -> None:
        self.user_feature = None
        self.merchant_feature = None
        self.user_merchant_feature = None

        self.enable_cache = enable_cache
        self.cache_path = cache_path

    def fit(self, X, y=None):
        if self.enable_cache and os.path.exists(self.cache_path):
            print(f"♻️ 从缓存加载特征: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                feature = pickle.load(f)
            self.user_feature = feature.get("user_feature", None)
            self.merchant_feature = feature.get("merchant_feature", None)
            self.user_merchant_feature = feature.get("user_merchant_feature", None)
            print("✅ 特征加载完成")
            return

        print("🔄 计算用户、商户及用户-商户交互特征...")
        df = X.copy()
        df = self.explode(df)
        self.create_user_feature(df)
        self.create_merchant_feature(df)
        self.create_user_merchant_feature(df)
        print("✅ 特征计算完成")

        if self.enable_cache:
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {
                        "user_feature": self.user_feature,
                        "merchant_feature": self.merchant_feature,
                        "user_merchant_feature": self.user_merchant_feature,
                    },
                    f,
                )
        return self

    def transform(self, X):
        df = X.copy()
        df = df.merge(self.user_feature, on="user_id", how="left")
        df = df.merge(self.merchant_feature, on="merchant_id", how="left")
        df = df.merge(self.user_merchant_feature, on=["user_id", "merchant_id"], how="left")
        df = df.drop(columns=["user_id", "merchant_id", "activity_log"], errors="ignore")
        print(f"🔄 特征合并后数据形状: {df.shape}")
        return df

    def explode(self, df: pd.DataFrame):
        df = df[df["activity_log"].notnull()]
        df = df.assign(activity_log=df["activity_log"].str.split("#")).explode("activity_log")

        split_columns = ["item_id", "cate_id", "brand_id", "time", "action_type"]
        df[split_columns] = df["activity_log"].str.split(":", expand=True)
        return df

    def create_user_feature(self, df: pd.DataFrame):
        print("🔄 计算用户特征...")
        # 每个用户不同action类型的行为占比
        action_ratio = (
            df.groupby(["user_id", "action_type"])
            .size()
            .div(df.groupby("user_id").size(), level="user_id")
            .unstack(fill_value=0)
            .add_prefix("action_ratio_")
        )
        action_ratio.columns.name = None

        # 计算每个用户每个月的行为占比
        df["month"] = df["time"].str[:2].astype(int)
        time_ratio = (
            df.groupby(["user_id", "month"])
            .size()
            .div(df.groupby("user_id").size(), level="user_id")
            .unstack(fill_value=0)
            .add_prefix("time_ratio_")
        )
        time_ratio.columns.name = None

        # 计算每个用户每个月不同action的占比。
        time_action_ratio = df.pivot_table(
            index="user_id",
            columns=["month", "action_type"],
            aggfunc="size",
            fill_value=0,
        ).div(df.groupby("user_id").size(), axis=0)
        time_action_ratio.columns = [
            f"time_action_ratio_month_{month}_action_{action}" for month, action in time_action_ratio.columns
        ]

        # 交互统计特征
        user_stats = df.groupby("user_id").agg(
            user_item_count=("item_id", "nunique"),
            user_cate_count=("cate_id", "nunique"),
            user_brand_count=("brand_id", "nunique"),
            user_merchant_count=("merchant_id", "nunique"),
            user_action_count=("action_type", "count"),  # 总行为次数
        )

        # 合并所有特征
        features = (
            action_ratio.join(time_ratio, how="outer")
            .join(time_action_ratio, how="outer")
            .join(user_stats, how="outer")
        )

        self.user_feature = features.reset_index()
        del action_ratio, time_ratio, time_action_ratio, user_stats, features, df
        gc.collect()
        return self

    def create_merchant_feature(self, df: pd.DataFrame):
        print("🔄 计算商户特征...")
        # 店铺各种action的占比
        action_ratio = (
            df.groupby(["merchant_id", "action_type"])
            .size()
            .div(df.groupby("merchant_id").size(), level="merchant_id")
            .unstack(fill_value=0)
            .add_prefix("merchant_action_ratio_")
        )
        action_ratio.columns.name = None

        df["month"] = df["time"].str[:2].astype(int)

        # 计算每个商户每个月的行为占比
        time_ratio = (
            df.groupby(["merchant_id", "month"])
            .size()
            .div(df.groupby("merchant_id").size(), level="merchant_id")
            .unstack(fill_value=0)
            .add_prefix("merchant_time_ratio_")
        )
        time_ratio.columns.name = None

        # 计算每个商户每个月不同action的占比。
        time_action_ratio = df.pivot_table(
            index="merchant_id",
            columns=["month", "action_type"],
            aggfunc="size",
            fill_value=0,
        ).div(df.groupby(["merchant_id"]).size(), axis=0)
        time_action_ratio.columns = [
            f"merchant_time_action_ratio_month_{month}_action_{action}" for month, action in time_action_ratio.columns
        ]

        # 交互统计特征
        merch_stats = df.groupby("merchant_id").agg(
            merch_item_count=("item_id", "nunique"),
            merch_cate_count=("cate_id", "nunique"),
            merch_brand_count=("brand_id", "nunique"),
            merch_user_count=("user_id", "nunique"),
            merch_action_count=("action_type", "count"),  # 总行为次数
        )
        # 合并所有特征
        self.features = (
            action_ratio.join(time_ratio, how="outer")
            .join(time_action_ratio, how="outer")
            .join(merch_stats, how="outer")
        )

        self.merchant_feature = self.features.reset_index()
        del action_ratio, time_ratio, time_action_ratio, merch_stats, df
        gc.collect()
        return self

    def create_user_merchant_feature(self, df: pd.DataFrame):
        print("🔄 计算用户-商户交互特征...")
        user_merchant_interactions = df.groupby(["user_id", "merchant_id"]).size()
        user_total_interactions = df.groupby("user_id").size()

        user_merchant_ratio = (
            user_merchant_interactions.div(user_total_interactions, level="user_id")
            .reset_index()
            .rename(columns={0: "user_merchant_interaction_ratio"})
        )

        merchant_total_interactions = df.groupby("merchant_id").size()
        merchant_user_ratio = (
            user_merchant_interactions.div(merchant_total_interactions, level="merchant_id")
            .reset_index()
            .rename(columns={0: "merchant_user_interaction_ratio"})
        )

        user_merchant_features = user_merchant_ratio.merge(
            merchant_user_ratio, on=["user_id", "merchant_id"], how="outer"
        )

        self.user_merchant_feature = user_merchant_features.reset_index()
        del (
            user_merchant_ratio,
            merchant_user_ratio,
            user_merchant_interactions,
            user_total_interactions,
            merchant_total_interactions,
            df,
        )
        gc.collect()
        return self


class ModelConfig(BaseModel):
    model_type: str = "rf"  # 可选 'rf', 'xgb', 'lgb'
    n_estimators: int = 200
    max_depth: int = 10
    learning_rate: float = 0.1
    scale_pos_weight: float = 7.0  # 用于处理类别不平衡


class TCDataConfig(BaseModel):
    fill_median_cols: list[str] = []
    fill_mode_cols: list[str] = []

    cache_clean_result: bool = False
    cache_clean_path: str = "data/cleaned_data.pkl"

    model: ModelConfig = ModelConfig()


class UserMerchantFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.user_merchant_features = None
        return

    def fit(self, X, y=None):
        X_copy = X.copy()

        user_merchant_interactions = X_copy.groupby(["user_id", "merchant_id"]).size()
        user_total_interactions = X_copy.groupby("user_id").size()

        user_merchant_ratio = (
            user_merchant_interactions.div(user_total_interactions, level="user_id")
            .reset_index()
            .rename(columns={0: "user_merchant_interaction_ratio"})
        )

        merchant_total_interactions = X_copy.groupby("merchant_id").size()
        merchant_user_ratio = (
            user_merchant_interactions.div(merchant_total_interactions, level="merchant_id")
            .reset_index()
            .rename(columns={0: "merchant_user_interaction_ratio"})
        )

        user_merchant_features = user_merchant_ratio.merge(
            merchant_user_ratio, on=["user_id", "merchant_id"], how="outer"
        )

        self.user_merchant_features = user_merchant_features.reset_index()
        del (
            user_merchant_ratio,
            merchant_user_ratio,
            user_merchant_interactions,
            user_total_interactions,
            merchant_total_interactions,
            X_copy,
        )
        gc.collect()
        return self

    def transform(self, X):
        """✅ 实现特征合并逻辑"""
        if self.user_merchant_features is None:
            raise ValueError("Transformer has not been fitted yet.")

        X_transformed = X.copy()

        # 合并用户-商户交互特征
        if all(col in X_transformed.columns for col in ["user_id", "merchant_id"]):
            X_transformed = X_transformed.merge(self.user_merchant_features, on=["user_id", "merchant_id"], how="left")
        else:
            print("⚠️ 输入数据缺少 user_id 或 merchant_id 列，跳过用户-商户特征合并")

        return X_transformed


class UserFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = None
        return

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy["activity_log"].notnull()]
        X_copy = X_copy.assign(activity_log=X_copy["activity_log"].str.split("#")).explode("activity_log")

        X_copy[["item_id", "cate_id", "brand_id", "time", "action_type"]] = X_copy["activity_log"].str.split(
            ":", expand=True
        )

        # 每个用户不同action类型的行为占比
        action_ratio = (
            X_copy.groupby(["user_id", "action_type"])
            .size()
            .div(X_copy.groupby("user_id").size(), level="user_id")
            .unstack(fill_value=0)
            .add_prefix("action_ratio_")
        )
        action_ratio.columns.name = None

        # 计算每个用户每个月的行为占比
        X_copy["month"] = X_copy["time"].str[:2].astype(int)
        time_ratio = (
            X_copy.groupby(["user_id", "month"])
            .size()
            .div(X_copy.groupby("user_id").size(), level="user_id")
            .unstack(fill_value=0)
            .add_prefix("time_ratio_")
        )
        time_ratio.columns.name = None

        # 计算每个用户每个月不同action的占比。
        time_action_ratio = X_copy.pivot_table(
            index="user_id",
            columns=["month", "action_type"],
            aggfunc="size",
            fill_value=0,
        ).div(X_copy.groupby("user_id").size(), axis=0)
        time_action_ratio.columns = [
            f"time_action_ratio_month_{month}_action_{action}" for month, action in time_action_ratio.columns
        ]

        # 交互统计特征
        user_stats = X_copy.groupby("user_id").agg(
            user_item_count=("item_id", "nunique"),
            user_cate_count=("cate_id", "nunique"),
            user_brand_count=("brand_id", "nunique"),
            user_merchant_count=("merchant_id", "nunique"),
            user_action_count=("action_type", "count"),  # 总行为次数
        )

        # 合并所有特征
        features = (
            action_ratio.join(time_ratio, how="outer")
            .join(time_action_ratio, how="outer")
            .join(user_stats, how="outer")
        )

        self.features = features.reset_index()
        del action_ratio, time_ratio, time_action_ratio, user_stats, features, X_copy
        gc.collect()
        return self

    def transform(self, X):
        # 假设 X 是一个包含用户信息的 DataFrame
        # 这里可以添加更多的特征工程步骤
        if self.features is None:
            raise ValueError("The transformer has not been fitted yet.")
        X_transformed = X.copy()
        X_transformed = X_transformed.merge(self.features, how="left", on="user_id")
        return X_transformed


class MerchantFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = None
        return

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy["activity_log"].notnull()]
        X_copy = X_copy.assign(activity_log=X_copy["activity_log"].str.split("#")).explode("activity_log")

        X_copy[["item_id", "cate_id", "brand_id", "time", "action_type"]] = X_copy["activity_log"].str.split(
            ":", expand=True
        )

        # 店铺各种action的占比
        action_ratio = (
            X_copy.groupby(["merchant_id", "action_type"])
            .size()
            .div(X_copy.groupby("merchant_id").size(), level="merchant_id")
            .unstack(fill_value=0)
            .add_prefix("merchant_action_ratio_")
        )
        action_ratio.columns.name = None

        X_copy["month"] = X_copy["time"].str[:2].astype(int)

        # 计算每个商户每个月的行为占比
        time_ratio = (
            X_copy.groupby(["merchant_id", "month"])
            .size()
            .div(X_copy.groupby("merchant_id").size(), level="merchant_id")
            .unstack(fill_value=0)
            .add_prefix("merchant_time_ratio_")
        )
        time_ratio.columns.name = None

        # 计算每个商户每个月不同action的占比。
        time_action_ratio = X_copy.pivot_table(
            index="merchant_id",
            columns=["month", "action_type"],
            aggfunc="size",
            fill_value=0,
        ).div(X_copy.groupby(["merchant_id"]).size(), axis=0)
        time_action_ratio.columns = [
            f"merchant_time_action_ratio_month_{month}_action_{action}" for month, action in time_action_ratio.columns
        ]

        # 交互统计特征
        merch_stats = X_copy.groupby("merchant_id").agg(
            merch_item_count=("item_id", "nunique"),
            merch_cate_count=("cate_id", "nunique"),
            merch_brand_count=("brand_id", "nunique"),
            merch_user_count=("user_id", "nunique"),
            merch_action_count=("action_type", "count"),  # 总行为次数
        )
        # 合并所有特征
        self.features = (
            action_ratio.join(time_ratio, how="outer")
            .join(time_action_ratio, how="outer")
            .join(merch_stats, how="outer")
        )

        self.features = self.features.reset_index()
        del action_ratio, time_ratio, time_action_ratio, merch_stats, X_copy
        gc.collect()

        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.merge(self.features, how="left", on="merchant_id")
        return X_transformed


def create_clean_pipeline(conf: TCDataConfig) -> Pipeline:
    """Create a data cleaning pipeline."""
    column_transformer = ColumnTransformer(
        [
            ("median_fill", SimpleImputer(strategy="median"), conf.fill_median_cols),
            ("mode_fill", SimpleImputer(strategy="most_frequent"), conf.fill_mode_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        verbose=True,
    )
    column_transformer.set_output(transform="pandas")

    steps = [
        ("column_transformer", column_transformer),
        ("user_feat", UserFeatureTransformer()),
        ("merch_feat", MerchantFeatureTransformer()),
        ("user_merchant_feat", UserMerchantFeatureTransformer()),
    ]
    return Pipeline(steps, verbose=True)


def create_sample_pipeline(conf: TCDataConfig):
    """创建数据采样管道"""
    try:
        steps = [("smote", SMOTE(random_state=42, k_neighbors=5))]

        return ImbPipeline(steps)
    except ImportError:
        print("⚠️ imbalanced-learn 未安装，跳过采样管道")
        return None


def create_model_pipeline(conf: ModelConfig) -> Pipeline:
    """创建模型训练管道，支持多种模型"""
    steps = [("scaler", StandardScaler())]

    if conf.model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=conf.n_estimators,
            max_depth=conf.max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif conf.model_type == "xgb":
        if xgb is None:
            raise ImportError("XGBoost 未安装，请安装后使用: pip install xgboost")
        classifier = xgb.XGBClassifier(
            n_estimators=conf.n_estimators,
            max_depth=conf.max_depth,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=conf.scale_pos_weight,
        )
    elif conf.model_type == "lgb":
        if lgb is None:
            raise ImportError("LightGBM 未安装，请安装后使用: pip install lightgbm")
        classifier = lgb.LGBMClassifier(
            n_estimators=conf.n_estimators,
            max_depth=conf.max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        raise ValueError("model_type 必须是 'rf', 'xgb' 或 'lgb'")

    steps.append(("classifier", classifier))
    return Pipeline(steps)


class TCDataPipeline:
    def __init__(self, conf: TCDataConfig):
        self.conf = conf
        self.cache_feature_transformer = CacheFeatureTransformer(
            enable_cache=conf.cache_clean_result,
            cache_path=conf.cache_clean_path,
        )
        self.clean_pipe = create_clean_pipeline(conf)
        self.sample_pipe = None  # create_sample_pipeline(conf)
        self.model_pipe = create_model_pipeline(conf.model)

        # 存储数据集
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # 存储清洗后的完整数据
        self.X_clean = None
        self.y_clean = None

        # 训练状态
        self.is_fitted = False

    def _clean(self, X, y=None):
        # 1. 数据清洗和特征工程
        if self.conf.cache_clean_result:
            if os.path.exists(self.conf.cache_clean_path):
                print(f"♻️ 从缓存加载清洗后的数据: {self.conf.cache_clean_path}")
                with open(self.conf.cache_clean_path, "rb") as f:
                    d = pickle.load(f)
                # cleaned_data = pd.read_pickle(self.conf.cache_clean_path)
                X_clean, y_clean = d["X"], d["y"]
            else:
                X_clean = self.clean_pipe.fit_transform(X, y)
                y_clean = y.copy()
                cleaned_data = {"X": X_clean, "y": y_clean}
                with open(self.conf.cache_clean_path, "wb") as f:
                    pickle.dump(cleaned_data, f)
                print(f"✅ 清洗后的数据已缓存到: {self.conf.cache_clean_path}")
        else:
            X_clean = self.clean_pipe.fit_transform(X, y)
            y_clean = y.copy()

        self.X_clean = X_clean
        self.y_clean = y_clean

    # def split_dataset(self, X, y, val_size=0.2, random_state=42):
    #     train_data_mask = y.isin([0, 1])
    #     test_data_mask = y.isnull()

    #     train_X, train_y = X[train_data_mask], y[train_data_mask]
    #     test_X, test_y = X[test_data_mask], y[test_data_mask]

    #     assert len(train_X) == len(train_y), "训练集特征和标签数量不匹配"
    #     assert train_y.nunique() > 1, "训练集标签必须至少包含两个类别"
    #     assert test_y.isnull().all(), "测试集标签必须全部为空"
    #     print(f"训练数据形状: {train_X.shape}, 测试数据形状: {test_X.shape}")

    #     self.cache_feature_transformer.fit(X, y)
    #     train_X = self.cache_feature_transformer.transform(train_X)

    #     print("\n🔄 Step 2: 数据集拆分...")
    #     train_X, val_X, train_y, val_y = train_test_split(
    #         train_X, train_y, test_size=val_size, random_state=random_state, stratify=train_y
    #     )
    #     return train_X, train_y, val_X, val_y, test_X, test_y

    def preprocess(self, X: pd.DataFrame, y: pd.Series = None):
        """
        仅进行数据清洗和特征工程
        """
        self.cache_feature_transformer.fit(X, y)

        train_mask = y.isin([0, 1])
        test_mask = y.isnull()

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        X_train = self.cache_feature_transformer.transform(X_train)
        X_test = self.cache_feature_transformer.transform(X_test)

        return X_train, y_train, X_test, y_test

    def fit(self, X, y):
        """
        完整的训练流程
        """
        train_X, train_y, test_X, test_y = self.preprocess(X, y)
        self.summary(train_X, None, test_X)
        # print("🔄 Step 1: 数据清洗和特征工程...")
        # self._clean(X, y)
        # self.X_clean = self.X_clean.drop(columns=["user_id", "merchant_id", "activity_log"], errors="ignore")
        # print(f"✅ 清洗后数据形状: {self.X_clean.shape}")
        # print(f"✅ 特征数量: {self.X_clean.shape[1]}")

        # 2. 数据集拆分
        # self.split_dataset(val_size, random_state)
        # self.X_train = self.X_train.drop(columns=["user_id", "merchant_id", "activity_log"], errors="ignore")
        # 3. 数据采样 (如果有训练数据且配置了采样管道)
        if self.sample_pipe is not None and train_X is not None:
            print("\n🔄 Step 3: 数据采样...")
            try:
                train_X, train_y = self.sample_pipe.fit_resample(train_X, train_y)
                print(f"✅ 采样后训练集形状: {train_X.shape}")
                print(f"✅ 采样后标签分布:\n{pd.Series(train_y).value_counts()}")
            except Exception as e:
                print(f"⚠️ 采样失败，跳过采样步骤: {e}")

        # 4. 模型训练 (如果有训练数据且配置了模型管道)
        if self.model_pipe is not None and train_X is not None and train_y is not None:
            print("\n🔄 Step 4: 模型训练...")
            try:
                self.feature_names = train_X.columns.tolist()
                self.model_pipe.fit(train_X, train_y)
                print("✅ 模型训练完成")
                self.is_fitted = True
            except Exception as e:
                print(f"❌ 模型训练失败: {e}")
                self.is_fitted = False
        return self

    # def split_dataset(self, val_size=0.2, random_state=42):
    #     """
    #     拆分数据集，支持有测试集标签为空的情况
    #     """
    #     X_clean = self.X_clean.reset_index(drop=True)
    #     y_clean = self.y_clean.reset_index(drop=True)

    #     print("数据集大小： ", X_clean.shape, y_clean.shape)
    #     # 检查是否有空标签（测试集）
    #     if y_clean.isnull().any():
    #         print("📋 检测到空标签，将其作为测试集...")
    #         test_mask = y_clean.isnull()
    #         self.X_test = X_clean[test_mask].reset_index(drop=True)
    #         self.y_test = y_clean[test_mask].reset_index(drop=True)

    #         # 剩余的作为训练+验证集
    #         train_val_mask = y_clean.isin([1, 0])
    #         X_train_val = X_clean[train_val_mask].reset_index(drop=True)
    #         y_train_val = y_clean[train_val_mask].reset_index(drop=True)
    #     else:
    #         print("📋 未检测到空标签，从完整数据中拆分测试集...")
    #         # 如果没有空标签，则随机拆分测试集（20%）
    #         X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
    #             X_clean,
    #             y_clean,
    #             test_size=0.2,
    #             random_state=random_state,
    #             stratify=y_clean if y_clean.nunique() > 1 else None,
    #         )

    #     # 从训练+验证集中拆分训练集和验证集
    #     if val_size > 0 and len(X_train_val) > 1:
    #         try:
    #             self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
    #                 X_train_val,
    #                 y_train_val,
    #                 test_size=val_size,
    #                 random_state=random_state,
    #                 stratify=y_train_val if y_train_val.nunique() > 1 else None,
    #             )
    #         except ValueError as e:
    #             print(f"⚠️ 分层拆分失败，使用随机拆分: {e}")
    #             self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
    #                 X_train_val,
    #                 y_train_val,
    #                 test_size=val_size,
    #                 random_state=random_state,
    #             )
    #     else:
    #         # 如果不需要验证集或数据太少
    #         self.X_train = X_train_val
    #         self.y_train = y_train_val
    #         self.X_val = None
    #         self.y_val = None

    #     # 打印拆分结果
    #     print(f"✅ 训练集: {self.X_train.shape if self.X_train is not None else 'None'}")
    #     print(f"✅ 验证集: {self.X_val.shape if self.X_val is not None else 'None'}")
    #     print(f"✅ 测试集: {self.X_test.shape if self.X_test is not None else 'None'}")

    #     # 打印标签分布
    #     if self.y_train is not None:
    #         print(f"训练集标签分布:\n{pd.Series(self.y_train).value_counts()}")
    #     if self.y_val is not None:
    #         print(f"验证集标签分布:\n{pd.Series(self.y_val).value_counts()}")

    def transform(self, X):
        """对新数据进行预处理"""
        if self.clean_pipe is None:
            raise ValueError("Pipeline has not been fitted yet.")

        return self.clean_pipe.transform(X)

    def predict(self, X):
        """预测新数据"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")

        # X_processed = self.transform(X)
        return self.model_pipe.predict(X[self.feature_names])

    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")

        # X_processed = self.transform(X)
        return self.model_pipe.predict_proba(X[self.feature_names])

    def evaluate(self, val_X, val_y):
        """评估模型性能"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")

        if val_X is None or val_y is None:
            print("❌ 验证集数据不存在")
            return None

        # 过滤掉空标签
        mask = val_y.notna()
        if not mask.any():
            print("❌ 验证集没有有效标签")
            return None

        X_eval_clean = val_X[mask]
        y_eval_clean = val_y[mask]

        # 预测
        y_pred = self.model_pipe.predict(X_eval_clean[self.feature_names])

        # 计算指标
        accuracy = accuracy_score(y_eval_clean, y_pred)

        print("📊 验证集评估结果:")
        print(f"准确率: {accuracy:.4f}")

        # 如果是二分类，计算AUC
        if hasattr(self.model_pipe, "predict_proba") and len(np.unique(y_eval_clean)) == 2:
            try:
                y_pred_proba = self.model_pipe.predict_proba(X_eval_clean[self.feature_names])
                auc = roc_auc_score(y_eval_clean, y_pred_proba[:, 1])
                print(f"AUC: {auc:.4f}")
            except Exception as e:
                print(f"⚠️ AUC计算失败: {e}")
                auc = None
        else:
            auc = None

        print(f"\n详细报告:\n{classification_report(y_eval_clean, y_pred)}")

        return {
            "accuracy": accuracy,
            "auc": auc,
            "y_true": y_eval_clean,
            "y_pred": y_pred,
        }

    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        if not self.is_fitted:
            print("模型尚未训练")
            return None

        # 尝试获取特征重要性
        model = self.model_pipe
        if hasattr(model, "named_steps"):
            # 如果是Pipeline，获取最后一个步骤
            final_step = list(model.named_steps.values())[-1]
        else:
            final_step = model

        if hasattr(final_step, "feature_importances_"):
            importances = final_step.feature_importances_
            feature_names = self.X_train.columns

            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
                "importance", ascending=False
            )

            print(f"🔝 Top {top_n} 重要特征:")
            print(importance_df.head(top_n))

            return importance_df
        else:
            print("模型不支持特征重要性分析")
            return None

    def summary(self, train_X, val_X, test_X):
        """打印管道摘要信息"""
        print("📋 TCDataPipeline 摘要:")
        print(f"数据清洗管道: {self.clean_pipe is not None}")
        print(f"采样管道: {self.sample_pipe is not None}")
        print(f"模型管道: {self.model_pipe is not None}")
        print(f"模型已训练: {self.is_fitted}")

        if train_X is not None:
            print(f"训练集: {train_X.shape}")
        if val_X is not None:
            print(f"验证集: {val_X.shape}")
        if test_X is not None:
            print(f"测试集: {test_X.shape}")

    def tune_model(self, X, y, param_grid=None, search_type="grid", cv=3, scoring="roc_auc", n_iter=20):
        """
        自动调参，支持 RF/XGB/LGB
        param_grid: dict，参数搜索空间
        search_type: "grid" 或 "random"
        """

        train_X, train_y, _, _ = self.preprocess(X, y)
        if train_X is None or train_y is None:
            print("❌ 训练集不存在，无法调参")
            return None

        if param_grid is None:
            # 默认参数空间
            if self.conf.model.model_type == "rf":
                param_grid = {
                    "classifier__n_estimators": [100, 200, 500],
                    "classifier__max_depth": [4, 6, 10, 16],
                }
            elif self.conf.model.model_type == "xgb":
                param_grid = {
                    "classifier__n_estimators": [200, 500, 1000],
                    "classifier__max_depth": [4, 6, 10],
                    "classifier__learning_rate": [0.05, 0.1, 0.2],
                    "classifier__scale_pos_weight": [5, 7, 10],
                }
            elif self.conf.model.model_type == "lgb":
                param_grid = {
                    "classifier__n_estimators": [200, 500, 1000],
                    "classifier__max_depth": [4, 6, 10],
                    "classifier__learning_rate": [0.05, 0.1, 0.2],
                    "classifier__class_weight": ["balanced", None],
                }

        search_cls = GridSearchCV if search_type == "grid" else RandomizedSearchCV

        if search_type == "grid":
            search = search_cls(
                self.model_pipe,
                param_grid,
                cv=cv,
                scoring=scoring,
                verbose=2,
                n_jobs=-1,
            )
        else:
            search = search_cls(
                self.model_pipe,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_iter=n_iter,
                verbose=2,
                n_jobs=-1,
            )
        print("🔍 开始模型调参...")
        search.fit(train_X, train_y)
        print(f"✅ 最优参数: {search.best_params_}")
        print(f"✅ 最优分数: {search.best_score_:.4f}")

        self.model_pipe = search.best_estimator_
        self.is_fitted = True
        return search

    def export_prediction(self, X: pd.DataFrame, filename="prediction.csv"):
        """
        导出预测结果到 prediction.csv
        格式：user_id, merchant_id, prob
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法导出预测结果。")

        _, _, X_test, _ = self.preprocess(X)

        # 只保留训练用的特征
        drop_cols = ["user_id", "merchant_id", "activity_log"]
        X_pred = X_test.drop(columns=[col for col in drop_cols if col in X_test.columns], errors="ignore")

        # 预测概率
        prob = self.predict_proba(X_pred)[:, 1]  # 取正类概率

        # 构建结果 DataFrame
        result = pd.DataFrame(
            {"user_id": X_test["user_id"].values, "merchant_id": X_test["merchant_id"].values, "prob": prob}
        )

        # 保存为 CSV
        result.to_csv(filename, index=False)
        print(f"✅ 预测结果已保存到 {filename}")


def analysis():
    """Perform data analysis."""
    dataset = load_data()

    print("train f1 info:")
    print(dataset.train_f1.info())
    print("=" * 50)

    print("user f1 info: ")
    print(dataset.user_f1.info())
    print("=" * 50)

    print("log f1 info: ")
    print(dataset.log_f1.info())
    print("=" * 50)

    print("train f2 info: ")
    print(dataset.train_f2.info())
    print("=" * 50)

    print("test f2 info:")
    print(dataset.test_f2.info())
    print("=" * 50)


def run():
    df = load_dataframe()
    X, y = df.drop(columns=["label"]), df["label"]
    pipe = create_clean_pipeline(
        TCDataConfig(
            fill_median_cols=["age_range"],
            fill_mode_cols=["gender"],
        )
    )
    pipe = TCDataPipeline(
        TCDataConfig(
            fill_median_cols=["age_range"],
            fill_mode_cols=["gender"],
            cache_clean_result=True,
            cache_clean_path="../data/cleaned_data.pkl",
            model=ModelConfig(model_type="lgb", n_estimators=500, max_depth=4, scale_pos_weight=7.0),
        )
    )

    pipe.fit(X, y)
    pipe.tune_model(X, y)
    # pipe.evaluate(stage="val")
    # print(X.head(10))

    if pipe.X_test is not None:
        pipe.export_prediction(X, filename="../output/prediction.csv")


if __name__ == "__main__":
    run()
