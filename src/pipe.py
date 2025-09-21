import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .data import load_data, load_dataframe
except ImportError:
    from data import load_data, load_dataframe


class TCDataConfig(BaseModel):
    fill_median_cols: list[str] = []
    fill_mode_cols: list[str] = []

    cache_clean_result: bool = False
    cache_clean_path: str = "data/cleaned_data.pkl"


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

        self.user_merchant_features = user_merchant_features
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


def create_model_pipeline(conf: TCDataConfig) -> Pipeline:
    """创建模型训练管道"""
    steps = [
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        ),
    ]

    return Pipeline(steps)


class TCDataPipeline:
    def __init__(self, conf: TCDataConfig):
        self.conf = conf
        self.clean_pipe = create_clean_pipeline(conf)
        self.sample_pipe = None # create_sample_pipeline(conf)
        self.model_pipe = create_model_pipeline(conf)

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

    def fit(self, X, y, val_size=0.2, random_state=42):
        """
        完整的训练流程
        """
        print("🔄 Step 1: 数据清洗和特征工程...")
        self._clean(X, y)
        self.X_clean = self.X_clean.drop(columns=["user_id", "merchant_id", "activity_log"], errors="ignore")
        print(f"✅ 清洗后数据形状: {self.X_clean.shape}")
        print(f"✅ 特征数量: {self.X_clean.shape[1]}")

        print("\n🔄 Step 2: 数据集拆分...")
        # 2. 数据集拆分
        self.split_dataset(val_size, random_state)

        # 3. 数据采样 (如果有训练数据且配置了采样管道)
        if self.sample_pipe is not None and self.X_train is not None:
            print("\n🔄 Step 3: 数据采样...")
            try:
                self.X_train, self.y_train = self.sample_pipe.fit_resample(self.X_train, self.y_train)
                print(f"✅ 采样后训练集形状: {self.X_train.shape}")
                print(f"✅ 采样后标签分布:\n{pd.Series(self.y_train).value_counts()}")
            except Exception as e:
                print(f"⚠️ 采样失败，跳过采样步骤: {e}")

        # 4. 模型训练 (如果有训练数据且配置了模型管道)
        if self.model_pipe is not None and self.X_train is not None and self.y_train is not None:
            print("\n🔄 Step 4: 模型训练...")
            try:
                self.model_pipe.fit(self.X_train, self.y_train)
                print("✅ 模型训练完成")
                self.is_fitted = True

                # 训练后立即评估
                if self.X_val is not None:
                    self.evaluate(stage="val")

            except Exception as e:
                print(f"❌ 模型训练失败: {e}")
                self.is_fitted = False

        return self

    def split_dataset(self, val_size=0.2, random_state=42):
        """
        拆分数据集，支持有测试集标签为空的情况
        """
        X_clean = self.X_clean.reset_index(drop=True)
        y_clean = self.y_clean.reset_index(drop=True)

        print("数据集大小： ", X_clean.shape, y_clean.shape)
        # 检查是否有空标签（测试集）
        if y_clean.isnull().any():
            print("📋 检测到空标签，将其作为测试集...")
            test_mask = y_clean.isnull()
            self.X_test = X_clean[test_mask].reset_index(drop=True)
            self.y_test = y_clean[test_mask].reset_index(drop=True)

            # 剩余的作为训练+验证集
            train_val_mask = y_clean.isin([1,0])
            X_train_val = X_clean[train_val_mask].reset_index(drop=True)
            y_train_val = y_clean[train_val_mask].reset_index(drop=True)
        else:
            print("📋 未检测到空标签，从完整数据中拆分测试集...")
            # 如果没有空标签，则随机拆分测试集（20%）
            X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
                X_clean,
                y_clean,
                test_size=0.2,
                random_state=random_state,
                stratify=y_clean if y_clean.nunique() > 1 else None,
            )

        # 从训练+验证集中拆分训练集和验证集
        if val_size > 0 and len(X_train_val) > 1:
            try:
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X_train_val,
                    y_train_val,
                    test_size=val_size,
                    random_state=random_state,
                    stratify=y_train_val if y_train_val.nunique() > 1 else None,
                )
            except ValueError as e:
                print(f"⚠️ 分层拆分失败，使用随机拆分: {e}")
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X_train_val,
                    y_train_val,
                    test_size=val_size,
                    random_state=random_state,
                )
        else:
            # 如果不需要验证集或数据太少
            self.X_train = X_train_val
            self.y_train = y_train_val
            self.X_val = None
            self.y_val = None

        # 打印拆分结果
        print(f"✅ 训练集: {self.X_train.shape if self.X_train is not None else 'None'}")
        print(f"✅ 验证集: {self.X_val.shape if self.X_val is not None else 'None'}")
        print(f"✅ 测试集: {self.X_test.shape if self.X_test is not None else 'None'}")

        # 打印标签分布
        if self.y_train is not None:
            print(f"训练集标签分布:\n{pd.Series(self.y_train).value_counts()}")
        if self.y_val is not None:
            print(f"验证集标签分布:\n{pd.Series(self.y_val).value_counts()}")

    def transform(self, X):
        """对新数据进行预处理"""
        if self.clean_pipe is None:
            raise ValueError("Pipeline has not been fitted yet.")

        return self.clean_pipe.transform(X)

    def predict(self, X):
        """预测新数据"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")

        X_processed = self.transform(X)
        return self.model_pipe.predict(X_processed)

    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")

        X_processed = self.transform(X)
        return self.model_pipe.predict_proba(X_processed)

    def evaluate(self, stage="val"):
        """评估模型性能"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")

        if stage == "val":
            X_eval, y_eval = self.X_val, self.y_val
        elif stage == "test":
            X_eval, y_eval = self.X_test, self.y_test
        elif stage == "train":
            X_eval, y_eval = self.X_train, self.y_train
        else:
            raise ValueError("stage must be 'train', 'val' or 'test'")

        if X_eval is None or y_eval is None:
            print(f"❌ {stage.upper()} 集数据不存在")
            return None

        # 过滤掉空标签
        mask = y_eval.notna()
        if not mask.any():
            print(f"❌ {stage.upper()} 集没有有效标签")
            return None

        X_eval_clean = X_eval[mask]
        y_eval_clean = y_eval[mask]

        # 预测
        y_pred = self.model_pipe.predict(X_eval_clean)

        # 计算指标
        accuracy = accuracy_score(y_eval_clean, y_pred)

        print(f"📊 {stage.upper()} 集评估结果:")
        print(f"准确率: {accuracy:.4f}")

        # 如果是二分类，计算AUC
        if hasattr(self.model_pipe, "predict_proba") and len(np.unique(y_eval_clean)) == 2:
            try:
                y_pred_proba = self.model_pipe.predict_proba(X_eval_clean)
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

    def summary(self):
        """打印管道摘要信息"""
        print("📋 TCDataPipeline 摘要:")
        print(f"数据清洗管道: {self.clean_pipe is not None}")
        print(f"采样管道: {self.sample_pipe is not None}")
        print(f"模型管道: {self.model_pipe is not None}")
        print(f"模型已训练: {self.is_fitted}")

        if self.X_clean is not None:
            print(f"清洗后数据形状: {self.X_clean.shape}")

        if self.X_train is not None:
            print(f"训练集: {self.X_train.shape}")
        if self.X_val is not None:
            print(f"验证集: {self.X_val.shape}")
        if self.X_test is not None:
            print(f"测试集: {self.X_test.shape}")


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
        )
    )

    pipe.summary()
    pipe.fit(X, y)
    pipe.evaluate(stage="val")
    print(X.head(10))


if __name__ == "__main__":
    run()
