import gc
import os
import pickle

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BehaviorTransformer(BaseEstimator, TransformerMixin):
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
        # df = df.drop(columns=["user_id", "merchant_id", "activity_log"], errors="ignore")
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
