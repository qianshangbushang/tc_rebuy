import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PrevTransformer(BaseEstimator, TransformerMixin):
    """先验概率特征转换器，生成用户复购和商户被复购的先验特征"""

    def __init__(
        self,
        min_user_interactions: int = 3,
        min_merchant_interactions: int = 5,
        time_window_days: int = 365,
        repurchase_window_days: int = 90,
        enable_cache: bool = True,
        cache_path: str = "../output/prev_feature.pkl",
        pseudo_count: float = 1.0,
    ):
        """
        Args:
            min_user_interactions: 用户最小交互次数阈值
            min_merchant_interactions: 商户最小交互次数阈值
            time_window_days: 观察时间窗口（天）
            repurchase_window_days: 复购时间窗口（天）
            enable_cache: 是否启用缓存
            cache_path: 缓存文件路径
            pseudo_count: 贝叶斯平滑伪计数
        """
        self.min_user_interactions = min_user_interactions
        self.min_merchant_interactions = min_merchant_interactions
        self.time_window_days = time_window_days
        self.repurchase_window_days = repurchase_window_days
        self.enable_cache = enable_cache
        self.cache_path = cache_path
        self.pseudo_count = pseudo_count

        # 先验概率统计
        self.user_prior_stats = {}  # 用户复购先验统计
        self.merchant_prior_stats = {}  # 商户被复购先验统计
        self.global_stats = {}  # 全局统计

        # 最终特征
        self.user_prior_features = None  # 用户先验特征DataFrame
        self.merchant_prior_features = None  # 商户先验特征DataFrame

        self.is_fitted = False

    def fit(self, X, y=None):
        """训练先验概率特征"""
        if self.enable_cache and os.path.exists(self.cache_path):
            print(f"♻️ 从缓存加载先验特征: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                cached_data = pickle.load(f)

            self.user_prior_features = cached_data["user_prior_features"]
            self.merchant_prior_features = cached_data["merchant_prior_features"]
            self.global_stats = cached_data["global_stats"]

            print("✅ 先验特征加载完成")
            self.is_fitted = True
            return self

        print("🔄 开始训练先验概率特征...")

        # 1. 准备训练数据（只使用有标签的数据）
        if y is not None:
            train_mask = y.isin([0, 1])
            X_train = X[train_mask].copy()
            y_train = y[train_mask].copy()
            print(f"📊 训练数据: {len(X_train)} 样本")
        else:
            X_train = X.copy()
            y_train = None
            print("⚠️ 未提供标签，使用全部数据")

        # 2. 解析活动日志并构建用户-商户交互数据
        interaction_data = self._parse_interactions(X_train)
        print(f"📊 解析出 {len(interaction_data)} 条交互记录")

        # 3. 计算全局统计
        self._calculate_global_stats(interaction_data, y_train)

        # 4. 计算用户先验概率
        self._calculate_user_priors(interaction_data, y_train)

        # 5. 计算商户先验概率
        self._calculate_merchant_priors(interaction_data, y_train)

        # 6. 生成特征DataFrame
        self._generate_feature_dataframes(X_train)

        # 7. 缓存结果
        self._cache_results()

        self.is_fitted = True
        print("✅ 先验概率特征训练完成")
        return self

    def transform(self, X):
        """转换数据为先验概率特征"""
        if not self.is_fitted:
            raise ValueError("PrevTransformer has not been fitted yet.")

        print("🔄 开始先验概率特征转换...")

        result = X.copy()

        # 关联用户先验特征
        if self.user_prior_features is not None:
            result = result.merge(self.user_prior_features, on="user_id", how="left")
            user_features = len([col for col in self.user_prior_features.columns if col != "user_id"])
            print(f"  ✅ 关联用户先验特征: {user_features} 维")

        # 关联商户先验特征
        if self.merchant_prior_features is not None:
            result = result.merge(self.merchant_prior_features, on="merchant_id", how="left")
            merchant_features = len([col for col in self.merchant_prior_features.columns if col != "merchant_id"])
            print(f"  ✅ 关联商户先验特征: {merchant_features} 维")

        # 填充缺失值
        prev_columns = [col for col in result.columns if col.startswith(("user_prev_", "merchant_prev_"))]
        for col in prev_columns:
            if col.startswith("user_prev_"):
                result[col] = result[col].fillna(self.global_stats.get("global_user_repurchase_rate", 0.3))
            elif col.startswith("merchant_prev_"):
                result[col] = result[col].fillna(self.global_stats.get("global_merchant_repurchase_rate", 0.3))

        print(f"✅ 先验概率特征转换完成，新增维度: {result.shape[1] - X.shape[1]}")
        return result

    def _parse_interactions(self, X):
        """解析activity_log，提取用户-商户交互数据"""
        print("📊 解析用户-商户交互数据...")

        interactions = []

        for idx, row in X.iterrows():
            user_id = row["user_id"]
            merchant_id = row["merchant_id"]
            activity_log = row["activity_log"]

            if pd.isna(activity_log):
                continue

            # 解析活动日志
            activities = self._parse_activity_log(activity_log)

            # 提取购买行为
            purchases = [act for act in activities if act.get("action_type") == "2"]

            if purchases:
                interactions.append(
                    {
                        "user_id": user_id,
                        "merchant_id": merchant_id,
                        "purchases": purchases,
                        "total_activities": len(activities),
                        "purchase_count": len(purchases),
                        "first_purchase": min(purchases, key=lambda x: x["timestamp"])["timestamp"],
                        "last_purchase": max(purchases, key=lambda x: x["timestamp"])["timestamp"],
                    }
                )

        return interactions

    def _parse_activity_log(self, activity_log):
        """解析单条activity_log"""
        activities = []

        if pd.isna(activity_log):
            return activities

        try:
            # 分割多个活动记录
            activity_records = activity_log.split("#")

            for record in activity_records:
                if record.strip():
                    parts = record.split(":")
                    if len(parts) >= 5:
                        activities.append(
                            {
                                "item_id": parts[0],
                                "category_id": parts[1],
                                "brand_id": parts[2],
                                "timestamp": parts[3],
                                "action_type": parts[4],
                            }
                        )
        except Exception as e:
            print(f"⚠️ 解析activity_log失败: {e}")

        return activities

    def _calculate_global_stats(self, interaction_data, y_train):
        """计算全局统计信息"""
        print("📊 计算全局统计信息...")

        if y_train is not None:
            # 基于标签计算全局复购率
            total_samples = len(y_train)
            repurchase_samples = sum(y_train == 1)
            global_repurchase_rate = repurchase_samples / total_samples if total_samples > 0 else 0.3
        else:
            # 基于交互数据推断复购率
            repurchase_count = sum(1 for inter in interaction_data if inter["purchase_count"] >= 2)
            global_repurchase_rate = repurchase_count / len(interaction_data) if interaction_data else 0.3

        # 用户和商户的全局统计
        all_users = set(inter["user_id"] for inter in interaction_data)
        all_merchants = set(inter["merchant_id"] for inter in interaction_data)

        self.global_stats = {
            "global_repurchase_rate": global_repurchase_rate,
            "global_user_repurchase_rate": global_repurchase_rate,  # 用户维度
            "global_merchant_repurchase_rate": global_repurchase_rate,  # 商户维度
            "total_users": len(all_users),
            "total_merchants": len(all_merchants),
            "total_interactions": len(interaction_data),
        }

        print(f"  ✅ 全局复购率: {global_repurchase_rate:.4f}")

    def _calculate_user_priors(self, interaction_data, y_train):
        """计算用户复购先验概率"""
        print("📊 计算用户复购先验概率...")

        user_stats = defaultdict(
            lambda: {
                "merchants_visited": set(),
                "total_purchases": 0,
                "repurchase_merchants": set(),
                "categories_purchased": set(),
                "brands_purchased": set(),
                "purchase_intervals": [],
                "avg_purchase_per_merchant": 0,
            }
        )

        # 统计每个用户的行为
        for inter in interaction_data:
            user_id = inter["user_id"]
            merchant_id = inter["merchant_id"]
            purchases = inter["purchases"]

            user_stats[user_id]["merchants_visited"].add(merchant_id)
            user_stats[user_id]["total_purchases"] += inter["purchase_count"]

            if inter["purchase_count"] >= 2:
                user_stats[user_id]["repurchase_merchants"].add(merchant_id)

            # 收集品类和品牌信息
            for purchase in purchases:
                user_stats[user_id]["categories_purchased"].add(purchase["category_id"])
                user_stats[user_id]["brands_purchased"].add(purchase["brand_id"])

        # 计算用户先验特征
        self.user_prior_stats = {}

        for user_id, stats in user_stats.items():
            merchants_count = len(stats["merchants_visited"])
            repurchase_merchants_count = len(stats["repurchase_merchants"])

            if merchants_count >= self.min_user_interactions:
                # 用户复购倾向 = 在多少比例的商户发生了复购
                user_repurchase_tendency = repurchase_merchants_count / merchants_count

                # 贝叶斯平滑
                smoothed_tendency = (
                    user_repurchase_tendency * merchants_count
                    + self.global_stats["global_user_repurchase_rate"] * self.pseudo_count
                ) / (merchants_count + self.pseudo_count)

                # 用户多样性特征
                category_diversity = len(stats["categories_purchased"])
                brand_diversity = len(stats["brands_purchased"])

                # 用户活跃度
                avg_purchases_per_merchant = stats["total_purchases"] / merchants_count

                self.user_prior_stats[user_id] = {
                    "user_repurchase_tendency": smoothed_tendency,
                    "user_merchant_diversity": merchants_count,
                    "user_category_diversity": category_diversity,
                    "user_brand_diversity": brand_diversity,
                    "user_avg_purchases_per_merchant": avg_purchases_per_merchant,
                    "user_total_purchases": stats["total_purchases"],
                    "user_sample_size": merchants_count,
                }

        print(f"  ✅ 计算了 {len(self.user_prior_stats)} 个用户的先验概率")

    def _calculate_merchant_priors(self, interaction_data, y_train):
        """计算商户被复购先验概率"""
        print("📊 计算商户被复购先验概率...")

        merchant_stats = defaultdict(
            lambda: {
                "users_served": set(),
                "total_purchases": 0,
                "repurchase_users": set(),
                "categories_offered": set(),
                "brands_offered": set(),
                "items_offered": set(),
                "user_purchase_counts": [],
            }
        )

        # 统计每个商户的行为
        for inter in interaction_data:
            user_id = inter["user_id"]
            merchant_id = inter["merchant_id"]
            purchases = inter["purchases"]

            merchant_stats[merchant_id]["users_served"].add(user_id)
            merchant_stats[merchant_id]["total_purchases"] += inter["purchase_count"]
            merchant_stats[merchant_id]["user_purchase_counts"].append(inter["purchase_count"])

            if inter["purchase_count"] >= 2:
                merchant_stats[merchant_id]["repurchase_users"].add(user_id)

            # 收集商户的产品组合信息
            for purchase in purchases:
                merchant_stats[merchant_id]["categories_offered"].add(purchase["category_id"])
                merchant_stats[merchant_id]["brands_offered"].add(purchase["brand_id"])
                merchant_stats[merchant_id]["items_offered"].add(purchase["item_id"])

        # 计算商户先验特征
        self.merchant_prior_stats = {}

        for merchant_id, stats in merchant_stats.items():
            users_count = len(stats["users_served"])
            repurchase_users_count = len(stats["repurchase_users"])

            if users_count >= self.min_merchant_interactions:
                # 商户被复购率 = 复购用户数 / 总用户数
                merchant_repurchase_rate = repurchase_users_count / users_count

                # 贝叶斯平滑
                smoothed_rate = (
                    merchant_repurchase_rate * users_count
                    + self.global_stats["global_merchant_repurchase_rate"] * self.pseudo_count
                ) / (users_count + self.pseudo_count)

                # 商户产品组合多样性
                category_diversity = len(stats["categories_offered"])
                brand_diversity = len(stats["brands_offered"])
                item_diversity = len(stats["items_offered"])

                # 商户专业化程度（分类越少越专业）
                specialization_score = 1 / (1 + category_diversity / 5)

                # 用户粘性（平均每用户购买次数）
                avg_purchases_per_user = stats["total_purchases"] / users_count

                # 购买集中度（基尼系数）
                purchase_counts = stats["user_purchase_counts"]
                purchase_concentration = self._calculate_gini_coefficient(purchase_counts)

                self.merchant_prior_stats[merchant_id] = {
                    "merchant_repurchase_rate": smoothed_rate,
                    "merchant_user_count": users_count,
                    "merchant_category_diversity": category_diversity,
                    "merchant_brand_diversity": brand_diversity,
                    "merchant_item_diversity": item_diversity,
                    "merchant_specialization_score": specialization_score,
                    "merchant_avg_purchases_per_user": avg_purchases_per_user,
                    "merchant_purchase_concentration": purchase_concentration,
                    "merchant_total_purchases": stats["total_purchases"],
                    "merchant_sample_size": users_count,
                }

        print(f"  ✅ 计算了 {len(self.merchant_prior_stats)} 个商户的先验概率")

    def _calculate_gini_coefficient(self, values):
        """计算基尼系数，衡量分布的不均匀程度"""
        if not values:
            return 0

        values = sorted(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

    def _generate_feature_dataframes(self, X_train):
        """生成特征DataFrame"""
        print("📊 生成特征DataFrame...")

        # 生成用户先验特征
        user_features_list = []
        all_users = X_train["user_id"].unique()

        for user_id in all_users:
            if user_id in self.user_prior_stats:
                stats = self.user_prior_stats[user_id]
                feature_row = {"user_id": user_id}
                for key, value in stats.items():
                    feature_row[f"user_prev_{key}"] = value
                user_features_list.append(feature_row)
            else:
                # 新用户或低频用户，使用全局平均值
                user_features_list.append(
                    {
                        "user_id": user_id,
                        "user_prev_user_repurchase_tendency": self.global_stats["global_user_repurchase_rate"],
                        "user_prev_user_merchant_diversity": 1,
                        "user_prev_user_category_diversity": 1,
                        "user_prev_user_brand_diversity": 1,
                        "user_prev_user_avg_purchases_per_merchant": 1.0,
                        "user_prev_user_total_purchases": 1,
                        "user_prev_user_sample_size": 1,
                    }
                )

        self.user_prior_features = pd.DataFrame(user_features_list)

        # 生成商户先验特征
        merchant_features_list = []
        all_merchants = X_train["merchant_id"].unique()

        for merchant_id in all_merchants:
            if merchant_id in self.merchant_prior_stats:
                stats = self.merchant_prior_stats[merchant_id]
                feature_row = {"merchant_id": merchant_id}
                for key, value in stats.items():
                    feature_row[f"merchant_prev_{key}"] = value
                merchant_features_list.append(feature_row)
            else:
                # 新商户或低频商户，使用全局平均值
                merchant_features_list.append(
                    {
                        "merchant_id": merchant_id,
                        "merchant_prev_merchant_repurchase_rate": self.global_stats["global_merchant_repurchase_rate"],
                        "merchant_prev_merchant_user_count": 1,
                        "merchant_prev_merchant_category_diversity": 1,
                        "merchant_prev_merchant_brand_diversity": 1,
                        "merchant_prev_merchant_item_diversity": 1,
                        "merchant_prev_merchant_specialization_score": 0.5,
                        "merchant_prev_merchant_avg_purchases_per_user": 1.0,
                        "merchant_prev_merchant_purchase_concentration": 0.5,
                        "merchant_prev_merchant_total_purchases": 1,
                        "merchant_prev_merchant_sample_size": 1,
                    }
                )

        self.merchant_prior_features = pd.DataFrame(merchant_features_list)

        print(f"  ✅ 用户特征: {self.user_prior_features.shape}")
        print(f"  ✅ 商户特征: {self.merchant_prior_features.shape}")

    def _cache_results(self):
        """缓存结果"""
        if self.enable_cache:
            print("💾 缓存先验特征...")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

            cached_data = {
                "user_prior_features": self.user_prior_features,
                "merchant_prior_features": self.merchant_prior_features,
                "global_stats": self.global_stats,
            }

            with open(self.cache_path, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"✅ 先验特征已缓存到: {self.cache_path}")

    def get_feature_names(self):
        """获取所有特征名称"""
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet.")

        feature_names = []

        if self.user_prior_features is not None:
            feature_names.extend([col for col in self.user_prior_features.columns if col != "user_id"])

        if self.merchant_prior_features is not None:
            feature_names.extend([col for col in self.merchant_prior_features.columns if col != "merchant_id"])

        return feature_names

    def get_stats_summary(self):
        """获取统计摘要"""
        if not self.is_fitted:
            return None

        summary = {
            "global_stats": self.global_stats,
            "user_prior_count": len(self.user_prior_stats) if self.user_prior_stats else 0,
            "merchant_prior_count": len(self.merchant_prior_stats) if self.merchant_prior_stats else 0,
        }

        if self.user_prior_features is not None:
            summary["user_features_shape"] = self.user_prior_features.shape

        if self.merchant_prior_features is not None:
            summary["merchant_features_shape"] = self.merchant_prior_features.shape

        return summary


if __name__ == "__main__":
    # 创建测试数据
    import numpy as np

    np.random.seed(42)
    n_samples = 1000

    # 模拟activity_log数据
    def generate_activity_log():
        activities = []
        n_activities = np.random.randint(1, 5)
        for _ in range(n_activities):
            item_id = f"item_{np.random.randint(1, 100)}"
            cat_id = f"cat_{np.random.randint(1, 10)}"
            brand_id = f"brand_{np.random.randint(1, 20)}"
            timestamp = f"2023{np.random.randint(1, 13):02d}{np.random.randint(1, 29):02d}"
            action_type = np.random.choice(["1", "2", "3"], p=[0.3, 0.5, 0.2])  # 1=购买
            activities.append(f"{item_id}:{cat_id}:{brand_id}:{timestamp}:{action_type}")
        return "#".join(activities)

    test_data = {
        "user_id": np.random.randint(1, 200, n_samples),
        "merchant_id": np.random.randint(1, 50, n_samples),
        "age_range": np.random.randint(1, 6, n_samples),
        "gender": np.random.randint(1, 3, n_samples),
        "activity_log": [generate_activity_log() for _ in range(n_samples)],
    }

    df = pd.DataFrame(test_data)
    X = df
    y = pd.Series(np.random.randint(0, 2, n_samples))  # 模拟标签

    print("🧪 使用模拟数据进行测试...")
    print(f"模拟数据形状: {X.shape}")

    transformer = PrevTransformer(min_user_interactions=2, min_merchant_interactions=3, enable_cache=False)

    transformer.fit(X, y)
    transformed_df = transformer.transform(X)

    print(f"转换后的数据形状: {transformed_df.shape}")
    print(f"特征名称: {transformer.get_feature_names()}")
    print(f"统计摘要: {transformer.get_stats_summary()}")
    print(f"转换后的数据预览:\n{transformed_df.head()}")
