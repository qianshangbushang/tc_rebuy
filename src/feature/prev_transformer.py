import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PrevTransformer(BaseEstimator, TransformerMixin):
    """先验概率特征转换器，生成：
    1) 用户复购先验特征
    2) 商户被复购先验特征
    3) 性别复购先验特征（平滑）
    4) 年龄段复购先验特征（平滑）
    """

    def __init__(
        self,
        min_user_interactions: int = 1,
        min_merchant_interactions: int = 1,
        time_window_days: int = 365,
        repurchase_window_days: int = 90,
        enable_cache: bool = True,
        cache_path: str = "../output/prev_feature.pkl",
        pseudo_count: float = 1.0,
    ):
        self.min_user_interactions = min_user_interactions
        self.min_merchant_interactions = min_merchant_interactions
        self.time_window_days = time_window_days
        self.repurchase_window_days = repurchase_window_days
        self.enable_cache = enable_cache
        self.cache_path = cache_path
        self.pseudo_count = pseudo_count

        # 先验概率统计
        self.user_prior_stats = {}
        self.merchant_prior_stats = {}
        self.global_stats = {}

        # 特征 DataFrame
        self.user_prior_features = None
        self.merchant_prior_features = None
        self.gender_prior_features = None
        self.age_prior_features = None

        # 映射（用于无缓存合并或缺失填充）
        self.gender_prior_map = {}
        self.age_prior_map = {}

        self.is_fitted = False

    def fit(self, X, y=None):
        """训练先验概率特征"""
        if self.enable_cache and os.path.exists(self.cache_path):
            print(f"♻️ 从缓存加载先验特征: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                cached_data = pickle.load(f)

            self.user_prior_features = cached_data.get("user_prior_features")
            self.merchant_prior_features = cached_data.get("merchant_prior_features")
            self.global_stats = cached_data.get("global_stats", {})
            self.gender_prior_features = cached_data.get("gender_prior_features")
            self.age_prior_features = cached_data.get("age_prior_features")
            self.gender_prior_map = cached_data.get("gender_prior_map", {})
            self.age_prior_map = cached_data.get("age_prior_map", {})

            print("✅ 先验特征加载完成")
            self.is_fitted = True
            return self

        print("🔄 开始训练先验概率特征...")

        # 1. 仅使用有标签样本
        if y is not None:
            train_mask = y.isin([0, 1])
            X_train = X[train_mask].copy()
            y_train = y[train_mask].copy()
            print(f"📊 训练数据: {len(X_train)} 样本")
        else:
            X_train = X.copy()
            y_train = None
            print("⚠️ 未提供标签，使用全部数据（性别/年龄先验将无法计算）")

        # 2. 解析交互
        interaction_data = self._parse_interactions(X_train)
        print(f"📊 解析出 {len(interaction_data)} 条交互记录")

        # 3. 全局统计
        self._calculate_global_stats(interaction_data, y_train)

        # 4. 用户先验
        self._calculate_user_priors(interaction_data, y_train)

        # 5. 商户先验
        self._calculate_merchant_priors(interaction_data, y_train)

        # 6. 生成用户/商户特征
        self._generate_feature_dataframes(X_train)

        # 7. 性别 / 年龄先验
        self._calculate_demo_priors(X_train, y_train)

        # 8. 缓存
        self._cache_results()

        self.is_fitted = True
        print("✅ 先验概率特征训练完成")
        return self

    def transform(self, X):
        """转换数据加入先验概率特征"""
        if not self.is_fitted:
            raise ValueError("PrevTransformer has not been fitted yet.")

        print("🔄 开始先验概率特征转换...")
        result = X.copy()

        # 用户特征
        if self.user_prior_features is not None:
            result = result.merge(self.user_prior_features, on="user_id", how="left")
            print(f"  ✅ 关联用户先验特征: {self.user_prior_features.shape[1] - 1} 维")

        # 商户特征
        if self.merchant_prior_features is not None:
            result = result.merge(self.merchant_prior_features, on="merchant_id", how="left")
            print(f"  ✅ 关联商户先验特征: {self.merchant_prior_features.shape[1] - 1} 维")

        # 性别先验
        if "gender" in result.columns:
            if self.gender_prior_features is not None:
                result = result.merge(self.gender_prior_features, on="gender", how="left")
            else:
                result["gender_prev_repurchase_rate"] = result["gender"].map(self.gender_prior_map)

        # 年龄先验
        if "age_range" in result.columns:
            if self.age_prior_features is not None:
                result = result.merge(self.age_prior_features, on="age_range", how="left")
            else:
                result["age_prev_repurchase_rate"] = result["age_range"].map(self.age_prior_map)

        # 缺失填充
        global_user = self.global_stats.get("global_user_repurchase_rate", 0.3)
        global_merchant = self.global_stats.get("global_merchant_repurchase_rate", 0.3)
        global_rate = self.global_stats.get("global_repurchase_rate", 0.3)

        for col in result.columns:
            if col.startswith("user_prev_"):
                result[col] = result[col].fillna(global_user)
            elif col.startswith("merchant_prev_"):
                result[col] = result[col].fillna(global_merchant)

        if "gender_prev_repurchase_rate" in result.columns:
            result["gender_prev_repurchase_rate"] = result["gender_prev_repurchase_rate"].fillna(global_rate)
        if "age_prev_repurchase_rate" in result.columns:
            result["age_prev_repurchase_rate"] = result["age_prev_repurchase_rate"].fillna(global_rate)

        print(f"✅ 先验概率特征转换完成，新增维度: {result.shape[1] - X.shape[1]}")
        return result

    # ---------------- 内部计算方法 ---------------- #

    def _parse_interactions(self, X):
        """解析activity_log"""
        print("📊 解析用户-商户交互数据...")
        interactions = []
        for _, row in X.iterrows():
            user_id = row["user_id"]
            merchant_id = row["merchant_id"]
            log = row.get("activity_log")
            if pd.isna(log):
                continue
            activities = self._parse_activity_log(log)
            # 约定: action_type == "2" 为购买（按当前文件逻辑保留）
            purchases = [a for a in activities if a.get("action_type") == "2"]
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

    def _parse_activity_log(self, activity_log: str):
        activities = []
        if pd.isna(activity_log):
            return activities
        try:
            for record in activity_log.split("#"):
                if not record.strip():
                    continue
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
        print("📊 计算全局统计信息...")
        if y_train is not None:
            total = len(y_train)
            pos = int((y_train == 1).sum())
            global_repurchase_rate = pos / total if total else 0.3
        else:
            repurchase = sum(1 for inter in interaction_data if inter["purchase_count"] >= 2)
            global_repurchase_rate = repurchase / len(interaction_data) if interaction_data else 0.3

        users = {i["user_id"] for i in interaction_data}
        merchants = {i["merchant_id"] for i in interaction_data}

        self.global_stats = {
            "global_repurchase_rate": global_repurchase_rate,
            "global_user_repurchase_rate": global_repurchase_rate,
            "global_merchant_repurchase_rate": global_repurchase_rate,
            "total_users": len(users),
            "total_merchants": len(merchants),
            "total_interactions": len(interaction_data),
        }
        print(f"  ✅ 全局复购率: {global_repurchase_rate:.4f}")

    def _calculate_user_priors(self, interaction_data, y_train):
        print("📊 计算用户复购先验概率...")
        user_stats = defaultdict(
            lambda: {
                "merchants_visited": set(),
                "total_purchases": 0,
                "repurchase_merchants": set(),
                "categories_purchased": set(),
                "brands_purchased": set(),
            }
        )
        for inter in interaction_data:
            uid = inter["user_id"]
            mid = inter["merchant_id"]
            user_stats[uid]["merchants_visited"].add(mid)
            user_stats[uid]["total_purchases"] += inter["purchase_count"]
            if inter["purchase_count"] >= 2:
                user_stats[uid]["repurchase_merchants"].add(mid)
            for p in inter["purchases"]:
                user_stats[uid]["categories_purchased"].add(p["category_id"])
                user_stats[uid]["brands_purchased"].add(p["brand_id"])

        self.user_prior_stats = {}
        for uid, stats in user_stats.items():
            m_cnt = len(stats["merchants_visited"])
            if m_cnt >= self.min_user_interactions:
                rep_cnt = len(stats["repurchase_merchants"])
                tendency = rep_cnt / m_cnt if m_cnt else 0
                smoothed = (tendency * m_cnt + self.global_stats["global_user_repurchase_rate"] * self.pseudo_count) / (
                    m_cnt + self.pseudo_count
                )
                self.user_prior_stats[uid] = {
                    "user_repurchase_tendency": smoothed,
                    "user_merchant_diversity": m_cnt,
                    "user_category_diversity": len(stats["categories_purchased"]),
                    "user_brand_diversity": len(stats["brands_purchased"]),
                    "user_avg_purchases_per_merchant": stats["total_purchases"] / m_cnt,
                    "user_total_purchases": stats["total_purchases"],
                    "user_sample_size": m_cnt,
                }
        print(f"  ✅ 计算了 {len(self.user_prior_stats)} 个用户的先验概率")

    def _calculate_merchant_priors(self, interaction_data, y_train):
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
        for inter in interaction_data:
            uid = inter["user_id"]
            mid = inter["merchant_id"]
            merchant_stats[mid]["users_served"].add(uid)
            merchant_stats[mid]["total_purchases"] += inter["purchase_count"]
            merchant_stats[mid]["user_purchase_counts"].append(inter["purchase_count"])
            if inter["purchase_count"] >= 2:
                merchant_stats[mid]["repurchase_users"].add(uid)
            for p in inter["purchases"]:
                merchant_stats[mid]["categories_offered"].add(p["category_id"])
                merchant_stats[mid]["brands_offered"].add(p["brand_id"])
                merchant_stats[mid]["items_offered"].add(p["item_id"])

        self.merchant_prior_stats = {}
        for mid, stats in merchant_stats.items():
            u_cnt = len(stats["users_served"])
            if u_cnt >= self.min_merchant_interactions:
                rep_u = len(stats["repurchase_users"])
                raw_rate = rep_u / u_cnt if u_cnt else 0
                smoothed = (
                    raw_rate * u_cnt + self.global_stats["global_merchant_repurchase_rate"] * self.pseudo_count
                ) / (u_cnt + self.pseudo_count)

                cat_div = len(stats["categories_offered"])
                brand_div = len(stats["brands_offered"])
                item_div = len(stats["items_offered"])
                specialization = 1 / (1 + cat_div / 5)
                avg_purchase_user = stats["total_purchases"] / u_cnt
                gini = self._calculate_gini_coefficient(stats["user_purchase_counts"])

                self.merchant_prior_stats[mid] = {
                    "merchant_repurchase_rate": smoothed,
                    "merchant_user_count": u_cnt,
                    "merchant_category_diversity": cat_div,
                    "merchant_brand_diversity": brand_div,
                    "merchant_item_diversity": item_div,
                    "merchant_specialization_score": specialization,
                    "merchant_avg_purchases_per_user": avg_purchase_user,
                    "merchant_purchase_concentration": gini,
                    "merchant_total_purchases": stats["total_purchases"],
                    "merchant_sample_size": u_cnt,
                }
        print(f"  ✅ 计算了 {len(self.merchant_prior_stats)} 个商户的先验概率")

    def _calculate_gini_coefficient(self, values):
        if not values:
            return 0
        values = sorted(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

    def _generate_feature_dataframes(self, X_train):
        print("📊 生成用户/商户特征 DataFrame...")
        # 用户
        user_rows = []
        for uid in X_train["user_id"].unique():
            if uid in self.user_prior_stats:
                stats = self.user_prior_stats[uid]
                row = {"user_id": uid}
                for k, v in stats.items():
                    row[f"user_prev_{k}"] = v
                user_rows.append(row)
            else:
                # 冷启动用户
                user_rows.append(
                    {
                        "user_id": uid,
                        "user_prev_user_repurchase_tendency": self.global_stats.get("global_user_repurchase_rate", 0.3),
                        "user_prev_user_merchant_diversity": 1,
                        "user_prev_user_category_diversity": 1,
                        "user_prev_user_brand_diversity": 1,
                        "user_prev_user_avg_purchases_per_merchant": 1.0,
                        "user_prev_user_total_purchases": 1,
                        "user_prev_user_sample_size": 1,
                    }
                )
        self.user_prior_features = pd.DataFrame(user_rows)

        # 商户
        merchant_rows = []
        for mid in X_train["merchant_id"].unique():
            if mid in self.merchant_prior_stats:
                stats = self.merchant_prior_stats[mid]
                row = {"merchant_id": mid}
                for k, v in stats.items():
                    row[f"merchant_prev_{k}"] = v
                merchant_rows.append(row)
            else:
                merchant_rows.append(
                    {
                        "merchant_id": mid,
                        "merchant_prev_merchant_repurchase_rate": self.global_stats.get(
                            "global_merchant_repurchase_rate", 0.3
                        ),
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
        self.merchant_prior_features = pd.DataFrame(merchant_rows)
        print(f"  ✅ 用户特征: {self.user_prior_features.shape} | 商户特征: {self.merchant_prior_features.shape}")

    def _calculate_demo_priors(self, X_train, y_train):
        """性别与年龄段先验（需要标签）"""
        if y_train is None:
            print("⚠️ 无标签，跳过性别/年龄先验计算")
            return
        print("📊 计算性别 / 年龄先验概率...")
        df = X_train[["gender", "age_range"]].copy()
        df["label"] = y_train.values
        global_rate = self.global_stats.get("global_repurchase_rate", 0.3)

        def _calc(group):
            cnt = len(group)
            pos = group["label"].sum()
            raw = pos / cnt if cnt else global_rate
            smooth = (pos + self.pseudo_count * global_rate) / (cnt + self.pseudo_count)
            return cnt, pos, raw, smooth

        gender_rows = []
        for g, gdf in df.groupby("gender"):
            cnt, pos, raw, smoothed = _calc(gdf)
            gender_rows.append(
                {
                    "gender": g,
                    "gender_prev_repurchase_rate": smoothed,
                    "gender_prev_raw_rate": raw,
                    "gender_prev_count": cnt,
                }
            )
            self.gender_prior_map[g] = smoothed
        self.gender_prior_features = pd.DataFrame(gender_rows)

        age_rows = []
        for a, adf in df.groupby("age_range"):
            cnt, pos, raw, smoothed = _calc(adf)
            age_rows.append(
                {
                    "age_range": a,
                    "age_prev_repurchase_rate": smoothed,
                    "age_prev_raw_rate": raw,
                    "age_prev_count": cnt,
                }
            )
            self.age_prior_map[a] = smoothed
        self.age_prior_features = pd.DataFrame(age_rows)
        print(f"  ✅ 性别先验: {len(self.gender_prior_features)} | 年龄先验: {len(self.age_prior_features)}")

    def _cache_results(self):
        if self.enable_cache:
            print("💾 缓存先验特征...")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            cached_data = {
                "user_prior_features": self.user_prior_features,
                "merchant_prior_features": self.merchant_prior_features,
                "global_stats": self.global_stats,
                "gender_prior_features": self.gender_prior_features,
                "age_prior_features": self.age_prior_features,
                "gender_prior_map": self.gender_prior_map,
                "age_prior_map": self.age_prior_map,
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"✅ 先验特征已缓存到: {self.cache_path}")

    def get_feature_names(self):
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted yet.")
        names = []
        if self.user_prior_features is not None:
            names.extend([c for c in self.user_prior_features.columns if c != "user_id"])
        if self.merchant_prior_features is not None:
            names.extend([c for c in self.merchant_prior_features.columns if c != "merchant_id"])
        if self.gender_prior_features is not None:
            names.extend([c for c in self.gender_prior_features.columns if c != "gender"])
        if self.age_prior_features is not None:
            names.extend([c for c in self.age_prior_features.columns if c != "age_range"])
        return names

    def get_stats_summary(self):
        if not self.is_fitted:
            return None
        summary = {
            "global_stats": self.global_stats,
            "user_prior_count": len(self.user_prior_stats),
            "merchant_prior_count": len(self.merchant_prior_stats),
        }
        if self.gender_prior_features is not None:
            summary["gender_prior_shape"] = self.gender_prior_features.shape
        if self.age_prior_features is not None:
            summary["age_prior_shape"] = self.age_prior_features.shape
        return summary


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 500

    def gen_log():
        acts = []
        for _ in range(np.random.randint(1, 6)):
            item = f"item_{np.random.randint(1, 50)}"
            cat = f"cat_{np.random.randint(1, 8)}"
            brand = f"brand_{np.random.randint(1, 15)}"
            ts = f"2023{np.random.randint(1, 13):02d}{np.random.randint(1, 29):02d}"
            act = np.random.choice(["1", "2", "3"], p=[0.25, 0.55, 0.20])
            acts.append(f"{item}:{cat}:{brand}:{ts}:{act}")
        return "#".join(acts)

    df = pd.DataFrame(
        {
            "user_id": np.random.randint(1, 180, n_samples),
            "merchant_id": np.random.randint(1, 60, n_samples),
            "age_range": np.random.randint(1, 7, n_samples),
            "gender": np.random.randint(1, 3, n_samples),
            "activity_log": [gen_log() for _ in range(n_samples)],
        }
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))

    tr = PrevTransformer(enable_cache=False, min_user_interactions=2, min_merchant_interactions=3)
    tr.fit(df, y)
    out = tr.transform(df)
    print(out.filter(regex="prev_").head())
    print("Feature count:", len(tr.get_feature_names()))
    print("Summary:", tr.get_stats_summary())
