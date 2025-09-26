import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature.behavior_transformer import BehaviorTransformer
from feature.matrix_transformer import MatrixTransformer
from feature.prev_transformer import PrevTransformer
from feature.tfidf_transformer import TfidfTransformer

try:
    from .data import load_data, load_dataframe
except ImportError:
    from data import load_data, load_dataframe


class ModelConfig(BaseModel):
    model_type: str = "rf"  # 可选 'rf', 'xgb', 'lgb'
    n_estimators: int = 200
    max_depth: int = 10
    learning_rate: float = 0.1
    scale_pos_weight: float = 7.0  # 用于处理类别不平衡


class TCDataConfig(BaseModel):
    fill_median_cols: list[str] = []
    fill_mode_cols: list[str] = []

    cache_behavior_feature: bool = True
    cache_behavior_path: str = "../data/behavior_feature.pkl"

    cache_tfidf_feature: bool = True
    cache_tfidf_path: str = "../data/tfidf_feature.pkl"

    cache_matrix_feature: bool = True
    cache_matrix_path: str = "../data/matrix_feature.pkl"

    cache_prev_feature: bool = True
    cache_prev_path: str = "../data/prev_feature.pkl"
    model: ModelConfig = ModelConfig()


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

        self.behavior_transformer = BehaviorTransformer(
            enable_cache=conf.cache_behavior_feature,
            cache_path=conf.cache_behavior_path,
        )

        self.tfidf_transformer = TfidfTransformer(
            enable_cache=conf.cache_tfidf_feature,
            cache_path=conf.cache_tfidf_path,
        )

        self.matrix_transformer = MatrixTransformer(
            n_components=80, random_state=42, enable_cache=conf.cache_matrix_feature, cache_path=conf.cache_matrix_path
        )
        self.prev_transformer = PrevTransformer(enable_cache=conf.cache_prev_feature, cache_path=conf.cache_prev_path)
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

    def preprocess(self, X: pd.DataFrame, y: pd.Series = None):
        """
        仅进行数据清洗和特征工程
        """
        self.behavior_transformer.fit(X, y)
        self.tfidf_transformer.fit(X, y)
        self.matrix_transformer.fit(X, y)
        self.prev_transformer.fit(X, y)

        train_mask = y.isin([0, 1])
        test_mask = y.isnull()

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        X_train = self.behavior_transformer.transform(X_train)
        X_test = self.behavior_transformer.transform(X_test)

        X_train = self.tfidf_transformer.transform(X_train)
        X_test = self.tfidf_transformer.transform(X_test)

        X_train = self.matrix_transformer.transform(X_train)
        X_test = self.matrix_transformer.transform(X_test)

        X_train = self.prev_transformer.transform(X_train)
        X_test = self.prev_transformer.transform(X_test)

        print("特征处理完成, 当前数据形状:")
        print(f"训练集: {X_train.shape},  测试集: {X_test.shape}")
        drop_cols = ["user_id", "merchant_id", "activity_log"]
        X_train = X_train.drop(columns=[col for col in drop_cols if col in X_train.columns], errors="ignore")
        # X_test = X_test.drop(columns=[col for col in drop_cols if col in X_test.columns], errors="ignore")

        return X_train, y_train, X_test, y_test

    def fit(self, X, y):
        """
        完整的训练流程
        """
        train_X, train_y, test_X, test_y = self.preprocess(X, y)
        self.summary(train_X, None, test_X)

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
                    "classifier__n_estimators": [
                        # 900,
                        # 1000,
                        1100,
                        # 1200,
                    ],
                    "classifier__max_depth": [3],
                    "classifier__learning_rate": [0.05],
                    "classifier__class_weight": [None],
                }

        search_cls = GridSearchCV if search_type == "grid" else RandomizedSearchCV

        if search_type == "grid":
            search = search_cls(
                self.model_pipe,
                param_grid,
                cv=cv,
                scoring=scoring,
                verbose=2,
                n_jobs=12,
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

    def export_prediction(self, X: pd.DataFrame, y, filename="prediction.csv"):
        """
        导出预测结果到 prediction.csv
        格式：user_id, merchant_id, prob
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法导出预测结果。")

        _, _, X_test, _ = self.preprocess(X, y)

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
    df = load_dataframe(nrow=None)
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
            # cache_behavior_feature=True,
            # cache_behavior_path="../data/cleaned_data.pkl",
            model=ModelConfig(model_type="lgb", n_estimators=500, max_depth=4, scale_pos_weight=7.0),
        )
    )

    pipe.fit(X, y)
    # pipe.tune_model(X, y)

    if pipe.is_fitted:
        pipe.export_prediction(X, y, filename="../output/prediction.csv")


if __name__ == "__main__":
    run()
