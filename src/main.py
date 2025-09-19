from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .data import load_data, load_dataframe


class TCDataConfig(BaseModel):
    fill_median_cols: list[str] = []
    fill_mode_cols: list[str] = []


class UserFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = None
        return

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy = X_copy[X_copy["activity_log"].notnull()]
        X_copy = X_copy.assign(
            activity_log=X_copy["activity_log"].str.split("#")
        ).explode("activity_log")

        X_copy[["item_id", "cate_id", "brand_id", "time", "action_type"]] = X_copy[
            "activity_log"
        ].str.split(":", expand=True)

        # 每个用户不同action类型的行为占比
        action_ratio = (
            X_copy.groupby(["user_id", "action_type"])
            .size()
            .div(X_copy.groupby("user_id").size(), level="user_id")
            .unstack(fill_value=0)
            .add_prefix("action_ratio_")
        )

        # 计算每个用户每个月的行为占比
        X_copy["month"] = X_copy["time"].str[:2].astype(int)
        time_ratio = (
            X_copy.groupby(["user_id", "month"])
            .size()
            .div(X_copy.groupby("user_id").size(), level="user_id")
            .unstack(fill_value=0)
            .add_prefix("time_ratio_")
        )

        # 计算每个用户每个月不同action的占比。
        time_action_ratio = (
            X_copy.pivot_table(
                index=["user_id", "month"],
                columns="action_type",
                aggfunc="size",
                fill_value=0,
            )
            .div(X_copy.groupby(["user_id", "month"]).size(), axis=0)
            .add_prefix("time_action_ratio_")
        )

        # 合并所有特征
        features = action_ratio.join(time_ratio, how="outer").join(
            time_action_ratio, how="outer"
        )

        self.features = features.reset_index()
        return self

    def transform(self, X):
        # 假设 X 是一个包含用户信息的 DataFrame
        # 这里可以添加更多的特征工程步骤
        X_transformed = X.copy()
        X_transformed["month"] = X_transformed["time"].str[:2].astype(int)
        X_transformed = X_transformed.join(self.features, how="outer")
        return X_transformed


class MerchantFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 假设 X 是一个包含商户信息的 DataFrame
        # 这里可以添加更多的特征工程步骤
        X_transformed = X.copy()
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
    ]

    return Pipeline(steps, verbose=True)


def create_sample_pipeline(conf: TCDataConfig) -> Pipeline:
    """Create a data sampling pipeline."""
    pass


def create_model_pipeline(conf: TCDataConfig) -> Pipeline:
    """Create a model training pipeline."""
    pass


class TCDataPipeline:
    def __init__(self, conf: TCDataConfig):
        self.clean_pipe = create_clean_pipeline(conf)
        self.sample_pipe = create_sample_pipeline(conf)
        self.model_pipe = create_model_pipeline(conf)

    def fit(self, X, y):
        X, y = self.clean_pipe.fit(X, y)

        pass

    def transform(self, X):
        pass


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

    X = pipe.fit_transform(X, y)
    print(X.head(10))


if __name__ == "__main__":
    run()
