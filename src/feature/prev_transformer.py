import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PrevTransformer(BaseEstimator, TransformerMixin):
    """
    为每个商户/cat/brand/item增加标签0/1比例编码特征
    """

    def fit(self, X, y=None):
        df = X.copy()
        if y is not None:
            df["label"] = y.values
        else:
            raise ValueError("y不能为空")

        # 计算比例特征
        self.merchant_label_ratio = df.groupby("merchant_id")["label"].mean().to_dict()
        self.cat_label_ratio = df.groupby("cate_id")["label"].mean().to_dict()
        self.brand_label_ratio = df.groupby("brand_id")["label"].mean().to_dict()
        self.item_label_ratio = df.groupby("item_id")["label"].mean().to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        df["merchant_label_ratio"] = df["merchant_id"].map(self.merchant_label_ratio)
        df["cat_label_ratio"] = df["cate_id"].map(self.cat_label_ratio)
        df["brand_label_ratio"] = df["brand_id"].map(self.brand_label_ratio)
        df["item_label_ratio"] = df["item_id"].map(self.item_label_ratio)
        # 缺失填充为0.0
        for col in ["merchant_label_ratio", "cat_label_ratio", "brand_label_ratio", "item_label_ratio"]:
            df[col] = df[col].fillna(0.0)
        return df

if __name__ == "__main__":
    # 简单测试
    import numpy as np
    n_samples = 1000
    test_data = {
        "merchant_id": np.random.randint(1, 50, n_samples),
        "cate_id": np.random.randint(1, 20, n_samples),
        "brand_id": np.random.randint(1, 30, n_samples),
        "item_id": np.random.randint(1, 500, n_samples),
    }
    X = pd.DataFrame(test_data)
    y = np.random.randint(0, 2, n_samples)

    transformer = PrevTransformer()
    transformer.fit(X, y)
    transformed_df = transformer.transform(X)
    print(transformed_df.head())