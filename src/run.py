"""重构后的模型入口"""

import gc
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import RebuyDataset_v1, build_dataloaders, split_train_val_test
from feature_engineering import (
    build_merchant_features,
    build_sequence_features,
    build_user_features,
    create_global_labelencoder,
)
from rebuy_model import RebuyModel, evaluate, predict, train


def create_explode_dataframe(df: pd.DataFrame):
    """加载并展开数据集"""
    df_explode = df.copy()
    df_explode["log_list"] = df_explode["activity_log"].str.split("#")
    df_explode = df_explode.explode("log_list")
    df_explode[["item_id", "cate_id", "brand_id", "time_str", "action_type"]] = df_explode["log_list"].str.split(
        ":", expand=True
    )
    return df_explode


def load_exploded_dataframe(nrow=None, special_sample_frac=0.1):
    """加载并展开数据集"""
    df_train = pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=nrow)
    df_test = pd.read_csv("./data/format2/data_format2/test_format2.csv", nrows=nrow)
    df = pd.concat([df_train, df_test])
    df_explode = df.copy()
    df_explode["log_list"] = df_explode["activity_log"].str.split("#")
    df_explode = df_explode.explode("log_list")
    df_explode[["item_id", "cate_id", "brand_id", "time_str", "action_type"]] = df_explode["log_list"].str.split(
        ":", expand=True
    )

    # special_mask = df_explode["time_str"].isin(["0618", "1111"])
    # special_df = df_explode[special_mask]
    # normal_df = df_explode[~special_mask]

    # # 对特殊日期采样（如只保留10%）
    # sampled_special_df = special_df.sample(frac=0.1, random_state=42)

    # # 合并采样后的数据
    # df_explode = pd.concat([normal_df, sampled_special_df], ignore_index=True)

    return df_explode.drop(columns=["log_list"])


def run(mode="dev"):
    """主流程"""
    # 配置
    config = {
        "seq_len": 30,
        "batch_size": 128,
        "emb_dim": 32,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 1e-3,
        "epochs": 5,
        "early_stop_rounds": 3,
        "val_ratio": 0.2,
        "enable_cache": True,
        "cache_dir": "./data/cache",
        "model_dir": "./data/models",
        "pred_path": "./data/output/test_pred.csv",
    }

    # 创建必要的目录
    os.makedirs(config["cache_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["pred_path"]), exist_ok=True)

    nrows = 10000 if mode == "dev" else None

    # 读取数据
    train_df = pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=nrows)
    test_df = pd.read_csv("./data/format2/data_format2/test_format2.csv", nrows=nrows)
    df = pd.concat([train_df, test_df])
    df_explode = create_explode_dataframe(df)
    del df, train_df, test_df  # 释放内存
    gc.collect()

    label_encoder = create_global_labelencoder(
        df_explode,
        columns=["item_id", "cate_id", "brand_id", "merchant_id", "action_type", "age_range", "gender"],
        cache_dir=config["cache_dir"],
    )

    # 1. 特征工程
    print("\n1. 构建特征")
    # 用户特征
    user_features, user_feature_dims = build_user_features(
        # df_explode,
        df_explode,
        global_le_encoder=label_encoder,
        cache_dir=config["cache_dir"],
    )

    # 商户特征
    merchant_features, merchant_feature_dims = build_merchant_features(
        df_explode,
        # create_explode_dataframe(df),
        cache_dir=config["cache_dir"],
        global_le_encoder=label_encoder,
    )

    # 序列特征
    sequence_features, encoders = build_sequence_features(
        df_explode,
        # create_explode_dataframe(df),
        user_features,
        merchant_features,
        user_feature_dims=user_feature_dims,
        merchant_feature_dims=merchant_feature_dims,
        seq_len=config["seq_len"],
        cache_dir=config["cache_dir"],
        global_le_encoder=label_encoder,
    )

    del df_explode
    gc.collect()
    # 2. 数据集划分
    print("\n2. 划分数据集")
    train_features, val_features, test_features = split_train_val_test(
        sequence_features,
        val_ratio=config["val_ratio"],
    )

    # 3. 构建数据加载器
    print("\n3. 构建数据加载器")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_features,
        val_features,
        test_features,
        batch_size=config["batch_size"],
    )

    # 4. 构建模型
    print("\n4. 构建模型")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 计算各种特征维度
    # total_items = sum(len(enc.classes_) for enc in encoders.values())
    # user_feat_dim = len(user_features.columns) - 1  # 减去user_id
    merchant_feat_dim = len(merchant_features.columns) - 1  # 减去merchant_id

    model = RebuyModel(
        total_items=len(label_encoder.classes_),
        user_num_feat_dim=user_feature_dims.get("num_feats", 0),
        user_cat_feat_dim=user_feature_dims.get("cat_feats", 0),
        merchant_feat_dim=merchant_feat_dim,
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    # 5. 训练
    print("\n5. 训练模型")
    train(
        train_loader,
        val_loader,
        model,
        epochs=config["epochs"],
        lr=config["lr"],
        early_stop_rounds=config["early_stop_rounds"],
        model_dir=config["model_dir"],
        device=device,
    )

    # 6. 加载最佳模型并评估
    print("\n6. 模型评估")
    model.load_state_dict(torch.load(f"{config['model_dir']}/best_model.pth"))

    print("\n训练集评估:")
    evaluate(model, train_loader, device=device)

    print("\n验证集评估:")
    evaluate(model, val_loader, device=device)

    # print("\n测试集评估:")
    # evaluate(model, test_loader, device=device)

    # 7. 预测并保存结果
    print("\n7. 预测并保存结果")
    predict(model, test_loader, save_path=config["pred_path"], device=device)


def test_build_merchant_features():
    from feature_engineering import build_merchant_features

    cache_dir = "./data/cache_test"
    df_explode = load_exploded_dataframe(nrow=1000)

    label_encoder = create_global_labelencoder(
        df_explode,
        columns=["item_id", "cate_id", "brand_id", "merchant_id", "action_type", "age_range", "gender"],
        cache_dir=cache_dir,
    )
    merchant_features, feature_dims = build_merchant_features(
        df_explode,
        cache_dir=cache_dir,
        global_le_encoder=label_encoder,
    )

    print("Merchant Features:")
    print(merchant_features.head())
    print(merchant_features.shape)
    print("\nFeature Dimensions:")
    print(feature_dims)


def test_build_sequence_features():
    from feature_engineering import build_merchant_features, build_sequence_features, build_user_features

    cache_dir = "./data/cache_test"
    df_explode = load_exploded_dataframe(nrow=100000)
    label_encoder = create_global_labelencoder(
        df_explode,
        columns=["item_id", "cate_id", "brand_id", "merchant_id", "action_type", "age_range", "gender"],
        cache_dir=cache_dir,
    )
    # df = load_dataframe(nrow=1000)
    user_features, user_feature_dims = build_user_features(
        df_explode, cache_dir=cache_dir, global_le_encoder=label_encoder
    )
    merchant_features, merchant_feature_dims = build_merchant_features(
        df_explode, cache_dir=cache_dir, global_le_encoder=label_encoder
    )
    sequence_features, _ = build_sequence_features(
        df_explode,
        user_features,
        merchant_features,
        user_feature_dims=user_feature_dims,
        merchant_feature_dims=merchant_feature_dims,
        global_le_encoder=label_encoder,
        seq_len=30,
        cache_dir=cache_dir,
    )
    for k, v in sequence_features.items():
        print(
            k,
            len(v),
        )
        print(v[:10])


def test_build_user_features():
    from feature_engineering import build_user_features

    cache_dir = "./data/cache_test"
    # df = pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=1000)
    df_explode = load_exploded_dataframe(nrow=100000)
    label_encoder = create_global_labelencoder(
        df_explode,
        columns=["item_id", "cate_id", "brand_id", "merchant_id", "action_type", "age_range", "gender"],
        cache_dir=cache_dir,
    )
    user_features, feature_dims = build_user_features(
        df_explode,
        cache_dir=cache_dir,
        global_le_encoder=label_encoder,
    )
    print("User Features:")
    print(user_features.head())
    print(user_features.shape)
    print(user_features.dtypes)
    print("\nFeature Dimensions:")
    print(feature_dims)


def predict_with_best_model(
    user_feature_dims={},
    merchant_feature_dims={},
    seq_len=30,
    batch_size=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    加载最佳模型，对新数据进行预测，并可保存结果
    """
    df = load_exploded_dataframe(nrow=None)
    cache_dir = "./data/cache"
    label_encoder = create_global_labelencoder(
        df,
        columns=["item_id", "cate_id", "brand_id", "merchant_id", "action_type", "age_range", "gender"],
        cache_dir=cache_dir,
    )
    # 构建序列特征
    sequence_features, _ = build_sequence_features(df, seq_len=30)

    # 2. 数据集划分
    print("\n2. 划分数据集")
    train_features, val_features, test_features = split_train_val_test(
        sequence_features,
        val_ratio=0.2,
    )
    # 构建 DataLoader
    test_loader = DataLoader(
        dataset=RebuyDataset_v1(test_features),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # 加载模型
    model_path = "./data/models/best_model.pth"
    model = torch.load(model_path, map_location=device)
    if isinstance(model, dict):  # 如果只保存了 state_dict
        model_instance = RebuyModel(
            total_items=len(label_encoder.classes_),
            user_num_feat_dim=user_feature_dims.get("num_feats", 9),
            user_cat_feat_dim=user_feature_dims.get("cat_feats", 2),
            merchant_feat_dim=13,
            emb_dim=32,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
        ).to(device)
        model_instance.load_state_dict(model)
        model = model_instance
    save_path = "./data/output/test_pred.csv"
    predict(model, test_loader, save_path=save_path, device=device)
    return


if __name__ == "__main__":
    # run(mode="prod")
    run(mode="prod")
    # test_build_merchant_features()
    # test_build_sequence_features()
    # test_build_user_features()
    # predict_with_best_model()
