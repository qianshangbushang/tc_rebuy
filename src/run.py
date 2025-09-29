"""重构后的模型入口"""

import os

import pandas as pd
import torch

from dataset import build_dataloaders, split_train_val_test
from feature_engineering import build_merchant_features, build_sequence_features, build_user_features
from rebuy_model import RebuyModel, evaluate, predict, train


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

    nrows = 1000 if mode == "dev" else None

    # 读取数据
    df_train = pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=nrows)
    df_test = pd.read_csv("./data/format2/data_format2/test_format2.csv", nrows=nrows)

    # 1. 特征工程
    print("\n1. 构建特征")
    # 用户特征
    user_features, user_feature_dims = build_user_features(
        pd.concat([df_train, df_test]),
        cache_dir=config["cache_dir"],
    )

    # 商户特征
    merchant_features, merchant_feature_dims = build_merchant_features(
        pd.concat([df_train, df_test]),
        cache_dir=config["cache_dir"],
    )

    # 序列特征
    sequence_features, encoders = build_sequence_features(
        pd.concat([df_train, df_test]),
        user_features,
        merchant_features,
        user_feature_dims=user_feature_dims,
        merchant_feature_dims=merchant_feature_dims,
        seq_len=config["seq_len"],
        cache_dir=config["cache_dir"],
    )

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
    total_items = sum(len(enc.classes_) for enc in encoders.values())
    # user_feat_dim = len(user_features.columns) - 1  # 减去user_id
    merchant_feat_dim = len(merchant_features.columns) - 1  # 减去merchant_id

    model = RebuyModel(
        total_items=total_items,
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

    merchant_features, feature_dims = build_merchant_features(
        pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=1000),
        cache_dir="./data/cache",
    )
    print("Merchant Features:")
    print(merchant_features.head())
    print(merchant_features.shape)
    print("\nFeature Dimensions:")
    print(feature_dims)


def test_build_sequence_features():
    from feature_engineering import build_merchant_features, build_sequence_features, build_user_features

    df = pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=1000)
    # df = load_dataframe(nrow=1000)
    user_features, user_feature_dims = build_user_features(df, cache_dir="./data/cache")
    merchant_features, merchant_feature_dims = build_merchant_features(df, cache_dir="./data/cache")
    sequence_features, encoders = build_sequence_features(
        df,
        user_features,
        merchant_features,
        user_feature_dims=user_feature_dims,
        merchant_feature_dims=merchant_feature_dims,
        seq_len=30,
        cache_dir="./data/cache",
    )
    for k, v in sequence_features.items():
        print(
            k,
            len(v),
        )
        print(v[:10])


def test_build_user_features():
    from feature_engineering import build_user_features

    user_features, feature_dims = build_user_features(
        pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=1000),
        cache_dir="./data/cache",
    )
    print("User Features:")
    print(user_features.head())
    print(user_features.shape)
    print(user_features.dtypes)
    print("\nFeature Dimensions:")
    print(feature_dims)


if __name__ == "__main__":
    run()
    # test_build_merchant_features()
    # test_build_sequence_features()
    # test_build_user_features()

    # test_build_sequence_features()
    # test_build_user_features()

    # test_build_sequence_features()
    # test_build_user_features()

    # test_build_sequence_features()
    # test_build_user_features()

    # test_build_sequence_features()
    # test_build_user_features()

    # test_build_sequence_features()
    # test_build_user_features()
