import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class UnifiedEncoder:
    def __init__(self, item_enc, cate_enc, brand_enc, merchant_enc, action_enc):
        # 计算各特征的偏移量
        self.offsets = {}
        offset = 0
        self.offsets["item"] = offset
        offset += len(item_enc.classes_)
        self.offsets["cate"] = offset
        offset += len(cate_enc.classes_)
        self.offsets["brand"] = offset
        offset += len(brand_enc.classes_)
        self.offsets["merchant"] = offset
        offset += len(merchant_enc.classes_)
        self.offsets["action"] = offset
        offset += len(action_enc.classes_)
        self.total_dim = offset
        self.item_enc = item_enc
        self.cate_enc = cate_enc
        self.brand_enc = brand_enc
        self.merchant_enc = merchant_enc
        self.action_enc = action_enc

    def encode_row(self, x, merchant_id):
        # x: [item_id, cate_id, brand_id, time, action_type]
        return [
            self.offsets["item"] + self.item_enc.transform([x[0]])[0] if len(x) > 0 else 0,
            self.offsets["cate"] + self.cate_enc.transform([x[1]])[0] if len(x) > 1 else 0,
            self.offsets["brand"] + self.brand_enc.transform([x[2]])[0] if len(x) > 2 else 0,
            self.offsets["merchant"] + self.merchant_enc.transform([str(merchant_id)])[0],
            self.offsets["action"] + self.action_enc.transform([x[4]])[0] if len(x) > 4 else 0,
        ]


class BehaviorSequenceDataset(Dataset):
    """
    优化：先explode日志，再编码，然后聚合为序列
    """

    def __init__(
        self,
        df,
        seq_len=30,
        label_col="label",
        enable_cache=True,
    ):
        cache_path = f"./data/seq_{seq_len}_size_{len(df)}.pkl"
        if enable_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.seqs = data["seqs"]
            self.time_gaps = data["time_gaps"]
            self.labels = data["labels"]
            self.item_encoders = data["item_encoders"]
            self.cate_encoders = data["cate_encoders"]
            self.brand_encoders = data["brand_encoders"]
            self.merchant_encoders = data["merchant_encoders"]
            self.action_encoders = data["action_encoders"]
            print(f"✅ 序列特征从缓存加载完成: {cache_path}")
            return

        self.seq_len = seq_len
        self.label_col = label_col
        print("初始标签分布:", df[label_col].value_counts().to_dict())
        df[label_col] = (df[label_col] == 1).astype(int)
        print(f"正负样本比例: {df[label_col].mean():.4f}")
        df = df[df["activity_log"].apply(lambda x: isinstance(x, str) and len(x) > 0)].copy()
        self.df = df

        # explode日志并批量处理
        df_explode = df.copy()
        df_explode["log_list"] = df_explode["activity_log"].str.split("#")
        df_explode = df_explode.explode("log_list")
        df_explode[["item_id", "cate_id", "brand_id", "time_str", "action_type"]] = df_explode["log_list"].str.split(
            ":", expand=True
        )

        # 初始化编码器并批量编码
        self.item_encoders = LabelEncoder()
        self.cate_encoders = LabelEncoder()
        self.brand_encoders = LabelEncoder()
        self.action_encoders = LabelEncoder()
        self.merchant_encoders = LabelEncoder()

        for col, encoder in [
            ("item_id", self.item_encoders),
            ("cate_id", self.cate_encoders),
            ("brand_id", self.brand_encoders),
            ("action_type", self.action_encoders),
            ("merchant_id", self.merchant_encoders),
        ]:
            df_explode[col] = df_explode[col].fillna("UNK")
            encoder.fit(df_explode[col].astype(str))

        df_explode["item_id_enc"] = self.item_encoders.transform(df_explode["item_id"].astype(str))
        df_explode["cate_id_enc"] = self.cate_encoders.transform(df_explode["cate_id"].astype(str))
        df_explode["brand_id_enc"] = self.brand_encoders.transform(df_explode["brand_id"].astype(str))
        df_explode["action_type_enc"] = self.action_encoders.transform(df_explode["action_type"].astype(str))
        df_explode["merchant_id_enc"] = self.merchant_encoders.transform(df_explode["merchant_id"].astype(str))

        # 聚合为序列
        seqs, time_gaps, labels = [], [], []
        grouped = df_explode.groupby(["user_id", "merchant_id"], sort=False)
        for (user_id, merchant_id), group in tqdm(grouped, total=len(grouped), desc="聚合行为序列"):
            seq = group[["item_id_enc", "cate_id_enc", "brand_id_enc", "merchant_id_enc", "action_type_enc"]].values
            times = pd.to_datetime("2024" + group["time_str"].fillna("0101"), errors="coerce")
            time_gap = np.diff(times.values).astype("timedelta64[D]").astype(float) if len(times) > 1 else np.zeros(1)
            time_gap = np.pad(time_gap, (0, max(0, seq_len - len(time_gap))), "constant")
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), seq.shape[1]))
                seq = np.vstack([seq, pad])
            else:
                seq = seq[-seq_len:]
                time_gap = time_gap[-seq_len:]
            seqs.append(seq)
            time_gaps.append(time_gap)
            labels.append(group[label_col].iloc[0])
        self.seqs, self.time_gaps, self.labels = seqs, time_gaps, labels

        # 持久化
        if enable_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "seqs": self.seqs,
                        "time_gaps": self.time_gaps,
                        "labels": self.labels,
                        "item_encoders": self.item_encoders,
                        "cate_encoders": self.cate_encoders,
                        "brand_encoders": self.brand_encoders,
                        "merchant_encoders": self.merchant_encoders,
                        "action_encoders": self.action_encoders,
                    },
                    f,
                )
            print(f"✅ 序列特征已缓存到: {cache_path}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seqs[idx], dtype=torch.long),
            torch.tensor(self.time_gaps[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class UnifiedEmbeddingRecModel(nn.Module):
    def __init__(self, total_dim, emb_dim=32, seq_len=30, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(total_dim, emb_dim)
        self.time_fc = nn.Linear(1, emb_dim)
        self.gru = nn.GRU(emb_dim * 5 + emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, seq, time_gap):
        # seq: [B, seq_len, 5]，每个元素是全局ID
        emb_list = [self.emb(seq[:, :, i]) for i in range(seq.shape[2])]
        time_gap = time_gap.unsqueeze(-1)
        time_feat = self.time_fc(time_gap)
        x = torch.cat(emb_list + [time_feat], dim=-1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


def build_dataloader(
    df,
    seq_len,
    label_col="label",
    batch_size=128,
    shuffle=False,
    enable_cache=True,
    pin_memory=True,
):
    dataset = BehaviorSequenceDataset(df, seq_len=seq_len, label_col=label_col, enable_cache=enable_cache)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return loader, dataset


def build_model(dataset, seq_len=30, emb_dim=32, hidden_dim=64, device=None):
    total_dim = (
        len(dataset.item_encoders.classes_)
        + len(dataset.cate_encoders.classes_)
        + len(dataset.brand_encoders.classes_)
        + len(dataset.merchant_encoders.classes_)
        + len(dataset.action_encoders.classes_)
    )
    model = UnifiedEmbeddingRecModel(
        total_dim=total_dim,
        emb_dim=emb_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
    )
    if device:
        model = model.to(device)
    print(f"✅ UnifiedEmbedding模型构建完成，参数量: {sum(p.numel() for p in model.parameters())}")
    return model


def train_model(model, loader, epochs=5, lr=1e-3, device=None, val_loader=None, early_stop_rounds=3, save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    best_val_loss = float("inf")
    no_improve_rounds = 0
    train_loss_history = []
    val_loss_history = []
    for epoch in range(epochs):
        losses = []
        for seq, time_gap, label in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            seq, time_gap, label = seq.to(device), time_gap.to(device), label.to(device)
            pred = model(seq, time_gap)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_train_loss = np.mean(losses)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}")

        # 验证集loss
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for seq, time_gap, label in val_loader:
                    seq, time_gap, label = seq.to(device), time_gap.to(device), label.to(device)
                    pred = model(seq, time_gap)
                    loss = criterion(pred, label)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            val_loss_history.append(avg_val_loss)
            print(f"Epoch {epoch + 1}: val_loss={avg_val_loss:.4f}")
            # 早停逻辑
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_rounds = 0
                # 保存模型
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"模型已保存到 {save_path}")
            else:
                no_improve_rounds += 1
                if no_improve_rounds >= early_stop_rounds:
                    print(f"验证集loss连续{early_stop_rounds}轮未提升，提前停止训练。")
                    break
            model.train()
        else:
            # 没有验证集也可以保存最后一轮模型
            if save_path and epoch == epochs - 1:
                torch.save(model.state_dict(), save_path)
                print(f"模型已保存到 {save_path}")
    # 记录loss变化
    pd.DataFrame({"train_loss": train_loss_history, "val_loss": val_loss_history if val_loss_history else None}).to_csv(
        "./train_val_loss.csv", index=False
    )
    print("训练/验证loss变化已保存到 ./train_val_loss.csv")


def evaluate_model(model, loader, device=None):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for seq, time_gap, label in loader:
            seq, time_gap = seq.to(device), time_gap.to(device)
            pred = model(seq, time_gap).cpu().numpy()
            preds.extend(pred)
            labels.extend(label.numpy())
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, np.array(preds) > 0.5)
    print(f"UnifiedEmbedding模型评估: AUC={auc:.4f}, ACC={acc:.4f}")
    print(classification_report(labels, np.array(preds) > 0.5))
    return auc, acc


def predict_model(model, loader, device=None, out_path=None):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for seq, time_gap, _ in loader:
            seq, time_gap = seq.to(device), time_gap.to(device)
            pred = model(seq, time_gap).cpu().numpy()
            all_preds.extend(pred)
    if out_path:
        pd.DataFrame({"pred": all_preds}).to_csv(out_path, index=False)
        print(f"测试集预测结果已保存到 {out_path}")
    return all_preds


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_train = pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=10000)
    df_test = pd.read_csv("./data/format2/data_format2/test_format2.csv", nrows=10000)

    # 拆分训练/验证集（80%/20%）
    df_train_sub = df_train.sample(frac=0.8, random_state=42)
    df_val_sub = df_train.drop(df_train_sub.index)

    # 构建 DataLoader
    train_loader, train_dataset = build_dataloader(
        df_train_sub,
        seq_len=10,
        batch_size=128,
        shuffle=True,
        enable_cache=True,
        pin_memory=device == "cuda",
    )
    val_loader, _ = build_dataloader(
        df_val_sub,
        seq_len=10,
        batch_size=128,
        shuffle=False,
        enable_cache=False,
        pin_memory=device == "cuda",
    )
    test_loader, _ = build_dataloader(
        df_test,
        seq_len=10,
        batch_size=128,
        shuffle=False,
        enable_cache=False,
        pin_memory=device == "cuda",
    )

    # 构建模型
    model = build_model(train_dataset, seq_len=10, device=device)

    # 训练（带验证集和早停/保存）
    train_model(
        model,
        train_loader,
        epochs=5,
        lr=1e-3,
        device=device,
        val_loader=val_loader,
        early_stop_rounds=10,
        save_path="./src/output/best_model.pth",
    )

    # 训练集评估
    print("训练集评估结果：")
    evaluate_model(model, train_loader, device=device)

    # 验证集评估
    print("验证集评估结果：")
    evaluate_model(model, val_loader, device=device)

    # 测试集评估
    # print("测试集评估结果：")
    # evaluate_model(model, test_loader, device=device)

    # 测试集预测并保存
    predict_model(model, test_loader, device=device, out_path="./src/output/test_pred.csv")


if __name__ == "__main__":
    run()
    