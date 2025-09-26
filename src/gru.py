import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class BehaviorSequenceDataset(Dataset):
    """
    用户-商户行为序列数据集
    每个样本为一行数据，自动解析 activity_log 并编码
    """

    def __init__(self, df, seq_len=30, label_col="label"):
        self.seq_len = seq_len
        self.label_col = label_col
        df[label_col] = (df[label_col] == 2).astype(int)  # 二分类
        self.df = df.copy()
        self.item_encoders = LabelEncoder()
        self.cate_encoders = LabelEncoder()
        self.brand_encoders = LabelEncoder()
        self.action_encoders = LabelEncoder()
        self.merchant_encoders = LabelEncoder()
        self.__init_encoders(df)

    def __init_encoders(self, df):
        logs = df["activity_log"].dropna()
        records = [record.split(":") for log in logs for record in log.split("#")]

        self.item_encoders.fit([r[0] for r in records if len(r) >= 5])
        self.cate_encoders.fit([r[1] for r in records if len(r) >= 5])
        self.brand_encoders.fit([r[2] for r in records if len(r) >= 5])
        self.action_encoders.fit([r[4] for r in records if len(r) >= 5])
        self.merchant_encoders.fit(df["merchant_id"].dropna().astype(str).unique())

        print("✅ 初始化离散特征编码器完成")
        print("item 类别数:", len(self.item_encoders.classes_))
        print("cate 类别数:", len(self.cate_encoders.classes_))
        print("brand 类别数:", len(self.brand_encoders.classes_))
        print("action 类别数:", len(self.action_encoders.classes_))
        print("merchant 类别数:", len(self.merchant_encoders.classes_))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        log = row["activity_log"]
        actions = [x.split(":") for x in log.split("#")]
        # 时间字符串补全年份（假设格式为 MMDD），如 "0101" -> "20240101"
        times = pd.to_datetime(["2024" + x[3] for x in actions], errors="coerce")
        time_gap = np.diff(times.values).astype("timedelta64[D]").astype(float) if len(times) > 1 else np.zeros(1)
        time_gap = np.pad(time_gap, (0, max(0, self.seq_len - len(time_gap))), "constant")
        seq = np.array(
            [
                [
                    self.item_encoders.transform([x[0]])[0],
                    self.cate_encoders.transform([x[1]])[0],
                    self.brand_encoders.transform([x[2]])[0],
                    self.merchant_encoders.transform([str(row["merchant_id"])])[0],
                    self.action_encoders.transform([x[4]])[0],
                ]
                for x in actions
            ]
        )
        if len(seq) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq), seq.shape[1]))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[-self.seq_len :]
            time_gap = time_gap[-self.seq_len :]
        label = row[self.label_col]
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(time_gap, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


class GRURecModel(nn.Module):
    def __init__(self, n_items, n_cates, n_brands, n_merchants, n_actions, emb_dim=32, seq_len=30, hidden_dim=64):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.cate_emb = nn.Embedding(n_cates, emb_dim)
        self.brand_emb = nn.Embedding(n_brands, emb_dim)
        self.merchant_emb = nn.Embedding(n_merchants, emb_dim)
        self.action_emb = nn.Embedding(n_actions, emb_dim)
        self.time_fc = nn.Linear(1, emb_dim)
        self.gru = nn.GRU(emb_dim * 5 + emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, seq, time_gap):
        # seq: [B, seq_len, 5]
        item = self.item_emb(seq[:, :, 0])
        cate = self.cate_emb(seq[:, :, 1])
        brand = self.brand_emb(seq[:, :, 2])
        merchant = self.merchant_emb(seq[:, :, 3])
        action = self.action_emb(seq[:, :, 4])
        time_gap = time_gap.unsqueeze(-1)  # [B, seq_len, 1]
        time_feat = self.time_fc(time_gap)
        # print(item.shape, time_feat.shape)
        x = torch.cat([item, cate, brand, merchant, action, time_feat], dim=-1)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out.squeeze(-1)


def extract_all_ids(df):
    """
    从df的activity_log字段中提取所有item_id, cate_id, brand_id, merchant_id的集合
    返回: item_ids, cate_ids, brand_ids, merchant_ids
    """
    item_ids = set()
    cate_ids = set()
    brand_ids = set()
    actions = set()
    merchant_ids = set()

    # 只处理非空 activity_log
    logs = df["activity_log"].dropna()
    for log in logs:
        for record in log.split("#"):
            parts = record.split(":")
            if len(parts) >= 5:
                item_ids.add(parts[0])
                cate_ids.add(parts[1])
                brand_ids.add(parts[2])
                actions.add(parts[4])
                # merchant_id 需从 df 的 merchant_id 列获取
    merchant_ids.update(df["merchant_id"].dropna().astype(str).unique())

    print("✅ 提取所有离散特征ID完成")
    return item_ids, cate_ids, brand_ids, actions, merchant_ids


def train_gru_model(
    df, seq_len=30, batch_size=128, epochs=10, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"
):
    # 构建数据集
    dataset = BehaviorSequenceDataset(df, seq_len=seq_len, label_col="label")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 获取类别数

    # 构建模型
    model = GRURecModel(
        n_items=len(dataset.item_encoders.classes_),
        n_cates=len(dataset.cate_encoders.classes_),
        n_brands=len(dataset.brand_encoders.classes_),
        n_merchants=len(dataset.merchant_encoders.classes_),
        n_actions=len(dataset.action_encoders.classes_),
        seq_len=seq_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 训练
    model.train()
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
        print(f"Epoch {epoch + 1}: loss={np.mean(losses):.4f}")
    return model, dataset


def evaluate_gru_model(model, dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
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
    print(f"GRU模型评估: AUC={auc:.4f}, ACC={acc:.4f}")
    return auc, acc


def load_data():
    df = pd.read_csv("./data/format2/data_format2/train_format2.csv")
    df = df[df["activity_log"].notnull()]
    df = df.assign(activity_log=df["activity_log"].str.split("#")).explode("activity_log")
    split_columns = ["item_id", "cate_id", "brand_id", "time", "action_type"]
    df[split_columns] = df["activity_log"].str.split(":", expand=True)
    df["timestamp"] = pd.to_datetime(df["time"].apply(lambda x: "2024" + x), errors="coerce")
    print(f"✅ 行为日志展开完成，数据形状: {df.shape}")
    return df


def test_dataset():
    df = pd.read_csv("./data/format2/data_format2/train_format2.csv", nrows=1000)
    dataset = BehaviorSequenceDataset(df, seq_len=30, label_col="label")
    for r in dataset:
        seq, time_gap, label = r
        print("行为序列形状:", seq.shape)
        print("时间间隔形状:", time_gap.shape)
        print("标签:", label)


def run():
    df = pd.read_csv("./data/format2/data_format2/train_format2.csv")
    # 训练模型
    model, dataset = train_gru_model(df, seq_len=30, batch_size=128, epochs=5)
    # 评估模型
    evaluate_gru_model(model, dataset)


if __name__ == "__main__":
    # test_dataset()
    run()
