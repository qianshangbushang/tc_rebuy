import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# 构建特征


class UserFeatureEncoder:
    """用户全局特征编码器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.action_types = None
        self.user_info = None
        # LabelEncoder for age_range and gender
        self.age_range_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()

    def fit(self, df: pd.DataFrame, user_info_df: pd.DataFrame) -> None:
        """构建用户全局特征

        Args:
            df: 原始数据，包含user_id, merchant_id, activity_log等字段
            user_info_df: 用户信息数据，包含user_id, age_range, gender等字段
        """
        # 展开日志
        df_explode = df.copy()
        df_explode["log_list"] = df_explode["activity_log"].str.split("#")
        df_explode = df_explode.explode("log_list")
        df_explode[["item_id", "cate_id", "brand_id", "time_str", "action_type"]] = df_explode["log_list"].str.split(
            ":", expand=True
        )

        # 计算用户统计特征
        user_stats = (
            df_explode.groupby("user_id")
            .agg(
                {
                    "merchant_id": ["nunique"],  # 交互过的商户数
                    "item_id": ["nunique"],  # 交互过的商品数
                    "action_type": ["count"],  # 总行为数
                    "cate_id": ["nunique"],  # 交互过的类目数
                    "brand_id": ["nunique"],  # 交互过的品牌数
                }
            )
            .reset_index()
        )
        user_stats.columns = ["user_id", "merchant_cnt", "item_cnt", "action_cnt", "cate_cnt", "brand_cnt"]

        # 计算用户在不同action_type上的分布
        action_dist = pd.crosstab(df_explode["user_id"], df_explode["action_type"], normalize="index").reset_index()
        self.action_types = action_dist.columns.tolist()[1:]

        # 处理用户信息数据
        user_info_df = user_info_df.copy()
        user_info_df["age_range"] = user_info_df["age_range"].fillna("unknown")
        user_info_df["gender"] = user_info_df["gender"].fillna("unknown")

        # 编码age_range和gender
        self.age_range_encoder.fit(user_info_df["age_range"])
        self.gender_encoder.fit(user_info_df["gender"])

        age_range_encoded = pd.get_dummies(user_info_df["age_range"], prefix="age")
        gender_encoded = pd.get_dummies(user_info_df["gender"], prefix="gender")

        # 合并用户信息
        self.user_info = pd.concat([user_info_df[["user_id"]], age_range_encoded, gender_encoded], axis=1)

        # 合并所有特征
        self.user_features = pd.merge(user_stats, action_dist, on="user_id", how="left")
        self.user_features = pd.merge(self.user_features, self.user_info, on="user_id", how="left")
        self.user_features = self.user_features.fillna(0)

        # 标准化数值特征
        num_cols = ["merchant_cnt", "item_cnt", "action_cnt", "cate_cnt", "brand_cnt"]
        self.user_features[num_cols] = self.scaler.fit_transform(self.user_features[num_cols])

        print("✅ 用户特征构建完成:")
        print(f"  - 数值特征: {num_cols}")
        print(f"  - 行为类型: {self.action_types}")
        print(f"  - 年龄段: {list(age_range_encoded.columns)}")
        print(f"  - 性别: {list(gender_encoded.columns)}")

    def transform(self, user_ids: list[str]) -> np.ndarray:
        """获取用户特征向量"""
        return self.user_features[self.user_features["user_id"].isin(user_ids)].iloc[:, 1:].values


class RebuyModel(nn.Module):
    """二次购买预测模型

    使用自注意力机制和层次化门控特征融合的深度神经网络模型，用于预测用户是否会对商家进行二次购买。

    特点:
    1. 多头自注意力机制捕获序列中的长距离依赖
    2. 区分处理用户的数值特征和类别特征
        - 数值特征通过全连接层处理
        - 类别特征通过嵌入层和全连接层处理
    3. 三层门控特征融合机制
        - 用户特征内部融合(数值特征和类别特征)
        - 用户-商户特征融合
        - 序列-全局特征融合
    4. 残差连接和层归一化提升训练稳定性
    5. 双向GRU提取序列特征的时序依赖
    """

    def __init__(
        self,
        total_items: int,
        user_num_feat_dim: int,
        user_cat_feat_dim: int,
        merchant_feat_dim: int,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """初始化模型

        Args:
            total_items: 所有类别特征的总维度
            user_num_feat_dim: 用户数值特征维度
            user_cat_feat_dim: 用户类别特征维度（需要embedding的特征）
            merchant_feat_dim: 商户特征维度
            emb_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: GRU层数
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()

        self.dropout = dropout

        # 1. 序列特征处理
        # 1.1 嵌入层
        self.item_emb = nn.Embedding(total_items, emb_dim)
        self.time_fc = nn.Linear(1, emb_dim)

        # 1.2 序列编码
        seq_input_dim = emb_dim * 6  # 5个类别特征 + 时间特征
        self.seq_proj = nn.Linear(seq_input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 1.3 自注意力机制
        # 设置多头注意力,添加key_padding_mask参数
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional GRU
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_layer_norm = nn.LayerNorm(hidden_dim * 2)

        # 2. 用户和商户特征处理
        # 2.1 用户数值特征处理
        self.user_num_fc = nn.Sequential(
            nn.Linear(user_num_feat_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2.2 用户类别特征处理
        # self.user_cat_emb = nn.Embedding(user_cat_feat_dim, emb_dim)
        self.user_cat_fc = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim * 2),  # 2是类别特征数量(age_range和gender)
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2.3 商户特征处理
        self.merchant_fc = nn.Sequential(
            nn.Linear(merchant_feat_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 3. 特征融合
        # 3.1 层次化门控融合机制
        # 各特征分支的输出维度相同,都是hidden_dim * 2
        fusion_dim = hidden_dim * 2

        # 3.1.1 用户特征内部融合(数值特征和类别特征)
        self.user_feat_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Sigmoid(),
        )

        # 3.1.2 用户-商户特征融合
        self.user_merchant_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Sigmoid(),
        )

        # 3.1.3 序列-全局特征融合
        self.seq_global_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Sigmoid(),
        )

        # 3.2 输出层
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),  # 移除Sigmoid，使用BCEWithLogitsLoss
        )

        # 4. 参数初始化
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for name, param in self.named_parameters():
            if "weight" in name and "norm" not in name:
                if len(param.shape) >= 2:
                    # 对于2维及以上的权重使用xavier初始化
                    nn.init.xavier_uniform_(param)
                else:
                    # 对于1维的参数(如bias)使用均匀分布初始化
                    nn.init.uniform_(param, -0.1, 0.1)

    def _create_padding_mask(self, seq):
        """创建填充掩码"""
        return (seq.sum(dim=-1) == 0).to(seq.device)  # [batch_size, seq_len]

    def forward(self, seq, time_gap, user_num_feat, user_cat_feat, merchant_feat):
        """前向传播

        Args:
            seq: 序列输入 [batch_size, seq_len, num_features]
            user_num_feat: 用户数值特征 [batch_size, user_num_feat_dim]
            user_cat_feat: 用户类别特征 [batch_size, user_cat_feat_dim]
            time_gap: 时间间隔 [batch_size, seq_len]
            user_feat: 用户特征 [batch_size, user_feat_dim] (包含人口学特征)
            merchant_feat: 商户特征 [batch_size, merchant_feat_dim] (包含顾客群体特征分布)

        Returns:
            torch.Tensor: 二次购买概率 [batch_size, 1]
        """
        # 1. 序列特征处理
        # 1.1 特征嵌入
        emb_list = [self.item_emb(seq[:, :, i]) for i in range(seq.size(2))]
        time_feat = self.time_fc(time_gap.unsqueeze(-1))
        seq = torch.cat(emb_list + [time_feat], dim=-1)  # [batch_size, seq_len, emb_dim * 6]

        # 1.2 序列编码
        seq = self.seq_proj(seq)  # [batch_size, seq_len, hidden_dim]
        seq, _ = self.gru(seq)  # [batch_size, seq_len, hidden_dim * 2]

        # 1.3 自注意力
        # 创建填充掩码 [batch_size, seq_len]
        key_padding_mask = self._create_padding_mask(seq)

        # 创建因果掩码 [seq_len, seq_len]
        sz = seq.size(1)
        causal_mask = torch.triu(torch.full((sz, sz), float("-inf"), device=seq.device), diagonal=1)

        # 使用key_padding_mask处理填充,使用attn_mask处理因果关系
        seq_attn, _ = self.self_attn(
            query=seq, key=seq, value=seq, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        seq = seq + seq_attn  # 残差连接
        seq = self.seq_layer_norm(seq)  # 层归一化

        # 取序列最后一个时间步的状态
        seq = seq[:, -1]  # [batch_size, hidden_dim * 2]

        # 2. 用户和商户特征处理
        # 2.1 用户数值特征处理
        user_num = self.user_num_fc(user_num_feat)  # [batch_size, hidden_dim * 2]

        # 2.2 用户类别特征处理
        # user_cat_emb = self.user_cat_emb(user_cat_feat.long())  # [batch_size, num_cat_feats, emb_dim]
        user_cat_emb = self.item_emb(user_cat_feat.long())  # [batch_size, num_cat_feats, emb_dim]
        user_cat_feats = user_cat_emb.view(user_cat_feat.size(0), -1)  # [batch_size, num_cat_feats * emb_dim]
        user_cat = self.user_cat_fc(user_cat_feats)  # [batch_size, hidden_dim * 2]

        # 2.3 商户特征处理
        merchant = self.merchant_fc(merchant_feat)  # [batch_size, hidden_dim * 2]

        # 3. 层次化特征融合
        # 3.1 用户特征内部融合(数值特征和类别特征)
        user_feat_gate = self.user_feat_gate(torch.cat([user_num, user_cat], dim=-1))
        user = user_feat_gate * user_num + (1 - user_feat_gate) * user_cat

        # 3.2 用户-商户特征融合
        user_merchant_gate = self.user_merchant_gate(torch.cat([user, merchant], dim=-1))
        global_feat = user_merchant_gate * user + (1 - user_merchant_gate) * merchant

        # 3.2 序列-全局融合
        seq_global_gate = self.seq_global_gate(torch.cat([seq, global_feat], dim=-1))
        fused = seq_global_gate * seq + (1 - seq_global_gate) * global_feat

        # 4. 输出层
        out = self.fc(fused)  # [batch_size, 1]

        return out


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 10,
    lr: float = 1e-3,
    pos_weight: float = 7.0,
    early_stop_rounds: int = 3,
    model_dir: str = "./data/models",
    device: torch.device | str | None = None,
) -> dict[str, list[float]]:
    """训练模型

    训练过程中使用早停策略，当验证集损失在指定轮数内未改善时停止训练。
    同时保存训练过程中的最佳模型。

    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 神经网络模型
        epochs: 最大训练轮数
        lr: 学习率
        early_stop_rounds: 早停判断的轮数，当验证集损失在该轮数内未改善时停止训练
        model_dir: 模型保存目录，保存最佳模型和训练历史
        device: 训练设备，可以是torch.device实例、字符串或None

    Returns:
        dict[str, list[float]]: 训练历史记录，包含以下指标:
            - train_loss: 每轮训练集损失
            - val_loss: 每轮验证集损失
            - val_auc: 每轮验证集AUC
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    os.makedirs(model_dir, exist_ok=True)
    best_val_loss = float("inf")
    no_improve_rounds = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(epochs):
        # 训练
        model.train()
        train_losses = []
        for seq, time_gap, user_num_feat, user_cat_feat, merchant_feat, label, _, _ in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            seq = seq.to(device)
            time_gap = time_gap.to(device)
            user_num_feat = user_num_feat.to(device)
            user_cat_feat = user_cat_feat.to(device)
            merchant_feat = merchant_feat.to(device)
            label = label.to(device).unsqueeze(1)  # 改变形状为 [batch_size, 1]

            pred = model(seq, time_gap, user_num_feat, user_cat_feat, merchant_feat)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for seq, time_gap, user_num_feat, user_cat_feat, merchant_feat, label, _, _ in val_loader:
                seq = seq.to(device)
                time_gap = time_gap.to(device)
                user_num_feat = user_num_feat.to(device)
                user_cat_feat = user_cat_feat.to(device)
                merchant_feat = merchant_feat.to(device)
                label = label.to(device).unsqueeze(1)  # 改变形状为 [batch_size, 1]

                pred = model(seq, time_gap, user_num_feat, user_cat_feat, merchant_feat)
                loss = criterion(pred, label)

                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        # 记录指标
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_auc = roc_auc_score(val_labels, val_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_rounds = 0
            torch.save(model.state_dict(), f"{model_dir}/best_model.pth")
            print(f"  Saved best model to {model_dir}/best_model.pth")
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= early_stop_rounds:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    # 记录训练历史
    pd.DataFrame(history).to_csv(f"{model_dir}/train_history.csv", index=False)
    return history


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | str | None = None,
) -> tuple[float, float]:
    """评估模型

    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备，可以是torch.device实例、字符串或None

    Returns:
        tuple: (AUC, ACC)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估"):
            seq, time_gap, user_num_feat, user_cat_feat, merchant_feat, label, user_id, merchant_id = batch

            seq = seq.to(device)
            time_gap = time_gap.to(device)
            user_num_feat = user_num_feat.to(device)
            user_cat_feat = user_cat_feat.to(device)
            merchant_feat = merchant_feat.to(device)

            logits = model(seq, time_gap, user_num_feat, user_cat_feat, merchant_feat)
            pred = torch.sigmoid(logits).cpu().numpy()

            preds.extend(pred.flatten())
            labels.extend(label.numpy())

    # 计算评估指标
    preds = np.array(preds)
    labels = np.array(labels)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds > 0.5)

    print("\n评估结果:")
    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    print("\n分类报告:")
    print(classification_report(labels, preds > 0.5))

    return float(auc), float(acc)  # 确保返回Python float类型


def predict(
    model: nn.Module, test_loader: DataLoader, save_path: str | None = None, device: torch.device | str | None = None
) -> pd.DataFrame:
    """模型预测

    Args:
        model: 模型
        test_loader: 测试数据加载器
        save_path: 预测结果保存路径，如果为None则不保存
        device: 设备，可以是torch.device实例、字符串或None

    Returns:
        pd.DataFrame: 预测结果DataFrame，包含user_id、merchant_id和prob列
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    results = {"user_id": [], "merchant_id": [], "prob": []}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测"):
            seq, time_gap, user_num_feat, user_cat_feature, merchant_feat, _, user_id, merchant_id = batch

            seq = seq.to(device)
            time_gap = time_gap.to(device)
            user_num_feat = user_num_feat.to(device)
            user_cat_feature = user_cat_feature.to(device)
            merchant_feat = merchant_feat.to(device)

            logits = model(seq, time_gap, user_num_feat, user_cat_feature, merchant_feat)
            prob = torch.sigmoid(logits).cpu().numpy()

            results["user_id"].extend([int(x) for x in user_id])
            results["merchant_id"].extend([int(x) for x in merchant_id])
            results["prob"].extend(prob.flatten())

    # 转换为DataFrame
    pred_df = pd.DataFrame(results)

    # 保存预测结果
    if save_path:
        pred_df.to_csv(save_path, index=False)
        print(f"预测结果已保存到: {save_path}")

    return pred_df
