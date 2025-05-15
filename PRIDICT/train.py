import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ========================
# 配置参数
# ========================
DATA_DIR = "/root/project/PRIDICT/PRIDICT_data"
TRAIN_CSV = os.path.join(DATA_DIR, "metrics_merged_all_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "metrics_merged_pridict1_mutated_val.csv")
TEST_CSV = os.path.join(DATA_DIR, "metrics_merged_pridict1_mutated_test.csv")

BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 1e-3
SEQ_LEN = 99  # 根据数据实际长度调整
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 碱基到索引的映射 (A,C,G,T,N)
NUC2IDX = {"A":0, "C":1, "G":2, "T":3, "N":4}

def encode_seq(seq):
    """将DNA序列编码为索引序列，并进行填充/截断"""
    arr = [NUC2IDX.get(ch, 4) for ch in seq.upper()]
    if len(arr) < SEQ_LEN: 
        arr += [4]*(SEQ_LEN - len(arr))  # 用N填充
    else:                
        arr = arr[:SEQ_LEN]  # 截断
    return arr

# ========================
# 数据集定义（重要修改）
# ========================
class PridictDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # 处理目标列
        self.targets = df[["Validly_Edited", "Unedited", "Erroneously_Edited"]].values.astype(float)
        
        # 处理双序列
        self.init_seqs = [encode_seq(s) for s in df["wide_initial_target"]]
        self.mut_seqs = [encode_seq(s) for s in df["wide_mutated_target"]]
        
        # 提取数值特征（排除序列和目标列）
        exclude_cols = ["uniqueindex", "wide_initial_target", "wide_mutated_target",
                       "Validly_Edited", "Unedited", "Erroneously_Edited"]
        self.features = df.drop(columns=exclude_cols).values.astype(float)
        
        print(f"Loaded dataset: {len(self)} samples")
        print(f"Feature dimension: {self.features.shape[1]}")
        print(f"Target dimension: {self.targets.shape[1]}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.init_seqs[idx], dtype=torch.long),  # 初始序列
            torch.tensor(self.mut_seqs[idx], dtype=torch.long),   # 突变序列
            torch.tensor(self.features[idx], dtype=torch.float32),# 数值特征
            torch.tensor(self.targets[idx], dtype=torch.float32)  # 目标值
        )

def collate_fn(batch):
    """自定义批次处理"""
    init_seqs, mut_seqs, feats, targets = zip(*batch)
    return (
        torch.stack(init_seqs),
        torch.stack(mut_seqs),
        torch.stack(feats),
        torch.stack(targets)
    )

# ========================
# 模型架构（完整实现）
# ========================
class AnnotEmbedder(nn.Module):
    """序列注释嵌入器"""
    def __init__(self, nucl_embed_dim=128, spec_embed_dim=32):
        super().__init__()
        # 核苷酸嵌入（A,C,G,T,N）
        self.nucl_embed = nn.Embedding(5, nucl_embed_dim, padding_idx=4)
        
        # 特殊标记嵌入（PBS、RT，根据数据特征动态生成）
        self.pbs_embed = nn.Embedding(2, spec_embed_dim)  # 二进制特征
        self.rt_embed = nn.Embedding(2, spec_embed_dim)   # 二进制特征
        
    def forward(self, x_seq, pbs_feat, rt_feat):
        # 核苷酸嵌入
        nucl_emb = self.nucl_embed(x_seq)  # (B, L, D)
        
        # 特殊特征嵌入（从数值特征动态生成）
        pbs_idx = (pbs_feat > 0.5).long()  # 阈值处理
        rt_idx = (rt_feat > 0.5).long()
        pbs_emb = self.pbs_embed(pbs_idx).unsqueeze(1).expand(-1, x_seq.size(1), -1)
        rt_emb = self.rt_embed(rt_idx).unsqueeze(1).expand(-1, x_seq.size(1), -1)
        
        return torch.cat([nucl_emb, pbs_emb, rt_emb], dim=-1)

class BiRNNEncoder(nn.Module):
    """双向RNN编码器"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers>1 else 0,
            batch_first=True
        )
        
    def forward(self, x):
        # 假设输入已经填充为相同长度
        outputs, _ = self.rnn(x)
        return outputs  # (B, L, 2H)

class FeatureAttention(nn.Module):
    """特征注意力机制"""
    def __init__(self, feat_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.scale = feat_dim ** 0.25
        
    def forward(self, x):
        # x: (B, L, D)
        scores = torch.matmul(x / self.scale, self.query / self.scale)  # (B, L)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1), attn_weights

class PRIDICT(nn.Module):
    """完整模型架构"""
    def __init__(self, num_features):
        super().__init__()
        # 双序列嵌入层
        self.init_embedder = AnnotEmbedder()
        self.mut_embedder = AnnotEmbedder()
        
        # RNN编码器
        self.init_encoder = BiRNNEncoder(input_dim=128+32*2)  # nucl(128) + pbs(32) + rt(32)
        self.mut_encoder = BiRNNEncoder(input_dim=128+32*2)
        
        # 注意力层
        self.init_attn = FeatureAttention(512)  # 2*256 hidden
        self.mut_attn = FeatureAttention(512)
        
        # 特征融合
        self.feat_processor = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(512*2 + 128, 256),  # init_attn + mut_attn + features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 预测3个目标
        )

    def forward(self, init_seq, mut_seq, features):
        # 从特征中提取PBS和RT信息
        pbs_feat = features[:, 9]  # PBSlength_norm列索引
        rt_feat = features[:, 10]  # RTlength_norm列索引
        
        # 初始序列处理
        init_emb = self.init_embedder(init_seq, pbs_feat, rt_feat)
        init_out = self.init_encoder(init_emb)
        init_attn, _ = self.init_attn(init_out)
        
        # 突变序列处理
        mut_emb = self.mut_embedder(mut_seq, pbs_feat, rt_feat)
        mut_out = self.mut_encoder(mut_emb)
        mut_attn, _ = self.mut_attn(mut_out)
        
        # 特征处理
        feat_emb = self.feat_processor(features)
        
        # 特征融合
        combined = torch.cat([init_attn, mut_attn, feat_emb], dim=-1)
        
        return self.regressor(combined)

# ========================
# 训练流程
# ========================
def main():
    # 初始化数据集
    train_ds = PridictDataset(TRAIN_CSV)
    val_ds = PridictDataset(VAL_CSV)
    test_ds = PridictDataset(TEST_CSV)

    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 初始化模型
    model = PRIDICT(num_features=train_ds.features.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS+1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for init_seq, mut_seq, feats, targets in train_loader:
            init_seq, mut_seq, feats, targets = [t.to(DEVICE) for t in [init_seq, mut_seq, feats, targets]]
            
            optimizer.zero_grad()
            outputs = model(init_seq, mut_seq, feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * init_seq.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for init_seq, mut_seq, feats, targets in val_loader:
                init_seq, mut_seq, feats, targets = [t.to(DEVICE) for t in [init_seq, mut_seq, feats, targets]]
                
                outputs = model(init_seq, mut_seq, feats)
                val_loss += criterion(outputs, targets).item() * init_seq.size(0)
        
        # 统计结果
        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # 测试阶段
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for init_seq, mut_seq, feats, targets in test_loader:
            init_seq, mut_seq, feats, targets = [t.to(DEVICE) for t in [init_seq, mut_seq, feats, targets]]
            
            outputs = model(init_seq, mut_seq, feats)
            test_loss += criterion(outputs, targets).item() * init_seq.size(0)
    
    print(f"\nFinal Test Loss: {test_loss/len(test_ds):.4f}")

if __name__ == "__main__":
    main()
