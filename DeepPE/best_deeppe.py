import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr

# ========== 路径与日志配置 ==========
BASE_DIR = '/root/project'
DATA_DIR = os.path.join(BASE_DIR, 'DeepPE', 'DeepPE_data')
LOG_DIR  = os.path.join(BASE_DIR, 'DeepPE', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, 'train_200epochs.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger()

# ========== 数据集定义 ==========
class PrimeEditingDataset(Dataset):
    def __init__(self, csv_path, length=99):
        df = pd.read_csv(csv_path)
        self.length    = length
        self.wide      = df['wide_initial_target'].values
        self.wide_mut  = df['wide_mutated_target'].values
        # 生物特征列
        num_cols = [
            'PBS_length','RT_length','PBS-RT_length',
            'Tm1_PBS','Tm2_RT','Tm3_cDNA_PAM','Tm4_RT_cDNA','deltaTm',
            'GC_count_1','GC_count_2','GC_count_3',
            'GC_content_1','GC_content_2','GC_content_3',
            'MFE_1_pegRNA','MFE_2_nospacer','MFE_3_RT_PBS_PolyT',
            'MFE_4_spacer_only','MFE_5_spacer_scaf','DeepSpCas9_score'
        ]
        self.features  = df[num_cols].astype(np.float32).values
        self.targets   = df[['Validly_Edited','Unedited','Erroneously_Edited']].astype(np.float32).values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        seq     = self.wide[idx][:self.length]
        seq_mod = self.wide_mut[idx][:self.length]
        return (
            self.one_hot(seq),
            self.one_hot(seq_mod),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx],  dtype=torch.float32)
        )

    def one_hot(self, seq):
        mapping = {'A':0,'C':1,'G':2,'T':3}
        arr = np.zeros((4, self.length), dtype=np.float32)
        for i, c in enumerate(seq.upper()):
            if i >= self.length:
                break
            if c in mapping:
                arr[mapping[c], i] = 1.0
        return torch.from_numpy(arr)

# ========== 模型定义 ==========
class DeepPEModel(nn.Module):
    def __init__(self, filter_size, filter_num, length,
                 node1, node2, feat_dim, bio_proj_dim, dropout):
        super().__init__()
        # 双路卷积
        self.conv     = nn.Conv1d(4, filter_num, kernel_size=filter_size)
        self.conv_mod = nn.Conv1d(4, filter_num, kernel_size=filter_size)
        conv_len = length - filter_size + 1
        flat_dim = conv_len * filter_num

        # 生物特征投影
        self.bio_proj = nn.Linear(feat_dim, bio_proj_dim)

        # 全连接
        total_in = flat_dim * 2 + bio_proj_dim
        self.fc1    = nn.Linear(total_in, node1)
        self.fc2    = nn.Linear(node1, node2) if node2 > 0 else None
        self.out    = nn.Linear(node2 if node2>0 else node1, 3)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, x_mod, bio):
        h1 = F.relu(self.conv(x))
        h1 = self.drop(h1).view(h1.size(0), -1)
        h2 = F.relu(self.conv_mod(x_mod))
        h2 = self.drop(h2).view(h2.size(0), -1)

        bio_p = F.relu(self.bio_proj(bio))
        bio_p = self.drop(bio_p)

        h = torch.cat([h1, h2, bio_p], dim=1)
        h = F.relu(self.fc1(h)); h = self.drop(h)
        if self.fc2:
            h = F.relu(self.fc2(h)); h = self.drop(h)
        return self.out(h)

# ========== 评估函数 ==========
def eval_metrics(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, X_mod, bio, y in loader:
            X, X_mod, bio = X.to(device), X_mod.to(device), bio.to(device)
            pred = model(X, X_mod, bio).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)

    pearsons = []
    spearmans = []
    for i in range(3):
        p = pearsonr(y_true[:, i], y_pred[:, i])[0]
        s = spearmanr(y_true[:, i], y_pred[:, i])[0]
        pearsons.append(p)
        spearmans.append(s)
    return pearsons, spearmans

# ========== 超参数 & 数据加载 ==========
# 最佳超参
best_params = {
    'filter_size': 7,
    'filter_num':  240,
    'node1':       256,
    'node2':       64,
    'bio_proj_dim':20,
    'dropout':     0.125,
    'lr':          0.0021
}
FEAT_DIM = 20    # 生物特征维度
LENGTH   = 99

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练集
train_csv = os.path.join(DATA_DIR, 'metrics_merged_all_train.csv')
train_loader = DataLoader(
    PrimeEditingDataset(train_csv, length=LENGTH),
    batch_size=64, shuffle=True
)

# 验证集：合并 HEK293 (pridict1 + pridict2_HEK293)
df1 = pd.read_csv(os.path.join(DATA_DIR, 'metrics_merged_pridict1_mutated_val.csv'))
df2 = pd.read_csv(os.path.join(DATA_DIR, 'metrics_merged_pridict2_HEK293_mutated_val.csv'))
df_hek = pd.concat([df1, df2], ignore_index=True)
temp_csv = os.path.join(DATA_DIR, 'tmp_hek293_val.csv')
df_hek.to_csv(temp_csv, index=False)
hek_loader = DataLoader(
    PrimeEditingDataset(temp_csv, length=LENGTH),
    batch_size=128, shuffle=False
)

# K562 验证集
k562_csv = os.path.join(DATA_DIR, 'metrics_merged_pridict2_K562_mutated_val.csv')
k562_loader = DataLoader(
    PrimeEditingDataset(k562_csv, length=LENGTH),
    batch_size=128, shuffle=False
)

# ========== 模型、优化器、损失 ==========
model = DeepPEModel(
    filter_size   = best_params['filter_size'],
    filter_num    = best_params['filter_num'],
    length        = LENGTH,
    node1         = best_params['node1'],
    node2         = best_params['node2'],
    feat_dim      = FEAT_DIM,
    bio_proj_dim  = best_params['bio_proj_dim'],
    dropout       = best_params['dropout']
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
loss_fn   = nn.MSELoss()

# ========== 训练 & 记录 ==========
NUM_EPOCHS = 200

for epoch in range(1, NUM_EPOCHS+1):
    # 训练
    model.train()
    for X, X_mod, bio, y in train_loader:
        X, X_mod, bio, y = X.to(device), X_mod.to(device), bio.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X, X_mod, bio)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    # 验证
    p_hek, s_hek   = eval_metrics(model, hek_loader, device)
    p_k562, s_k562 = eval_metrics(model, k562_loader, device)

    logger.info(
        f"Epoch {epoch:03d} | "
        f"HEK293 Pearson={p_hek}, Spearman={s_hek} || "
        f"K562 Pearson={p_k562}, Spearman={s_k562}"
    )
    torch.save(model.state_dict(),  f"DeepPE/model_pth/deeppe_{epoch}.pth")

print(f"训练完毕，日志保存在：{log_file}")