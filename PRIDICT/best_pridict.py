import os
import logging
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr
from Bio import pairwise2  # pip install biopython

# ========================
# 配置参数
# ========================
DATA_DIR     = "/root/project/PRIDICT/PRIDICT_data"
CACHE_DIR    = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_CSV        = os.path.join(DATA_DIR, "metrics_merged_all_train.csv")
VAL1_CSV         = os.path.join(DATA_DIR, "metrics_merged_pridict1_mutated_val.csv")
VAL2_HEK293_CSV  = os.path.join(DATA_DIR, "metrics_merged_pridict2_HEK293_mutated_val.csv")
VAL2_K562_CSV    = os.path.join(DATA_DIR, "metrics_merged_pridict2_K562_mutated_val.csv")

BATCH_SIZE  = 64
NUM_EPOCHS  = 200
SEQ_LEN     = 99
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = "/root/project/PRIDICT/log"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train_200epochs.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 最佳超参数
best_params = {
    'lr': 0.0005725692011458429,
    'hidden_dim': 448,
    'num_layers': 1,
    'dropout': 0.26118575333678495,
    'nucl_emb': 256,
    'spec_emb': 32
}

# 碱基到索引的映射
BASE2IDX = {'A':0, 'C':1, 'T':2, 'G':3, '-':4, 'N':5}

def encode_seq(seq: str):
    arr = [BASE2IDX.get(ch, 5) for ch in seq.upper()]
    if len(arr) < SEQ_LEN:
        arr += [BASE2IDX['N']] * (SEQ_LEN - len(arr))
    else:
        arr = arr[:SEQ_LEN]
    return arr

def align_pair(seq1: str, seq2: str):
    aln = pairwise2.align.globalms(
        seq1, seq2,
        2, -1, -0.5, -0.1,
        one_alignment_only=True
    )[0]
    return aln.seqA, aln.seqB

def preprocess_and_cache(csv_path):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    cache_file = os.path.join(CACHE_DIR, f"{stem}.npz")
    if os.path.exists(cache_file):
        return np.load(cache_file, allow_pickle=True)
    df = pd.read_csv(csv_path)
    init_seqs, mut_seqs = [], []
    for _, row in df.iterrows():
        s1, s2 = align_pair(row["wide_initial_target"], row["wide_mutated_target"])
        init_seqs.append(encode_seq(s1))
        mut_seqs.append(encode_seq(s2))
    exclude = ["uniqueindex","wide_initial_target","wide_mutated_target",
               "Validly_Edited","Unedited","Erroneously_Edited"]
    features = df.drop(columns=exclude).values.astype(np.float32)
    targets  = df[["Validly_Edited","Unedited","Erroneously_Edited"]].values.astype(np.float32)
    np.savez_compressed(cache_file,
                        init_seqs=np.array(init_seqs, dtype=np.int64),
                        mut_seqs =np.array(mut_seqs,  dtype=np.int64),
                        features=features,
                        targets=targets)
    return np.load(cache_file, allow_pickle=True)

# 预处理并缓存
_cache = {
    TRAIN_CSV:       preprocess_and_cache(TRAIN_CSV),
    VAL1_CSV:        preprocess_and_cache(VAL1_CSV),
    VAL2_HEK293_CSV: preprocess_and_cache(VAL2_HEK293_CSV),
    VAL2_K562_CSV:   preprocess_and_cache(VAL2_K562_CSV),
}

class PridictDataset(Dataset):
    def __init__(self, data_npz):
        self.init_seqs = data_npz["init_seqs"]
        self.mut_seqs  = data_npz["mut_seqs"]
        self.features  = data_npz["features"]
        self.targets   = data_npz["targets"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.init_seqs[idx], dtype=torch.long),
            torch.tensor(self.mut_seqs[idx],  dtype=torch.long),
            torch.tensor(self.features[idx],  dtype=torch.float32),
            torch.tensor(self.targets[idx],   dtype=torch.float32)
        )

def collate_fn(batch):
    init, mut, feat, targ = zip(*batch)
    return (
        torch.stack(init),
        torch.stack(mut),
        torch.stack(feat),
        torch.stack(targ)
    )

# ========================
# 模型定义（同之前）
# ========================
class AnnotEmbedder(nn.Module):
    def __init__(self, nucl_dim, spec_dim):
        super().__init__()
        self.nucl = nn.Embedding(6, nucl_dim, padding_idx=BASE2IDX['N'])
        self.pbs  = nn.Embedding(2, spec_dim)
        self.rt   = nn.Embedding(2, spec_dim)
    def forward(self, seq, pbs_feat, rt_feat):
        n = self.nucl(seq)
        p = self.pbs((pbs_feat>0.5).long()).unsqueeze(1).expand(-1, seq.size(1), -1)
        r = self.rt((rt_feat>0.5).long()).unsqueeze(1).expand(-1, seq.size(1), -1)
        return torch.cat([n,p,r], dim=-1)

class BiRNNEncoder(nn.Module):
    def __init__(self, inp, hid, nl, dp):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, nl,
                            bidirectional=True,
                            dropout=dp if nl>1 else 0,
                            batch_first=True)
    def forward(self, x):
        out,_ = self.lstm(x)
        return out

class FeatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))
        self.s = dim**0.25
    def forward(self, x):
        sc = torch.matmul(x/self.s, self.q/self.s)
        w  = torch.softmax(sc, dim=-1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)

class PRIDICT(nn.Module):
    def __init__(self, feat_dim, nucl_dim, spec_dim, hid, nl, dp):
        super().__init__()
        self.ie = AnnotEmbedder(nucl_dim, spec_dim)
        self.me = AnnotEmbedder(nucl_dim, spec_dim)
        inp = nucl_dim+2*spec_dim
        self.ei = BiRNNEncoder(inp, hid, nl, dp)
        self.em = BiRNNEncoder(inp, hid, nl, dp)
        self.ai = FeatureAttention(2*hid)
        self.am = FeatureAttention(2*hid)
        self.fn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(), nn.Dropout(dp)
        )
        self.head = nn.Sequential(
            nn.Linear(2*hid*2 + 128, 256),
            nn.ReLU(), nn.Dropout(dp),
            nn.Linear(256, 3)
        )
    def forward(self, si, sm, f):
        p = f[:,9]; r = f[:,10]
        ei  = self.ie(si, p, r); oi = self.ei(ei); ci = self.ai(oi)
        em  = self.me(sm, p, r); om = self.em(em); cm = self.am(om)
        fn  = self.fn(f)
        x   = torch.cat([ci, cm, fn], dim=-1)
        return self.head(x)

# ========================
# 数据加载
# ========================
train_ds    = PridictDataset(_cache[TRAIN_CSV])
train_loader= DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 合并 HEK293 验证集
data1 = _cache[VAL1_CSV]; data2 = _cache[VAL2_HEK293_CSV]
hek_init = np.concatenate([data1["init_seqs"], data2["init_seqs"]], axis=0)
hek_mut  = np.concatenate([data1["mut_seqs"],  data2["mut_seqs"]],  axis=0)
hek_feat = np.concatenate([data1["features"],  data2["features"]],  axis=0)
hek_targ = np.concatenate([data1["targets"],   data2["targets"]],   axis=0)
hek_npz = {'init_seqs': hek_init, 'mut_seqs': hek_mut, 'features': hek_feat, 'targets': hek_targ}

k562_npz = _cache[VAL2_K562_CSV]

val_hek_ds  = PridictDataset(hek_npz)
val_k562_ds= PridictDataset(k562_npz)
val_hek_loader   = DataLoader(val_hek_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
val_k562_loader  = DataLoader(val_k562_ds,batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========================
# 模型、优化器、损失
# ========================
model = PRIDICT(
    feat_dim=train_ds.features.shape[1],
    nucl_dim=best_params['nucl_emb'],
    spec_dim=best_params['spec_emb'],
    hid=best_params['hidden_dim'],
    nl=best_params['num_layers'],
    dp=best_params['dropout']
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
loss_fn   = nn.MSELoss()

# ========================
# 验证函数和主训练循环
# ========================
for epoch in range(1, NUM_EPOCHS+1):
    # 训练
    model.train()
    for si, sm, f, tg in train_loader:
        si, sm, f, tg = [t.to(DEVICE) for t in (si, sm, f, tg)]
        optimizer.zero_grad()
        loss = loss_fn(model(si, sm, f), tg)
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    def eval_loader(loader):
        preds_all, trues_all = [], []
        with torch.no_grad():
            for si, sm, f, tg in loader:
                si, sm, f = si.to(DEVICE), sm.to(DEVICE), f.to(DEVICE)
                pred = model(si, sm, f).cpu().numpy()
                preds_all.append(pred)
                trues_all.append(tg.numpy())
        preds_all = np.concatenate(preds_all, axis=0)
        trues_all = np.concatenate(trues_all, axis=0)

        metrics = []
        for i in range(3):  # 对每个预测分量
            pearson = pearsonr(trues_all[:, i], preds_all[:, i])[0]
            spearman = spearmanr(trues_all[:, i], preds_all[:, i])[0]
            metrics.append((pearson, spearman))
        return metrics

    hek_metrics  = eval_loader(val_hek_loader)
    k562_metrics = eval_loader(val_k562_loader)

    log_line = f"Epoch{epoch:03d} "
    for i, (p, s) in enumerate(hek_metrics):
        log_line += f"HEK293[{i}] P={p:.4f} S={s:.4f}  "
    for i, (p, s) in enumerate(k562_metrics):
        log_line += f"K562[{i}] P={p:.4f} S={s:.4f}  "

    logging.info(log_line)
    torch.save(model.state_dict(),  f"PRIDICT/model_pth/pridict_{epoch}.pth")

print("Training complete. 详细日志见：", os.path.join(LOG_DIR, "train_200epochs.log"))
