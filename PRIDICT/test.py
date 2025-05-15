import os
import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from Bio import pairwise2

# ========================
# 配置参数
# ========================
DATA_DIR   = "/root/project/PRIDICT/PRIDICT_data"
CACHE_DIR  = os.path.join(DATA_DIR, "cache")
MODEL_DIR  = "/root/project/PRIDICT/model_pth"
SEQ_LEN    = 99
BATCH_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试集文件名
TEST_CSV = {
    "pridict1":        "metrics_merged_pridict1_mutated_test.csv",
    "pridict2_HEK293": "metrics_merged_pridict2_HEK293_mutated_test.csv",
    "pridict2_K562":   "metrics_merged_pridict2_K562_mutated_test.csv",
}

# 最佳超参数（与你训练时相同）
best_params = {
    'lr': 0.0005725692011458429,
    'hidden_dim': 448,
    'num_layers': 1,
    'dropout': 0.26118575333678495,
    'nucl_emb': 256,
    'spec_emb': 32
}

BASE2IDX = {'A':0, 'C':1, 'T':2, 'G':3, '-':4, 'N':5}

# ========================
# 工具函数
# ========================
def encode_seq(seq: str):
    arr = [BASE2IDX.get(ch, 5) for ch in seq.upper()]
    if len(arr) < SEQ_LEN:
        arr += [BASE2IDX['N']] * (SEQ_LEN - len(arr))
    else:
        arr = arr[:SEQ_LEN]
    return arr

def align_pair(seq1: str, seq2: str):
    aln = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.5, -0.1, one_alignment_only=True)[0]
    return aln.seqA, aln.seqB

def load_or_cache(csv_path):
    """
    对齐、编码并缓存到 .npz，避免重复计算
    返回一个 npz 字典，包含 init_seqs, mut_seqs, features, targets
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    cache_file = os.path.join(CACHE_DIR, f"{stem}.npz")
    if os.path.exists(cache_file):
        return np.load(cache_file, allow_pickle=True)

    df = pd.read_csv(csv_path)
    init_seqs, mut_seqs = [], []
    for _, row in df.iterrows():
        a, b = align_pair(row["wide_initial_target"], row["wide_mutated_target"])
        init_seqs.append(encode_seq(a))
        mut_seqs.append(encode_seq(b))

    drop_cols = ["uniqueindex","wide_initial_target","wide_mutated_target",
                 "Validly_Edited","Unedited","Erroneously_Edited"]
    features = df.drop(columns=drop_cols).values.astype(np.float32)
    targets  = df[["Validly_Edited","Unedited","Erroneously_Edited"]].values.astype(np.float32)

    np.savez_compressed(cache_file,
                        init_seqs=np.array(init_seqs, dtype=np.int64),
                        mut_seqs =np.array(mut_seqs,  dtype=np.int64),
                        features=features,
                        targets=targets)
    return np.load(cache_file, allow_pickle=True)

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
    si, sm, f, tg = zip(*batch)
    return (
        torch.stack(si),
        torch.stack(sm),
        torch.stack(f),
        torch.stack(tg)
    )

# ========================
# 模型定义（与训练脚本一致）
# ========================
class AnnotEmbedder(nn.Module):
    def __init__(self, nucl_dim, spec_dim):
        super().__init__()
        self.nucl = nn.Embedding(6, nucl_dim, padding_idx=BASE2IDX['N'])
        self.pbs  = nn.Embedding(2, spec_dim)
        self.rt   = nn.Embedding(2, spec_dim)
    def forward(self, seq, pbs_feat, rt_feat):
        n = self.nucl(seq)  # (B, L, nucl_dim)
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
        out, _ = self.lstm(x)
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
        inp = nucl_dim + 2*spec_dim
        self.ei = BiRNNEncoder(inp, hid, nl, dp)
        self.em = BiRNNEncoder(inp, hid, nl, dp)
        self.ai = FeatureAttention(2*hid)
        self.am = FeatureAttention(2*hid)
        self.fn = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(dp)
        )
        self.head = nn.Sequential(
            nn.Linear(2*hid*2 + 128, 256), nn.ReLU(), nn.Dropout(dp),
            nn.Linear(256, 3)
        )

    def forward(self, si, sm, f):
        p = f[:,9]; r = f[:,10]
        oi = self.ei(self.ie(si,p,r))
        cm = self.am(self.em(self.me(sm,p,r)))
        ci = self.ai(oi)
        fn = self.fn(f)
        x  = torch.cat([ci, cm, fn], dim=-1)
        return self.head(x)

# ========================
# 推理 & 保存函数
# ========================
def collect_predictions(model, loader, device):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for si, sm, f, tg in loader:
            si, sm, f = si.to(device), sm.to(device), f.to(device)
            out = model(si, sm, f).cpu().numpy()
            ys.append(tg.numpy())
            ps.append(out)
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)

if __name__ == "__main__":
    # 1. 加载并缓存所有单独的测试集
    npz_data = {
        name: load_or_cache(os.path.join(DATA_DIR, fname))
        for name, fname in TEST_CSV.items()
    }

    # 2. 为每个单独 split 构造 DataLoader
    loaders = {}
    for name, data in npz_data.items():
        ds = PridictDataset(data)
        loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. 合并 HEK293 测试集 (pridict1 + pridict2_HEK293)
    #    注意，这里用的是 TEST_CSV 中的两个 key："pridict1" 和 "pridict2_HEK293"
    data1 = npz_data["pridict1"]
    data2 = npz_data["pridict2_HEK293"]
    hek_npz = {
        "init_seqs": np.concatenate([data1["init_seqs"], data2["init_seqs"]], axis=0),
        "mut_seqs":  np.concatenate([data1["mut_seqs"],  data2["mut_seqs"]],  axis=0),
        "features":  np.concatenate([data1["features"],  data2["features"]],  axis=0),
        "targets":   np.concatenate([data1["targets"],   data2["targets"]],   axis=0),
    }
    # 为合并后的 HEK293 构造 DataLoader
    loaders["HEK293_merged"] = DataLoader(
        PridictDataset(hek_npz),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 4. 初始化模型并加载指定 checkpoint
    feat_dim = npz_data["pridict1"]["features"].shape[1]
    model = PRIDICT(
        feat_dim=feat_dim,
        nucl_dim=best_params['nucl_emb'],
        spec_dim=best_params['spec_emb'],
        hid=best_params['hidden_dim'],
        nl=best_params['num_layers'],
        dp=best_params['dropout']
    ).to(DEVICE)

    # 这里指明加载第 200 个 epoch 的权重
    ckpt_path = os.path.join(MODEL_DIR, "pridict_200.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded weights from {ckpt_path}")

    # 5. 对所有 loader（包括合并后的）进行预测并保存
    for name, loader in loaders.items():
        y_true, y_pred = collect_predictions(model, loader, DEVICE)

        out_dir = DATA_DIR
        np.save(os.path.join(out_dir, f"y_true_{name}.npy"), y_true)
        np.save(os.path.join(out_dir, f"y_pred_{name}.npy"), y_pred)
        print(f"[{name}] → 保存 y_true_{name}.npy ({y_true.shape}), y_pred_{name}.npy ({y_pred.shape})")

    print("全部推理完成。")
