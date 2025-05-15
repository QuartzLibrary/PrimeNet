import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ast
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# --------- Dataset & Model Definitions ---------#
def seq_concat(seq_row, length=99):
    """Concatenate one-hot encoded WT and edited sequences into a (2, length, 4) array."""
    mapping = {'A':0,'C':1,'G':2,'T':3,'a':0,'c':1,'g':2,'t':3,'X':None,'x':None}
    wt = seq_row['wide_initial_target']
    ed = seq_row['wide_mutated_target']
    def one_hot(seq):
        arr = np.zeros((length, 4), dtype=np.float32)
        for i, c in enumerate(seq):
            idx = mapping.get(c)
            if idx is None:
                continue
            arr[i, idx] = 1.0
        return arr
    wt_oh = one_hot(wt)
    ed_oh = one_hot(ed)
    stacked = np.stack([wt_oh, ed_oh], axis=0)  # (2, length, 4)
    # scale to [-1, +1]
    return 2 * stacked - 1

class GeneDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.df = df
        self.features = df[[
            'PBS_length', 'RT_length', 'PBS-RT_length',
            'Tm1_PBS','Tm2_RT','Tm3_cDNA_PAM','Tm4_RT_cDNA','deltaTm',
            'GC_count_1','GC_content_1','GC_count_2','GC_content_2',
            'GC_count_3','GC_content_3',
            'MFE_1_pegRNA','MFE_2_nospacer','MFE_3_RT_PBS_PolyT',
            'MFE_4_spacer_only','MFE_5_spacer_scaf','DeepSpCas9_score'
        ]].values.astype(np.float32)
        self.targets = df[[
            'Validly_Edited','Unedited','Erroneously_Edited'
        ]].values.astype(np.float32)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_tensor = torch.tensor(
            seq_concat(row),
            dtype=torch.float32
        )  # shape: (2, 99, 4)
        feat_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        tgt_tensor  = torch.tensor(self.targets[idx], dtype=torch.float32)
        return seq_tensor, feat_tensor, tgt_tensor

class GeneInteractionModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_features=20, dropout=0.1):
        super().__init__()
        # sequence branch
        self.seq_processor = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=(2, 3), padding=(0,1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Flatten(start_dim=2),          # -> (B, 128, 99)
            nn.Conv1d(128, 108, kernel_size=3, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(2),                  # -> (B, 108, 49)
            nn.Conv1d(108, 108, kernel_size=3, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(2),                  # -> (B, 108, 24)
            nn.Conv1d(108, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AvgPool1d(2)                   # -> (B, 128, 12)
        )
        self.gru = nn.GRU(128, hidden_size, num_layers,
                          batch_first=True, bidirectional=True)
        self.seq_fc = nn.Linear(2 * hidden_size, 12)

        # feature branch
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128)
        )

        # head
        self.head = nn.Sequential(
            nn.BatchNorm1d(140),
            nn.Dropout(dropout),
            nn.Linear(140, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, seq_input, feature_input):
        x = self.seq_processor(seq_input)     # (B, 128, L)
        x = x.permute(0, 2, 1)                # (B, L, 128)
        _, h = self.gru(x)                    # h: (2, B, hidden_size)
        h_cat = torch.cat((h[-2], h[-1]), dim=1)  # (B, 2*hidden_size)
        seq_feat = self.seq_fc(h_cat)         # (B, 12)

        feat_emb = self.feature_processor(feature_input)  # (B, 128)

        combined = torch.cat([seq_feat, feat_emb], dim=1) # (B, 140)
        return self.head(combined)            # (B, 3)


# --------- Prediction & Saving ---------#
def collect_predictions(model, loader, device):
    ys, preds = [], []
    model.eval()
    with torch.no_grad():
        for seqs, feats, targets in loader:
            seqs  = seqs.to(device)
            feats = feats.to(device)
            out   = model(seqs, feats).cpu().numpy()
            ys.append(targets.numpy())
            preds.append(out)
    return np.concatenate(ys, axis=0), np.concatenate(preds, axis=0)

if __name__ == "__main__":
    # Paths
    BASE_DIR   = "deepprime"
    DATA_DIR   = os.path.join(BASE_DIR, "deepprime_data")
    MODEL_PATH = os.path.join(BASE_DIR, "model_pth", "deepprime_epoch49.pth")

    TEST_FILES = {
        "pridict1":         "metrics_merged_pridict1_mutated_test.csv",
        "pridict2_HEK293":  "metrics_merged_pridict2_HEK293_mutated_test.csv",
        "pridict2_K562":    "metrics_merged_pridict2_K562_mutated_test.csv"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型权重
    model = GeneInteractionModel(num_features=20).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded weights from {MODEL_PATH}")

    # 2. 单独对每个 split 做预测并保存
    results = {}  # 存放所有 y_true / y_pred
    for name, fname in TEST_FILES.items():
        csv_path = os.path.join(DATA_DIR, fname)
        ds = GeneDataset(csv_path)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        y_true, y_pred = collect_predictions(model, loader, device)
        results[name] = (y_true, y_pred)

        out_dir = os.path.dirname(csv_path)
        np.save(os.path.join(out_dir, f"y_true_{name}.npy"),  y_true)
        np.save(os.path.join(out_dir, f"y_pred_{name}.npy"),  y_pred)
        print(f"[{name}] → 保存 y_true_{name}.npy, y_pred_{name}.npy")

    # 3. 合并 HEK293 (pridict1 + pridict2_HEK293)
    #    从 results 中取出 numpy 数组然后拼接
    y_true_1, y_pred_1 = results["pridict1"]
    y_true_2, y_pred_2 = results["pridict2_HEK293"]

    y_true_HEK = np.concatenate([y_true_1, y_true_2], axis=0)
    y_pred_HEK = np.concatenate([y_pred_1, y_pred_2], axis=0)

    # 保存合并后的 HEK293 预测
    merged_dir = DATA_DIR  # 或者指定一个你想要的输出目录
    np.save(os.path.join(merged_dir, "y_true_HEK293.npy"), y_true_HEK)
    np.save(os.path.join(merged_dir, "y_pred_HEK293.npy"), y_pred_HEK)
    print(f"[HEK293 合并] → 保存 y_true_HEK293.npy ({y_true_HEK.shape}), y_pred_HEK293.npy ({y_pred_HEK.shape})")

    print("全部推理 & 保存 完成。")
