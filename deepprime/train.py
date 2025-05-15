import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import logging

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def preprocess_seq(seq_list):
    length = 99
    seq_onehot = np.zeros((len(seq_list), 1, length, 4), dtype=float)
    for l, seq in enumerate(seq_list):
        for i in range(length):
            if seq[i] in "Aa":
                seq_onehot[l, 0, i, 0] = 1
            elif seq[i] in "Cc":
                seq_onehot[l, 0, i, 1] = 1
            elif seq[i] in "Gg":
                seq_onehot[l, 0, i, 2] = 1
            elif seq[i] in "Tt":
                seq_onehot[l, 0, i, 3] = 1
            elif seq[i] in "Xx":
                continue
            else:
                print(f"Non-ATGC character: {seq[i]} at position {i} in sequence {seq}")
                sys.exit()
    return seq_onehot  # shape: (N, 1, 99, 4)


def seq_concat(data, col1='wide_initial_target', col2='wide_mutated_target'):
    wt = preprocess_seq([data[col1].iloc[0]])  # shape: (1, 1, 99, 4)
    ed = preprocess_seq([data[col2].iloc[0]])  # shape: (1, 1, 99, 4)
    seq = np.concatenate((wt, ed), axis=1)  # shape: (1, 2, 99, 4)
    seq = 2 * seq - 1
    return seq.squeeze(0)  # shape: (2, 99, 4)


class GeneDataset(Dataset):
    def __init__(self, dataframe, seq_cols=('wide_initial_target', 'wide_mutated_target'),
                 feature_cols=('PBS_length', 'RT_length', 'PBS-RT_length', 'Tm1_PBS', 'Tm2_RT',
                               'Tm3_cDNA_PAM', 'Tm4_RT_cDNA', 'deltaTm', 'GC_count_1', 'GC_content_1',
                               'GC_count_2', 'GC_content_2', 'GC_count_3', 'GC_content_3', 'MFE_1_pegRNA',
                               'MFE_2_nospacer', 'MFE_3_RT_PBS_PolyT', 'MFE_4_spacer_only',
                               'MFE_5_spacer_scaf', 'DeepSpCas9_score'),
                 target_cols=('Validly_Edited', 'Unedited', 'Erroneously_Edited')):
        self.seq_data = dataframe[list(seq_cols)]
        self.features = dataframe[list(feature_cols)].values.astype(np.float32)
        self.targets = dataframe[list(target_cols)].values.astype(np.float32)

        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        seq_row = self.seq_data.iloc[[idx]]
        seq_tensor = torch.tensor(seq_concat(seq_row), dtype=torch.float32)  # shape: (2, 99, 4)
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return seq_tensor, features, targets


class GeneInteractionModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, num_features=20, dropout=0.1):
        super().__init__()

        self.seq_processor = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=(2, 3), padding=(0, 1)),  # input: (B, 2, 99, 4)
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Flatten(start_dim=2),  # shape: (B, 128, 99*1) => (B, 128, 99)
            nn.Conv1d(128, 108, kernel_size=3, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(108, 108, kernel_size=3, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(108, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AvgPool1d(2)
        )

        self.gru = nn.GRU(128, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.seq_fc = nn.Linear(2 * hidden_size, 12)

        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(140),
            nn.Dropout(dropout),
            nn.Linear(140, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, seq_input, feature_input):
        x_seq = self.seq_processor(seq_input)  # (B, 128, L)
        x_seq = x_seq.permute(0, 2, 1)  # [B, L, 128]
        _, h_seq = self.gru(x_seq)
        seq_features = self.seq_fc(torch.cat((h_seq[-2], h_seq[-1]), dim=1))  # (B, 12)

        feature_emb = self.feature_processor(feature_input)  # (B, 128)

        combined = torch.cat([seq_features, feature_emb], dim=1)  # (B, 140)
        return self.head(combined)  # (B, 3)


def train_model(model, train_loader, val_loaders, epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seqs, features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            seqs, features, targets = seqs.to(device), features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(seqs, features)              # (B, 3)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = {}
        val_pearson = {}
        val_spearman = {}

        with torch.no_grad():
            for name, loader in val_loaders.items():
                all_preds = []
                all_trues = []
                for seqs, features, targets in loader:
                    seqs, features, targets = seqs.to(device), features.to(device), targets.to(device)
                    preds = model(seqs, features)
                    all_preds.append(preds.cpu().numpy())
                    all_trues.append(targets.cpu().numpy())

                preds = np.concatenate(all_preds, axis=0)    # (N, 3)
                trues = np.concatenate(all_trues, axis=0)    # (N, 3)

                # MSELoss 平均 loss
                mse = ((preds - trues) ** 2).mean()
                val_loss[name] = mse

                # 对每个输出维度分别计算 Pearson 和 Spearman
                pearson_per_dim = []
                spearman_per_dim = []
                for i in range(preds.shape[1]):
                    p, _ = pearsonr(preds[:, i], trues[:, i])
                    s, _ = spearmanr(preds[:, i], trues[:, i])
                    pearson_per_dim.append(p)
                    spearman_per_dim.append(s)
                val_pearson[name] = pearson_per_dim
                val_spearman[name] = spearman_per_dim

        # Logging
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        logging.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"deepprime/model_pth/deepprime_epoch{epoch}.pth")

        for name in val_loaders:
            ps = val_pearson[name]
            ss = val_spearman[name]
            logging.info(f"{name} MSE: {val_loss[name]:.4f}")
            logging.info(
                f"{name} Pearson per-dim: [{ps[0]:.4f}, {ps[1]:.4f}, {ps[2]:.4f}], "
                f"Spearman per-dim: [{ss[0]:.4f}, {ss[1]:.4f}, {ss[2]:.4f}]"
            )


def test_model(model, test_loaders):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_mae = {}
    test_pearson = {}
    test_spearman = {}

    with torch.no_grad():
        for name, loader in test_loaders.items():
            all_preds = []
            all_trues = []
            for seqs, features, targets in loader:
                seqs, features = seqs.to(device), features.to(device)
                preds = model(seqs, features).cpu().numpy()
                all_preds.append(preds)
                all_trues.append(targets.numpy())

            preds = np.concatenate(all_preds, axis=0)    # (N, 3)
            trues = np.concatenate(all_trues, axis=0)    # (N, 3)

            # MAE per-dim
            mae_per_dim = np.mean(np.abs(preds - trues), axis=0).tolist()
            test_mae[name] = mae_per_dim

            # Pearson & Spearman per-dim
            pearson_per_dim = []
            spearman_per_dim = []
            for i in range(preds.shape[1]):
                p, _ = pearsonr(preds[:, i], trues[:, i])
                s, _ = spearmanr(preds[:, i], trues[:, i])
                pearson_per_dim.append(p)
                spearman_per_dim.append(s)
            test_pearson[name] = pearson_per_dim
            test_spearman[name] = spearman_per_dim

    # Logging
    logging.info("\nTest Results:")
    for name in test_loaders:
        mae = test_mae[name]
        ps = test_pearson[name]
        ss = test_spearman[name]
        logging.info(
            f"{name} MAE per-dim: [{mae[0]:.4f}, {mae[1]:.4f}, {mae[2]:.4f}], "
            f"Pearson per-dim: [{ps[0]:.4f}, {ps[1]:.4f}, {ps[2]:.4f}], "
            f"Spearman per-dim: [{ss[0]:.4f}, {ss[1]:.4f}, {ss[2]:.4f}]"
        )

if __name__ == "__main__":
    train_df = pd.read_csv("deepprime/deepprime_data/metrics_merged_all_train.csv")
    train_dataset = GeneDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    datasets = {
        'val_HEK293_pridict2': 'metrics_merged_pridict2_HEK293_mutated_val.csv',
        'test_HEK293_pridict2': 'metrics_merged_pridict2_HEK293_mutated_test.csv',
        'val_K562_pridict2': 'metrics_merged_pridict2_K562_mutated_val.csv',
        'test_K562_pridict2': 'metrics_merged_pridict2_K562_mutated_test.csv',
        'val_pridict1': 'metrics_merged_pridict1_mutated_val.csv',
        'test_pridict1': 'metrics_merged_pridict1_mutated_test.csv'
    }

    val_loaders = {}
    test_loaders = {}
    for name, path in datasets.items():
        df = pd.read_csv(f"deepprime/deepprime_data/{path}")
        dataset = GeneDataset(df)
        loader = DataLoader(dataset, batch_size=32)
        if 'val' in name:
            val_loaders[name] = loader
        else:
            test_loaders[name] = loader

    model = GeneInteractionModel(num_features=20)
    train_model(model, train_loader, val_loaders, epochs=200)
    test_model(model, test_loaders)