import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr

# --------- Paths & Logging Setup ---------#
BASE_DIR = '/root/project'
DATA_DIR = os.path.join(BASE_DIR, 'DeepPE', 'DeepPE_data')
LOG_DIR = os.path.join(BASE_DIR, 'DeepPE', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
log_file = os.path.join(LOG_DIR, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# --------- Dataset ---------#
class PrimeEditingDataset(Dataset):
    def __init__(self, csv_path, length=99):
        self.df = pd.read_csv(csv_path)
        self.length = length
        self.wide = self.df['wide_initial_target'].values
        self.wide_mut = self.df['wide_mutated_target'].values
        self.proto_loc = self.df['protospacerlocation_only_initial'].apply(ast.literal_eval).values
        num_cols = [
            'PBS_length','RT_length','PBS-RT_length',
            'Tm1_PBS','Tm2_RT','Tm3_cDNA_PAM','Tm4_RT_cDNA','deltaTm',
            'GC_count_1','GC_count_2','GC_count_3',
            'GC_content_1','GC_content_2','GC_content_3',
            'MFE_1_pegRNA','MFE_2_nospacer','MFE_3_RT_PBS_PolyT',
            'MFE_4_spacer_only','MFE_5_spacer_scaf','DeepSpCas9_score'
        ]
        self.features = self.df[num_cols].astype(float).values
        # 三个目标列
        self.targets = self.df[['Validly_Edited','Unedited','Erroneously_Edited']].astype(float).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wide = self.wide[idx]
        loc = self.proto_loc[idx]
        seq = wide[loc[0]:loc[1]]
        mod_seq = self.wide_mut[idx][loc[0]:loc[1]]
        X = self.one_hot(seq)
        X_mod = self.one_hot(mod_seq)
        bio = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return X, X_mod, bio, y

    def one_hot(self, seq):
        mapping = {'A':0,'C':1,'G':2,'T':3}
        arr = np.zeros((4, self.length), dtype=np.float32)
        for i, c in enumerate(seq.upper()):
            if c in mapping and i < self.length:
                arr[mapping[c], i] = 1
        return torch.from_numpy(arr)

# --------- Model ---------#
class DeepPEModel(nn.Module):
    def __init__(self, filter_size, filter_num, length=99, node1=80, node2=60, bio_num=20, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(4, filter_num, kernel_size=filter_size)
        self.conv_mod = nn.Conv1d(4, filter_num, kernel_size=filter_size)
        conv_len = length - filter_size + 1
        flat_dim = conv_len * filter_num
        total_in = flat_dim*2 + bio_num
        self.fc1 = nn.Linear(total_in, node1)
        self.fc2 = nn.Linear(node1, node2) if node2>0 else None
        # 输出 3 维
        self.out = nn.Linear(node2 if node2>0 else node1, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mod, bio):
        h1 = F.relu(self.conv(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.conv_mod(x_mod))
        h2 = self.dropout(h2)
        h1 = h1.view(h1.size(0), -1)
        h2 = h2.view(h2.size(0), -1)
        h = torch.cat([h1, h2, bio], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        if self.fc2:
            h = F.relu(self.fc2(h))
            h = self.dropout(h)
        out = self.out(h)
        return out

# --------- Metrics Evaluation ---------#
def evaluate_metrics(model, dataloader, device):
    model.eval()
    ys, preds = [], []
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for X, X_mod, bio, y in dataloader:
            X, X_mod, bio, y = X.to(device), X_mod.to(device), bio.to(device), y.to(device)
            pred = model(X, X_mod, bio)
            total_loss += criterion(pred, y).item() * X.size(0)
            ys.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    mse = total_loss / len(dataloader.dataset)
    # 对每个输出维度计算相关系数和 mse
    pearsons, spearmans, mses = [], [], []
    for i in range(3):
        mses.append(((y_true[:,i] - y_pred[:,i])**2).mean())
        pearsons.append(pearsonr(y_true[:,i], y_pred[:,i])[0])
        spearmans.append(spearmanr(y_true[:,i], y_pred[:,i])[0])
    return mses, pearsons, spearmans

# --------- Training Loop with Val & Test ---------#
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_csv = os.path.join(DATA_DIR, 'metrics_merged_all_train.csv')
    val_files = [
        'metrics_merged_pridict1_mutated_val.csv',
        'metrics_merged_pridict2_HEK293_mutated_val.csv',
        'metrics_merged_pridict2_K562_mutated_val.csv'
    ]
    test_files = [
        'metrics_merged_pridict1_mutated_test.csv',
        'metrics_merged_pridict2_HEK293_mutated_test.csv',
        'metrics_merged_pridict2_K562_mutated_test.csv'
    ]

    train_loader = DataLoader(PrimeEditingDataset(train_csv), batch_size=64, shuffle=True)
    val_loaders = {vf: DataLoader(PrimeEditingDataset(os.path.join(DATA_DIR, vf)), batch_size=64)
                   for vf in val_files}
    test_loaders = {tf: DataLoader(PrimeEditingDataset(os.path.join(DATA_DIR, tf)), batch_size=64)
                    for tf in test_files}

    model = DeepPEModel(filter_size=1, filter_num=80, length=99).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = 20

    logger.info('Starting training...')
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for X, X_mod, bio, y in train_loader:
            X, X_mod, bio, y = X.to(device), X_mod.to(device), bio.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X, X_mod, bio)
            loss = nn.MSELoss()(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        logger.info(f'Epoch {epoch}/{epochs} - Train MSE (avg over dims): {train_loss:.4f}')
        for name, loader in val_loaders.items():
            mses, pears, spears = evaluate_metrics(model, loader, device)
            logger.info(f'    Val ({name}):')
            for i, label in enumerate(['Validly_Edited','Unedited','Erroneously_Edited']):
                logger.info(f'        {label} - MSE: {mses[i]:.4f}, Pearson: {pears[i]:.4f}, Spearman: {spears[i]:.4f}')

    logger.info('Starting testing...')
    for name, loader in test_loaders.items():
        mses, pears, spears = evaluate_metrics(model, loader, device)
        logger.info(f'Test ({name}):')
        for i, label in enumerate(['Validly_Edited','Unedited','Erroneously_Edited']):
            logger.info(f'    {label} - MSE: {mses[i]:.4f}, Pearson: {pears[i]:.4f}, Spearman: {spears[i]:.4f}')

    model_path = os.path.join(BASE_DIR, 'DeepPE', 'deeppe_model.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f'Model saved to {model_path}')
