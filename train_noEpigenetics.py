import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from model_noEpigenetics import PrimeNet
import ast

def generate_synthetic_image(dna_seq, protospacerlocation=None, RT_initial_location=None,
                             PBSlocation=None, RT_mutated_location=None, target_length=128):
    img = np.zeros((target_length, 6, 1))
    for i, base in enumerate(dna_seq):
        if i >= target_length: break
        if base == 'A': img[i, 0, 0] = 1
        elif base == 'G': img[i, 1, 0] = 1
        elif base == 'T': img[i, 2, 0] = 1
        elif base == 'C': img[i, 3, 0] = 1
    def mark(loc, ch):
        loc = ast.literal_eval(loc)
        for j in range(loc[0] - 1, loc[1]):
            if 0 <= j < target_length:
                img[j, ch, 0] = 1
    if protospacerlocation: mark(protospacerlocation, 4)
    if PBSlocation:          mark(PBSlocation, 4)
    if RT_initial_location:  mark(RT_initial_location, 5)
    if RT_mutated_location:  mark(RT_mutated_location, 5)
    return img

class SequenceDataset(Dataset):
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = generate_synthetic_image(
            dna_seq=row['wide_initial_target'],
            protospacerlocation=row['protospacerlocation_only_initial'],
            RT_initial_location=row['RT_initial_location']
        )
        img2 = generate_synthetic_image(
            dna_seq=row['wide_mutated_target'],
            PBSlocation=row['PBSlocation'],
            RT_mutated_location=row['RT_mutated_location']
        )
        combined = np.concatenate((img1, img2), axis=2)
        tensor = torch.tensor(combined, dtype=torch.float32).permute(1, 0, 2)
        target = torch.tensor([
            row['Validly_Edited'],
            row['Unedited'],
            row['Erroneously_Edited']
        ], dtype=torch.float32)
        return tensor, target

def orthogonal_init(layer):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

def evaluate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    res = {}
    for i, name in enumerate(['Validly_Edited','Unedited','Erroneously_Edited']):
        mse = mean_squared_error(y_true[:,i], y_pred[:,i])
        mae = mean_absolute_error(y_true[:,i], y_pred[:,i])
        r2  = r2_score(y_true[:,i], y_pred[:,i])
        sp  = spearmanr(y_true[:,i], y_pred[:,i]).correlation
        res[name] = {'MSE':mse,'MAE':mae,'R2':r2,'Spearman':sp}
    return res

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PrimeNet().to(device)
model.apply(orthogonal_init)

class Lookahead(optim.Optimizer):
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        if not 0.0 <= alpha <= 1.0: raise ValueError
        if k < 1: raise ValueError
        self.optimizer = base_optimizer
        self.k, self.alpha, self.counter = k, alpha, 0
        self.param_groups = self.optimizer.param_groups
        self.slow_weights = [[p.clone().detach() for p in g['params']]
                             for g in self.param_groups]
        for w in self.slow_weights:
            for p in w: p.requires_grad=False
        super().__init__(self.param_groups, {'k':k,'alpha':alpha})

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.counter += 1
        if self.counter % self.k == 0:
            base_state = {p:(self.optimizer.state[p]['exp_avg'],
                              self.optimizer.state[p]['exp_avg_sq'])
                          for p in self.param_groups[0]['params']}
            for group, slow in zip(self.param_groups, self.slow_weights):
                for p, q in zip(group['params'], slow):
                    q.data.add_(p.data - q.data, alpha=self.alpha)
                    p.data.copy_(q.data)
            for p in self.param_groups[0]['params']:
                self.optimizer.state[p]['exp_avg'], \
                self.optimizer.state[p]['exp_avg_sq'] = base_state[p]
        return loss

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

train_df = pd.read_csv("data/merged_data_all.csv")
train_loader = DataLoader(SequenceDataset(train_df), batch_size=32, shuffle=True, num_workers=2)

base_opt = optim.Adam(model.parameters(), lr=8e-4)
optimizer = Lookahead(base_opt, k=5, alpha=0.666)
criterion = nn.MSELoss()

for epoch in range(200):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/200, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "PrimeNet/PrimeNet_noEpigenetics.pth")
print("Saved to PrimeNet/PrimeNet_noEpigenetics.pth")