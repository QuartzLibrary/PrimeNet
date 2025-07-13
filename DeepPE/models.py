import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ========== 模型定义 ==========
class BestDeepPEModel(nn.Module):
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