import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 128)
        self.conv5 = GCNConv(128, 128)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 16)
        self.lin3 = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
        )

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_max_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)

        return x


class GCN_2(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.conv4 = GCNConv(64, 32)
        self.conv5 = GCNConv(32, 32)
        self.conv6 = GCNConv(32, 16)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.lin1 = torch.nn.Linear(16, 16)
        self.lin2 = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
        )

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv6(x, edge_index, edge_weight))
        x = global_max_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x


class GCN_3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.conv2 = GCNConv(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
        )

        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight)))
        x = global_max_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x


class GCN_4(torch.nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        self.conv1 = GCNConv(1, 128)
        self.conv2 = GCNConv(128, 96)
        self.conv3 = GCNConv(96, 64)
        self.conv4 = GCNConv(64, 32)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.lin1 = torch.nn.Linear(32, 32)
        self.lin2 = torch.nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
        )

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = global_max_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x
