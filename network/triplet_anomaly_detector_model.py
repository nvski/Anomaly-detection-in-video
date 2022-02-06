from torch.nn import functional as F
from torch import nn
import torch


class TripletAnomalyDetector(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        output_dim=128,
        dropout_rate=0.5,
        use_last_bn=True,
        norm_out_to_unit=True,
        norm_on_eval=True,
    ):
        super(TripletAnomalyDetector, self).__init__()
        self.use_last_bn = use_last_bn
        self.norm_out_to_unit = norm_out_to_unit
        self.norm_on_eval = norm_on_eval
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, output_dim)

        if use_last_bn:
            self.bn_last = nn.BatchNorm1d(output_dim)

        # In the original keras code they use "glorot_normal"
        # As I understand, this is the same as xavier normal in Pytorch
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        if self.use_last_bn:
            x = self.bn_last(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        if self.norm_out_to_unit or (not self.training and self.norm_on_eval):
            x = F.normalize(x, p=2, dim=-1)
        return x

class TripletAnomalyDetectorWithProjection(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        output_dim=64,
        projection_dim=16,
        dropout_rate=0.5,
        use_last_bn=True,
        norm_out_to_unit=True,
        norm_on_eval=True,
    ):
        super(TripletAnomalyDetector, self).__init__()
        self.use_last_bn = use_last_bn
        self.norm_out_to_unit = norm_out_to_unit
        self.norm_on_eval = norm_on_eval
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(output_dim, projection_dim)

        if use_last_bn:
            self.bn_last = nn.BatchNorm1d(output_dim)

        # In the original keras code they use "glorot_normal"
        # As I understand, this is the same as xavier normal in Pytorch
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x, projection=True):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        if projection:
            x = self.dropout3(self.relu(x))
            x = self.proj(x)
        if self.use_last_bn:
            x = self.bn_last(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        if self.norm_out_to_unit or (not self.training and self.norm_on_eval):
            x = F.normalize(x, p=2, dim=-1)
        return x
