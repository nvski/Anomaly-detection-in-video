from torch.nn import functional as F
from torch import nn


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
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 512)
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
        x = self.dropout2(self.fc2(x))
        if self.use_last_bn:
            x = self.bn_last(self.fc3(x).permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        if self.norm_out_to_unit or (not self.training and self.norm_on_eval):
            x = F.normalize(x, p=2, dim=-1)
        return x
