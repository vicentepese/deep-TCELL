from torch import nn
from torch.nn import functional as F

class M_TCR_ACT(nn.Module):
    def __init__(self, config):
        """A model to process embeddings of CDR3 a+b data

        Args:
            config: An instance of the config class with set attributes.
        """
        super().__init__()
        self.config = config
        self.return_features = config.return_features
        self.embedding_size = config.embedding_size
        self.linear_1 = nn.Linear(self.embedding_size, 300)
        self.dropout = nn.Dropout(p = config.do)
        self.linear_2 = nn.Linear(300, 200)
        self.linear_3 = nn.Linear(200, 100)
        self.linear_class = nn.Linear(100, 3)

    def forward(self, X):
        X = F.relu(self.linear_1(X))
        X = self.dropout(X)
        X = F.relu(self.linear_2(X))
        X = F.relu(self.linear_3(X))
        C = self.linear_class(X)
        if self.return_features:
            return {'pred': C, 'feat': X}
        else:
            return {'pred': C}