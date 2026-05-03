import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config




# LOAD CSV
df = pd.read_csv(
    Config.DATA_PATH,
    header=None,
    usecols=[0, 1, 2, 3]
)
df.columns = [
    "Speed",
    "Load",
    "Ignition Angle",
    "Fuel Cutoff"
]

print(df.head())
print(df.columns)
print(df.shape)


# SELECT FEATURES
features = features = df.columns.tolist()


# CREATE INPUT MATRIX
X = df[features].values


# DATASET CLASS
class EngineDataset(Dataset):

    def __init__(self, data, seq_len):

        self.data = torch.tensor(
            data,
            dtype=torch.float32
        )

        self.seq_len = seq_len

    def __len__(self):

        return len(self.data) - self.seq_len

    def __getitem__(self, idx):

        x_seq = self.data[
            idx : idx + self.seq_len
        ]

        target = self.data[
            idx + self.seq_len
        ]

        return x_seq, target


# CREATE DATASET
dataset = EngineDataset(X, seq_len=Config.SEQ_LEN)


# CREATE DATALOADER
loader = DataLoader(
    dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True
)


# TEST
for x_t, x_next in loader:

    print(x_t.shape)
    print(x_next.shape)

    break