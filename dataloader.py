import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config




# LOAD CSV
df = pd.read_csv(
    Config.DATA_PATH,
    header=None
)
df.columns = [
    "Speed",
    "Load",
    "Lambda",
    "Ignition Angle",
    "Fuel Cutoff",
    "Particle Numbers",
    "CO",
    "CO2",
    "HC",
    "Nox",
    "O2",
    "Temp Exhaust",
    "Temp Catalyst"
]

print(df.head())
print(df.columns)
print(df.shape)


# SELECT FEATURES
features = features = df.columns.tolist()


# CREATE INPUT MATRIX
X = df.values


# DATASET CLASS
class EngineDataset(Dataset):

    def __init__(self, data, tau=1):

        self.data = torch.tensor(
            data,
            dtype=torch.float32
        )

        self.tau = tau

    def __len__(self):

        return len(self.data) - self.tau

    def __getitem__(self, idx):

        x_t = self.data[idx]

        x_next = self.data[idx + self.tau]

        return x_t, x_next


# CREATE DATASET
dataset = EngineDataset(X)


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