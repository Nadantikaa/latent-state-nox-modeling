import pandas as pd
import matplotlib.pyplot as plt

from config import Config


df = pd.read_csv(
    Config.DATA_PATH,
    header=None
)

df.columns = [
    "Speed",
    "Load",
    "Ignition Angle",
    "Fuel Cutoff"
]


print(df.describe())

print(df.isnull().sum())


df["Nox"].plot()

plt.title("NOx Over Time")

plt.show()