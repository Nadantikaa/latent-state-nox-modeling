from model import GRUEncoder

from dataloader import loader


model = GRUEncoder()


for x_seq, target in loader:

    z = model(x_seq)

    print(z.shape)

    break