import torch
import torch.nn as nn

from config import Config


class GRUEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # GRU
        self.gru = nn.GRU(

            input_size=Config.INPUT_DIM,

            hidden_size=Config.HIDDEN_DIM,

            num_layers=Config.NUM_LAYERS,

            batch_first=True
        )

        # MLP Projection
        self.projection = nn.Sequential(

            nn.Linear(
                Config.HIDDEN_DIM,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                Config.LATENT_DIM
            )
        )

    def forward(self, x):

        """
        x shape:
        (batch_size, seq_len, input_dim)
        """

        output, h_n = self.gru(x)

        """
        h_n shape:
        (num_layers, batch_size, hidden_dim)
        """

        # top GRU layer hidden state
        final_hidden = h_n[-1]

        # latent thermodynamic state
        z = self.projection(final_hidden)

        return z