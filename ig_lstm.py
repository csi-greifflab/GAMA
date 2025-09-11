"""
Implements a one-layer LSTM model that provides a sample function
and two functions that implement the integrated gradient.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

class LSTMModel(nn.Module):
    """
    This class implements an LSTM model.
    """
    def __init__(self, hidden_dim, device):
        super().__init__()
        self.device = device
        self.interpolation_steps = 500
        self.embedding_dim = 22
        self.embedding = nn.Embedding(num_embeddings=self.embedding_dim, embedding_dim=self.embedding_dim)
        self.embedding.weight = torch.nn.Parameter(torch.eye(self.embedding_dim, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = hidden_dim, batch_first=True)
        self.nn = nn.Linear(hidden_dim, self.embedding_dim)

    def forward(self, sequence):
        """
        Implements the forward pass of the LSTM model.

        Args:
        sequence: [torch LongTensor] input sequence

        Returns:
            tensor
        """
        embeds = self.embedding(sequence)
        lstm_out, _ = self.lstm(embeds.view(1, -1, self.embedding_dim))
        out = self.nn(lstm_out)
        return F.log_softmax(out, dim=2)

    def sample(self):
        """
        Sample a novel sequence with autoregressive forward sampling.


        Returns:
            [torch LongTensor]
        """
        with torch.no_grad():
            aa = []
            current_aa = torch.LongTensor([0]).view(1, 1).to(self.device)
            embed = self.embedding(current_aa)
            aa_embed, h = self.lstm(embed)
            while current_aa.tolist()[0][0] != 21:
                aa_logit = self.nn(aa_embed)
                current_aa = Categorical(logits=aa_logit).sample()
                aa.append(current_aa.tolist()[0][0])
                embed = self.embedding(current_aa)
                aa_embed, h = self.lstm(embed, h)
            return aa[0:-1]

    def ig_sample(self, peptide):
        """
        Compute the integrated gradient for an input sequence.

        Args:
        peptide: [torch LongTensor] sequence for which the IG is computed

        Returns:
            tensor
        """
        embeds = self.embedding(torch.LongTensor(peptide).to(self.device)).detach()
        baseline = torch.zeros_like(embeds).to(self.device)
        results = torch.zeros_like(embeds).to(self.device)
        for lin_step in torch.linspace(0.001, 1, self.interpolation_steps):
            self.zero_grad()
            polation = torch.lerp(baseline, embeds, lin_step.to(self.device)).to(self.device)
            polation.requires_grad = True
            lstm_out, _ = self.lstm(polation.view(1, -1, self.embedding_dim))
            out = self.nn(lstm_out)

            torch.sum(torch.diagonal(out[0,:,peptide])).backward()
            results[:, :] += polation.grad
        results[:, :] *= (embeds - baseline) / self.interpolation_steps
        return results

    def ig_sample_spc(self, peptide, output_pos_dim, diff_aa):
        """
        Compute the integrated gradient for an input sequence for a specific
        output position w.r.t. to the diff_aa position.

        Args:
        peptide: [torch LongTensor] sequence for which the IG is computed
        output_pos_dim: [int] index of the output dimension
        diff_aa: [int] index of the input dimension

        Returns:
            tensor
        """
        embeds = self.embedding(peptide).detach()
        baseline = torch.zeros_like(embeds).to(self.device)
        results = torch.zeros(*embeds.shape, output_pos_dim, diff_aa).to(self.device)
        for lin_step in torch.linspace(0.001, 1, self.interpolation_steps):
            self.zero_grad()
            polation = torch.lerp(baseline, embeds, lin_step.to(self.device)).to(self.device)
            polation.requires_grad = True
            lstm_out, _ = self.lstm(polation.view(1, -1, self.embedding_dim))
            out = self.nn(lstm_out)
            for output_pos_d in range(output_pos_dim):
                for aa_d in range(diff_aa):
                    self.zero_grad()
                    out[0, output_pos_d, aa_d].backward(retain_graph=True)
                    results[:, :, output_pos_d, aa_d] += polation.grad
            del out
            self.zero_grad(set_to_none=True)
        for output_pos_d in range(output_pos_dim):
            for aa_d in range(diff_aa):
                results[:, :, output_pos_d, aa_d] *= (embeds - baseline) / self.interpolation_steps
        return results
