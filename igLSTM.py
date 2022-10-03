import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size, DEVICE):
        super().__init__()
        self.INTERPOLATION_STEPS = 500
        self.embedding_dim = 22
        self.embedding = nn.Embedding(num_embeddings=self.embedding_dim, embedding_dim=self.embedding_dim)
        self.embedding.weight = torch.nn.Parameter(torch.eye(self.embedding_dim, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = hidden_dim, batch_first=True)
        self.nn = nn.Linear(hidden_dim, self.embedding_dim)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(1, -1, self.embedding_dim))
        out = self.nn(lstm_out)
        return F.log_softmax(out, dim=2)

    def sample(self):
        aa = list()
        current_aa = torch.LongTensor([0]).view(1, 1).to(DEVICE)
        embed = self.embedding(current_aa)
        aa_embed, h = self.lstm(embed)
        while current_aa.tolist()[0][0] != 21:
            aa_logit = self.nn(aa_embed)
            current_aa = Categorical(logits=aa_logit).sample()
            aa.append(current_aa.tolist()[0][0])
            embed = self.embedding(current_aa)
            aa_embed, h = self.lstm(embed, h)
        return aa[0:-1]

    def IG_sample(self, peptide):
        embeds = self.embedding(torch.LongTensor(peptide).to(DEVICE)).detach()
        baseline = torch.zeros_like(embeds).to(DEVICE)
        results = torch.zeros_like(embeds).to(DEVICE)
        for lin_step in torch.linspace(0.001, 1, self.INTERPOLATION_STEPS):
            model.zero_grad()
            polation = torch.lerp(baseline, embeds, lin_step.to(DEVICE)).to(DEVICE)
            polation.requires_grad = True
            lstm_out, _ = self.lstm(polation.view(1, -1, self.embedding_dim))
            out = self.nn(lstm_out)
            
            torch.sum(torch.diagonal(out[0,:,peptide])).backward()
            results[:, :] += polation.grad
        results[:, :] *= (embeds - baseline) / self.INTERPOLATION_STEPS
        return results

    def IG_sample_spc(self, peptide, output_pos_d, aa_d):
        embeds = self.embedding(torch.LongTensor(peptide).to(DEVICE)).detach()
        baseline = torch.zeros_like(embeds).to(DEVICE)
        results = torch.zeros_like(embeds).to(DEVICE)
        for lin_step in torch.linspace(0.001, 1, self.INTERPOLATION_STEPS):
            model.zero_grad()
            polation = torch.lerp(baseline, embeds, lin_step.to(DEVICE)).to(DEVICE)
            polation.requires_grad = True
            lstm_out, _ = self.lstm(polation.view(1, -1, self.embedding_dim))
            out = self.nn(lstm_out)
            out[0 ,output_pos_d, aa_d].backward()
            results[:, :] += polation.grad
        results[:, :] *= (embeds - baseline) / self.INTERPOLATION_STEPS
        return results

