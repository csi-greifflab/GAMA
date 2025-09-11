"""
Function can run any experimental condition from the synthetic dataset.
"""

import os
import sys

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

from generate_data import generate_data
sys.path.append('..')
from ig_lstm import LSTMModel




def experiment_run(logic_op, sequence_length, signal_pos=(), signal_sequences_n=10_000,\
                   ig_sequences_n=1000, signal2noise=1.0, device='cpu', prj_path='', seed_manual=0):
    """
    Function can run any experimental condition from the synthetic dataset.

    Args:
    logic_op: string with ('AND', 'OR', 'XOR') indicating the motif interaction
    sequence_length: integer, length of generated sequences
    signal_pos: tupel of integers of length 2-4, giving the positions of motifs
    signal_sequences_n: int, number of sequences generated
    ig_sequences_n: int, number of sequences for which GAMA should be calculated
    signal2noise: ratio of signal sequences to random sequences
    device: device the experiment should be run on
    prj_path: path to where the results will be stored
    seed_manual: int
    """
    torch.manual_seed(seed_manual)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not os.path.exists(f'{prj_path}/models'):
        os.mkdir(f'{prj_path}/models')
        train_list = generate_data(logic_op=logic_op, sequence_length=sequence_length,\
            signal_pos=signal_pos, signal_sequences_n=signal_sequences_n, signal2noise=signal2noise, device=device)

        model = LSTMModel(1024, device).to(device)
        torch.save(model.state_dict(), f'{prj_path}/models/Init_model.pt')
        optimizer = optim.Adam(model.parameters(), lr=0.00008)
        loss_function = nn.NLLLoss().to(device)
        train_data = DataLoader(train_list, shuffle=True, batch_size=None)

        train_loss = []
        for epoch in range(3000):
            loss_o = 0
            for sentence in train_data:
                model.zero_grad(set_to_none=True)
                tag_scores = model(sentence[0:-1])
                loss = loss_function(tag_scores.view(-1, 22), sentence[1:])
                loss.backward()
                optimizer.step()
                loss_o += loss.detach().tolist()
            train_loss.append(loss_o / len(train_list))
            print(f'{epoch=}, {loss_o=}')
        torch.save(model.state_dict(), f'{prj_path}/models/lstm_epoch_{epoch}_{loss_o}.pt')

        ig_sequences_n_pos = int(ig_sequences_n * signal2noise)
        ig_sequences_n_neg = int(ig_sequences_n * (1 - signal2noise))
        ig_test_sequ = [torch.LongTensor(i.to('cpu')).to(device)\
        for i in train_list[0:ig_sequences_n_pos]+train_list[signal_sequences_n:signal_sequences_n+ig_sequences_n_neg]]

        save = [i.tolist() for i in ig_test_sequ]
        save_df = pd.DataFrame(save)
        save_df.to_csv(f'{prj_path}/training_sequences.csv', index=False)
    else:
        df = pd.read_csv(f'{prj_path}/training_sequences.csv')
        ig_test_sequ = [torch.LongTensor(i[1].tolist()).to(device) for i in df.iterrows()]

    model_paths = os.listdir(f'{prj_path}/models')
    if len(model_paths) != 2:
        raise RuntimeWarning('incorrect amount of trained models!')
    for path in model_paths:
        if os.path.exists(f'{prj_path}/ig_matrix_{path}'):
            continue
        try:
            del model
        except:
            pass
        model = LSTMModel(1024, device).to(device)
        model.load_state_dict(torch.load(f'{prj_path}/models/{path}', map_location=device))
        input_pos_dim = len(ig_test_sequ[0])
        encoding_dim = 22
        output_pos_dim = len(ig_test_sequ[0])
        diff_aa = 22
        hycube_ig = torch.zeros((input_pos_dim, encoding_dim, output_pos_dim, diff_aa, len(ig_test_sequ)))
        for sequ_idx, sequ in enumerate(ig_test_sequ):
            print(sequ_idx)
            hycube_ig[:, :, :, :, sequ_idx] = model.ig_sample_spc(peptide=sequ, output_pos_dim=output_pos_dim, diff_aa=diff_aa)
        torch.save(hycube_ig, f'{prj_path}/ig_matrix_{path}')
        break

if __name__ == '__main__':
    LOGIC_OP='AND'
    SEQUENCE_LENGTH=5
    SIGNAL_POS=(1, 3)
    SIGNAL_SEQUENCES_N=5_000
    IG_SEQUENCES_N=150
    SIGNAL2NOISE=1
    DEVICE='cuda:0'
    PRJ_PATH='./testrun_IGfunc4'

    experiment_run(logic_op=LOGIC_OP, sequence_length=SEQUENCE_LENGTH, signal_pos=SIGNAL_POS,\
                   signal_sequences_n=SIGNAL_SEQUENCES_N, ig_sequences_n=IG_SEQUENCES_N,\
                   signal2noise=SIGNAL2NOISE, device=DEVICE, prj_path=PRJ_PATH)
