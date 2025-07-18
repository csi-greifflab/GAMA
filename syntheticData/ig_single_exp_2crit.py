import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import pandas as pd

from generate_data import generate_data
from check_model import check_model
from igLSTM import LSTMModel




def experiment_run(logic_op, sequence_length, signal_pos=(), signal_sequences_n=10_000,\
                   ig_sequences_n=1000, signal2noise=1.0, DEVICE='cpu', prj_path='', seed_manual=0):
    torch.manual_seed(seed_manual)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not os.path.exists(f'{prj_path}/models'):
        os.mkdir(f'{prj_path}/models')
        train_list = generate_data(logic_op=logic_op, sequence_length=sequence_length, signal_pos=signal_pos, signal_sequences_n=signal_sequences_n, signal2noise=signal2noise, DEVICE=DEVICE)

        model = LSTMModel(1024, 22, DEVICE).to(DEVICE)
        torch.save(model.state_dict(), f'{prj_path}/models/Init_model.pt')
        optimizer = optim.Adam(model.parameters(), lr=0.00008)
        loss_function = nn.NLLLoss().to(DEVICE)
        train_data = DataLoader(train_list, shuffle=True, batch_size=None)

        train_loss = list()
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
        ig_test_sequ = [torch.LongTensor(i.to('cpu')).to(DEVICE) for i in train_list[0:ig_sequences_n_pos]+train_list[signal_sequences_n:signal_sequences_n+ig_sequences_n_neg]]

        save = [i.tolist() for i in ig_test_sequ]
        save_df = pd.DataFrame(save)
        save_df.to_csv(f'{prj_path}/training_sequences.csv', index=False)
    else:
        df = pd.read_csv(f'{prj_path}/training_sequences.csv')
        ig_test_sequ = [torch.LongTensor(i[1].tolist()).to(DEVICE) for i in df.iterrows()]

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
        model = LSTMModel(1024, 22, DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(f'{prj_path}/models/{path}', map_location=DEVICE))
        input_pos_dim = len(ig_test_sequ[0])
        encoding_dim = 22
        output_pos_dim = len(ig_test_sequ[0])
        diff_aa = 22
        hycube_ig = torch.zeros((input_pos_dim, encoding_dim, output_pos_dim, diff_aa, len(ig_test_sequ)))
        for sequ_idx, sequ in enumerate(ig_test_sequ):
            print(sequ_idx)
            hycube_ig[:, :, :, :, sequ_idx] = model.IG_sample_spc(peptide=sequ, output_pos_dim=output_pos_dim, diff_aa=diff_aa)
        torch.save(hycube_ig, f'{prj_path}/ig_matrix_{path}')
        break

if __name__ == '__main__':
    logic_op='AND'
    sequence_length=5
    signal_pos=(1, 3)
    signal_sequences_n=5_000
    ig_sequences_n=150
    signal2noise=1
    DEVICE='cuda:0'
    prj_path='./testrun_IGfunc4'

    experiment_run(logic_op=logic_op, sequence_length=sequence_length, signal_pos=signal_pos,\
                   signal_sequences_n=signal_sequences_n, ig_sequences_n=ig_sequences_n,\
                   signal2noise=signal2noise, DEVICE=DEVICE, prj_path=prj_path)