import numpy as np
from scipy.stats import chisquare

def check_model(model, logic_op, sequence_length, signal_pos=()):
    # negative indicates not signal (left: not 12, right not 10)
    logic_positive = list()
    if logic_op == 'AND':
        if len(signal_pos) == 2:
            logic_positive = [[12, 10]]
        if len(signal_pos) == 3:
            logic_positive = [[12, 10, 8]]
        if len(signal_pos) == 4:
            logic_positive = [[12, 10, 8, 6]]
    if logic_op == 'OR':
        if len(signal_pos) == 2:
            logic_positive = [[12, 10], [12, -10], [-12, 10]]
        if len(signal_pos) == 3:
            logic_positive = [[12, 10, 8], [12, 10, -8], [12, -10, 8], [-12, 10, 8], [-12, -10, 8], [-12, 10, -8, [12, -10, -8]]]
        if len(signal_pos) == 4:
            logic_positive = [[12, 10, 8, 6], [-12, 10, 8, 6], [12, -10, 8, 6], [12, 10, -8, 6], [12, 10, 8, -6], [-12, -10, 8, 6], [12, -10, -8, 6], [12, 10, -8, -6], [-12, 10, 8, -6], [-12, 10, -8, 6], [12, -10, 8, -6], [-12, -10, -8, 6], [-12, -10, 8, -6], [-12, 10, -8, -6], [12, -10, -8, -6]]
    if logic_op == 'XOR':
        if len(signal_pos) == 2:
            logic_positive = [[12, -10], [-12, 10]]
        if len(signal_pos) == 3:
            logic_positive = [[-12, -10, 8], [-12, 10, -8], [12, -10, -8]]
        if len(signal_pos) == 4:
            logic_positive = [[-12, -10, -8, 6], [-12, -10, 8, -6], [-12, 10, -8, -6], [12, -10, -8, -6]]
    if logic_op == 'NAND':
        if len(signal_pos) == 2:
            logic_positive = [[-12, -10], [12, -10], [-12, 10]]
        if len(signal_pos) == 3:
            logic_positive = [[-12, -10, -8], [-12, 10, 8], [12, -10, 8], [12, 10, -8], [-12, -10, 8], [12, -10, -8], [-12, 10, -8]]
        if len(signal_pos) == 4:
            logic_positive = [[-12, 10, 8, 6], [12, -10, 8, 6], [12, 10, -8, 6], [12, 10, 8, -6], [-12, -10, 8, 6], [-12, 10, -8, 6], [-12, 10, 8, -6], [12, -10, -8, 6], [12, -10, 8, -6], [12, 10, -8, -6], [-12, -10, -8, 6], [-12, -10, 8, -6], [-12, 10, -8, -6], [12, -10, -8, -6]]


    # non_signal_counts = np.zeros((sequence_length - len(signal_pos), 20))
    for checkI in range(100):
        # compute s2n estimate of generated data
        s=[]
        t = 0
        while len(s) != sequence_length:
            s = model.sample()
            t += 1
            if t > 100:
                return -1
        s = [0] + s
        N_sig = 0
        N_non_sig = 0
        checkpoint_bool_any = False
        for logic_pattern in logic_positive:
            checkpoint_bool_all = True
            for pos_sequ, signal in zip(signal_pos, logic_pattern):
                if signal >= 0 and s[pos_sequ] != signal:
                    checkpoint_bool_all = False
                if signal < 0 and s[pos_sequ] == abs(signal):
                    checkpoint_bool_all = False
            checkpoint_bool_any = checkpoint_bool_any or checkpoint_bool_all
        if checkpoint_bool_any:
            N_sig =+ 1
        else:
            N_non_sig =+ 1

        # for pos in reversed(signal_pos):
        #     del s[pos]
        # for index, element in enumerate(s[1:]):
        #     non_signal_counts[index, element-1] += 1

    # compute chi square to uniformity of non singal regions 
    # chi = chisquare(non_signal_counts, axis=1)

    return N_sig / (N_sig + N_non_sig)