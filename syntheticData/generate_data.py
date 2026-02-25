"""
Generates a dataset of random integers between 1-20 simulating amino acids
and adds a '0' start and a '21' end token. And implants specified motifs.
Also resamples negative sequences if they have the motif by chance.
"""
import random

import torch


def generate_data(logic_op, sequence_length, signal_pos=(), signal_sequences_n=10_000, signal2noise=1.0, device='cpu', random_seed=0):
    """
    Generates a dataset of random integers between 1-20 simulating amino acids
    and adds a '0' start and a '21' end token. And implants specified motifs.
    Also resamples negative sequences if they have the motif by chance.

    Args:
    logic_op: string with ('AND', 'OR', 'XOR') indicating the motif interaction
    sequence_length: integer, length of generated sequences
    signal_pos: tupel of integers of length 2-4, giving the positions of motifs
    signal_sequences_n: int, number of sequences generated
    signal2noise: ratio of signal sequences to random sequences
    device: device the torch tensors should be on
    random_seed: int

    Returns:
        [int]: list of random long torch integers with implanted motifs
    """
    random.seed(random_seed)

    # negative indicates not signal (left: not 12, right not 10)
    logic_positive = []
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
            logic_positive = [[12, 10, 8, 6], [-12, 10, 8, 6], [12, -10, 8, 6], [12, 10, -8, 6], [12, 10, 8, -6], [-12, -10, 8, 6],\
            [12, -10, -8, 6], [12, 10, -8, -6], [-12, 10, 8, -6], [-12, 10, -8, 6], [12, -10, 8, -6], [-12, -10, -8, 6], [-12, -10, 8, -6], [-12, 10, -8, -6], [12, -10, -8, -6]]
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
            logic_positive = [[-12, 10, 8, 6], [12, -10, 8, 6], [12, 10, -8, 6], [12, 10, 8, -6], [-12, -10, 8, 6], [-12, 10, -8, 6],\
             [-12, 10, 8, -6], [12, -10, -8, 6], [12, -10, 8, -6], [12, 10, -8, -6], [-12, -10, -8, 6], [-12, -10, 8, -6], [-12, 10, -8, -6], [12, -10, -8, -6]]

    data = []
    # positive data instances
    for _ in range(signal_sequences_n):
        s = torch.LongTensor([0] + [random.randint(1, 20) for _ in range(sequence_length)] + [21]).to(device)
        logic_index = random.randrange(0, len(logic_positive))
        for pos_sequ, signal in zip(signal_pos, logic_positive[logic_index]):
            if signal >= 0:
                s[pos_sequ] = signal
        data.append(s)

    # negative data instances
    for _ in range(round(signal_sequences_n / signal2noise - signal_sequences_n)):
        contin = True
        while contin: # rejection sampling
            s = torch.LongTensor([0] + [random.randint(1, 20) for _ in range(sequence_length)] + [21]).to(device)
            checkpoint_bool_any = False
            for logic_pattern in logic_positive:
                checkpoint_bool_all = True
                for pos_sequ, signal in zip(signal_pos, logic_pattern):
                    if signal >= 0 and s[pos_sequ] != signal:
                        checkpoint_bool_all = False
                    if signal < 0 and s[pos_sequ] == abs(signal):
                        checkpoint_bool_all = False
                checkpoint_bool_any = checkpoint_bool_any or checkpoint_bool_all
            if not checkpoint_bool_any:
                contin = False
        data.append(s)
    return data
