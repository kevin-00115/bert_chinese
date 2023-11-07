import torch
import os
import torch.nn as nn
import torch.optim as optim

from attention import Attention
from decoder import Decoder
from encoder import Encoder
from model import Model

from util import build_dataset, get_time, cost_time, load_data, train_path
from execute import train



if __name__ == '__main__':
    print('train by single gpu or cpu')
    
    t1 = get_time()
    print('loading data...')
    max_len = None
    data = load_data(train_path, max_len=max_len)
    print('total data length:', len(data))
    t2 = get_time()
    h, m, s = cost_time(t1, t2)
    print("loading data cost %dh %dm %ds" % cost_time(t1, t2))

    device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
    print('device:', device)


    t1 = get_time()
    print('building dataset and vocab...')
    batch_size = 8
    min_freq = 48
    print('min_freq:', min_freq)
    dataset, vocab = build_dataset(data, batch_size, device, min_freq)
    print('dataset length: ', dataset.get_length())
    t2 = get_time()
    h, m, s = cost_time(t1, t2)
    print("build dataset cost %dh %dm %ds" % cost_time(t1, t2))


    enc_input_dim = len(vocab.stoi['en'])
    enc_emb_dim = 256
    enc_hid_dim = 512
    enc_dropout = 0.5

    dec_input_dim = len(vocab.stoi['zh'])
    dec_emb_dim = 256
    dec_hid_dim = 512
    dec_dropout = 0.5

    print('input_dim:', enc_input_dim)
    print('output_dim:', dec_input_dim)