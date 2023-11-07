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
    model_path = 'chinese_L-12_H-768_A-12 2'
    print('model_path:', model_path)
    
    t1 = get_time()
    print('loading data...')
    max_len = 20000
    data = load_data(train_path, max_len=max_len)
    print('total data length:', len(data))
    t2 = get_time()
    h, m, s = cost_time(t1, t2)
    print("loading data cost %dh %dm %ds" % cost_time(t1, t2))

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print('device:', device)


    t1 = get_time()
    print('building dataset and vocab...')
    batch_size = 128
    min_freq = 2
    print('batch_size:', batch_size)
    print('min_freq:', min_freq)
    dataset, vocab = build_dataset(data, batch_size, device, min_freq)
    print('dataset length: ', dataset.get_length())
    t2 = get_time()
    h, m, s = cost_time(t1, t2)
    print("build dataset cost %dh %dm %ds" % cost_time(t1, t2))


    enc_input_dim = len(vocab.stoi['en'])
    # enc_emb_dim = 256
    # enc_hid_dim = 512
    enc_emb_dim = 128
    enc_hid_dim = 256
    enc_dropout = 0.5

    dec_input_dim = len(vocab.stoi['zh'])
    # dec_emb_dim = 256
    # dec_hid_dim = 512
    dec_emb_dim = 128
    dec_hid_dim = 256
    dec_dropout = 0.5

    print('input_dim:', enc_input_dim)
    print('output_dim:', dec_input_dim)

    encoder = Encoder(enc_input_dim, enc_emb_dim, enc_hid_dim, enc_dropout)
    attention = Attention(enc_hid_dim, dec_emb_dim)
    decoder = Decoder(attention, dec_input_dim, dec_emb_dim, dec_hid_dim, enc_hid_dim, dec_dropout)
    model = Model(encoder, decoder, dec_input_dim, enc_hid_dim, dec_hid_dim, device).to(device)
    
    
    if os.path.exists(model_path):
        print('model exist')
        model.load_state_dict(torch.load(model_path))
    else:
        print('model does not exist')
        
    loss = nn.CrossEntropyLoss(ignore_index = vocab.stoi['zh']['<pad>']).to(device)
    # lr = 1e-3
    lr = 0.5e-3
    print('learning rate:', lr)
    # optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    optimizer = optim.Adam(model.parameters(), lr = lr, amsgrad = True)

    global_start_time = get_time()
    pre_epoch_loss = 0
    for epoch in range(6000):
        start_time = get_time()
        epoch_loss = train(model, dataset, optimizer, loss, epoch)
        end_time = get_time()
        h, m, s = cost_time(start_time, end_time)
        print('<====================================================================================>')
        print("<epoch: %d over| epoch_loss: %10f | cost %dh %dm %ds " % (epoch, epoch_loss, h, m, s))
        h, m, s = cost_time(global_start_time, end_time)
        print("<epoch_loss change: %10f | total cost %dh %dm %ds "%(epoch_loss - pre_epoch_loss, h, m, s))
        print('<====================================================================================>')
        if epoch_loss - pre_epoch_loss < 0:
            torch.save(model.state_dict(), model_path)
        pre_epoch_loss = epoch_loss
