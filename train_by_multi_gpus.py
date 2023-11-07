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
    print('train by multi gpu')
    model_path = 'result/modelV3.pkl'
    print('model_path:', model_path)
    
    t1 = get_time()
    print('loading data...')
    max_len = None
    data = load_data(train_path, max_len=max_len)
    print('total data length:', len(data))
    t2 = get_time()
    h, m, s = cost_time(t1, t2)
    print("loading data cost %dh %dm %ds" % cost_time(t1, t2))

    main_device = torch.device('cuda:0')
    print('main_device:', main_device)

    device_ids = [0, 1, 2]
    print('device_ids', device_ids)

    device_count = len(device_ids)

    batch_size = device_count*20
    print('batch_size', batch_size)



    t1 = get_time()
    print('building dataset and vocab...')
    min_freq = 32
    print('batch_size:', batch_size)
    print('min_freq:', min_freq)
    dataset, vocab = build_dataset(data, batch_size, main_device, min_freq)
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

    encoder = Encoder(enc_input_dim, enc_emb_dim, enc_hid_dim, enc_dropout)
    attention = Attention(enc_hid_dim, dec_emb_dim)
    decoder = Decoder(attention, dec_input_dim, dec_emb_dim, dec_hid_dim, enc_hid_dim, dec_dropout)
    model = Model(encoder, decoder, dec_input_dim, enc_hid_dim, dec_hid_dim, main_device).to(main_device)
    model = nn.DataParallel(model, device_ids = device_ids, output_device = main_device)
    

    if os.path.exists(model_path):
        print('model exist')
        model.load_state_dict(torch.load(model_path))
    else:
        print('model does not exist')
        
    loss = nn.CrossEntropyLoss(ignore_index = vocab.stoi['zh']['<pad>']).to(main_device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    global_start_time = get_time()
    for epoch in range(10):
        start_time = get_time()
        epoch_loss = train(model, dataset, optimizer, loss, epoch)
        end_time = get_time()
        h, m, s = cost_time(start_time, end_time)
        print('<====================================================================================>')
        print("<epoch: %d over| epoch_loss: %10f | cost %dh %dm %ds " % (epoch, epoch_loss, h, m, s))
        print("<total cost %dh %dm %ds "%cost_time(global_start_time, end_time))
        print('<====================================================================================>')
        torch.save(model.state_dict(), model_path)

