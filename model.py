from filecmp import cmp
import torch
import random
import torch.nn as nn
from attention import Attention
from decoder import Decoder
from encoder import Encoder

class Model(nn.Module):
    def __init__(self, encoder, decoder, target_dim, enc_hid_dim, dec_hid_dim, device, sos = 0, eos = 1, pad = 3):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_dim = target_dim
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.device = device
        self.enc_hid_fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)
    
    # ori, trg [batch_size, seq_len]
    def forward(self, ori, trg = None, teacher_force_ratio = 0.6):
        # enc_output [batch_size, seq_len, hid_dim*2]
        # enc_hid [batch_size, 2, hid_dim]
        enc_output, enc_hid = self.encoder(ori)
        batch_size = ori.size(0)
        #output [seq_len, batch_size, output dim]
        output = torch.randn(1, batch_size, self.target_dim).to(self.device)
        #output_idx [seq_len, batch_size, 1]
        output_idx = torch.randn(1, batch_size, 1).long().to(self.device)
        # [batch_size, 1], init [[0],[0]]
        dec_input = torch.LongTensor([self.sos]*batch_size).unsqueeze(1).to(self.device)
        output_idx[0] = dec_input
        # [1, batch_size, dec_hid_dim]
        dec_hid = self.enc_hid_fc(torch.cat((enc_hid[0,:,:], enc_hid[1,:,:]), 1)).unsqueeze(0)

        idx = 1
        while self.__finish(output_idx, idx, trg):
            # dec_output [1, batch_size, output_dim]
            # hidden; [1, batch_size, dec_hid_dim]
            dec_output, hidden = self.decoder(dec_input, enc_output, dec_hid)
            dec_hid = hidden
            #output [seq_len, batch_size, output dim]
            output = torch.cat((output, dec_output), dim = 0)

            dec_output_idx = dec_output.squeeze(0).argmax(1).unsqueeze(1)
            output_idx = torch.cat((output_idx, dec_output_idx.unsqueeze(0)), dim = 0)
            teacher_force_bool = random.random() < teacher_force_ratio

            dec_input = trg[:, idx].unsqueeze(1) if teacher_force_bool else dec_output_idx
            idx += 1
        return output.contiguous().transpose(0, 1)

    def __finish(self, output_idx, idx, trg):
        if trg is not None:
            return idx < trg.size(1)
        else:
            tmp = output_idx.squeeze(2).transpose(0, 1)
            cp = torch.eq(tmp, self.eos)
            cp = torch.any(cp, 1)
            return not torch.all(cp).item()



if __name__ == '__main__':
    print('model')
    from util import build_dataset, get_time, cost_time, load_data, train_path

    t1 = get_time()
    print('loading data...')
    max_len = 4
    data = load_data(train_path, max_len=max_len)
    print('total data length:', len(data))
    t2 = get_time()
    h, m, s = cost_time(t1, t2)
    print("loading data cost %dh %dm %ds" % cost_time(t1, t2))

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print('device:', device)


    t1 = get_time()
    print('building dataset and vocab...')
    batch_size = 2
    min_freq = 1
    dataset, vocab = build_dataset(data, batch_size, device, min_freq)
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

    model = Model(encoder, decoder, dec_input_dim, enc_hid_dim, dec_hid_dim, device).to(device)

    en = list(dataset)[0]['en']
    print('en size:', en.size())
    print('en:', en)

    # testdata = next(iter(train_d))
    testdata = list(dataset)[0]
    input = testdata['en']
    target = testdata['zh']
    print('input size:', input.size())
    print('target size:', target.size())

    print('target: ', target)
    # output = model(input, target)
    output = model(input, teacher_force_ratio = 0)
    print(output.size())
    print(model)

    # a = torch.Tensor([[0,3,3,4,5,0,3,3], [0,2,2,5,6,7,9,8]]).long()
    # print('a', a.size())
    # print(a)
    # b = torch.eq(a, 1)
    # print('b', b)
    # c = torch.any(b, 1)
    # print('c', c)
    # print(not torch.all(c).item())
