
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, bias = True, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(dropout)
    # x [batch_size, seq_len]
    def forward(self, x):
        # [batch_size, seq_len, emb_dim]
        x = self.embedding(x)
        x = self.dropout(x)
        # output [batch_size, seq_len, 2*hid_dim]
        # hidden [2, batch_size, hid_dim]
        output, hidden = self.gru(x)
        return output, hidden


if __name__ == '__main__':
    print('encoder')
    from util import build_dataset, get_time, cost_time, load_data, train_path

    t1 = get_time()
    print('loading data...')
    max_len = 2
    data = load_data(train_path, max_len=max_len)
    print('total data length:', len(data))
    t2 = get_time()
    h, m, s = cost_time(t1, t2)
    print("loading data cost %dh %dm %ds" % cost_time(t1, t2))

    device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
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
    en = list(dataset)[0]['en']
    print('en size:', en.size())
    print('en:', en)

    output, hidden = encoder(en)
    print(output.size())
    print(hidden.size())