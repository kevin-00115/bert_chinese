
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, attention, input_dim, emb_dim, hid_dim, enc_hid_dim, dropout):
        super(Decoder, self).__init__()
        self.attention = attention
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim+enc_hid_dim*2, hid_dim, bias = True, batch_first = True)
        self.fc_out = nn.Linear(hid_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
    # dec_input [batch_size, 1]
    # enc_output [batch_size, seq_len, enc_hid_dim*2]
    # dec_hid [1, batch_size, dec_hid_dim]
    def forward(self, dec_input, enc_output, dec_hid):
        # embedding [batch_size, 1 , emb_dim]
        embedding = self.embedding(dec_input)
        embedding = self.dropout(embedding)
        
        # attn [batch_size, 1, enc_hid_dim*2]
        attn = self.attention(embedding, enc_output)
        # gru_input [batch_size, 1, enc_hid_dim*2+emb_dim]
        gru_input = torch.cat((embedding, attn), 2)

        # gru_output [batch_size, 1, hid_dim]
        # hidden [1, batch_size, hid_dim]
        gru_output, hidden = self.gru(gru_input, dec_hid)

        # [1, batch_size, dec_input_dim]
        output = self.dropout(self.fc_out(gru_output)).transpose(0, 1)
        return output, hidden

if __name__ == '__main__':
    print('decoder')
    import util
    import time
    from torchtext.legacy.data import BucketIterator

    stt = time.time()
    data = util.load_data(util.train_path, max_len=800)
    print('data len', len(data))
    print('loading data cost', time.time() - stt, 's')

    stt = time.time()
    dataset, en_vocab, zh_vocab = util.build_dataset(data)
    print('build dataset cost', time.time() - stt, 's')

    td, vd = dataset.split(split_ratio = 0.7)
    train_d, vaild_d = BucketIterator.splits(
        (td, vd), 
        batch_sizes = (8, 8)
    )

    print('zh vocab len', len(zh_vocab.stoi))
    dec_input_dim = len(zh_vocab.stoi)
    dec_emb_dim = 256
    dec_hid_dim = 512
    dec_dropout = 0.5

    decoder = Decoder(dec_input_dim, dec_emb_dim, dec_hid_dim, dec_dropout)


    testdata = next(iter(train_d))
    input = testdata.en
    target = testdata.zh

    output = decoder(input)