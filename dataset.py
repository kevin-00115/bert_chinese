import torch

class Dataset:
    def __init__(self, data, tokenize, vocab, batch_size = 2, device = 'cpu', start_token = '<cls>', end_token = '<eos>', sort_key = None):
        self.data = data
        self.batch_size = batch_size
        self.vocab = vocab
        self.start_token = start_token
        self.end_token = end_token
        self.device = device
        self.data_len = len(data)
        self.idx = 0
        self.tokenize = tokenize
        self.keys = list(tokenize.keys())
        self.res = [] # [{}, {}, {},... {}]
        self.len = 0

        temp_data = list(map(self.__cut_word, data))
        if sort_key is not None:
            temp_data.sort(key=sort_key)

        self.res = self.__generate_dataset(temp_data)

        # [batch_size, seq_len]

    def __generate_dataset(self, data):
        res = [] # [{}, {}, {},... {}]
        btsize = self.batch_size
        for i in range(int(self.data_len / btsize)):
            btc_data = data[i*btsize: i*btsize+btsize]
            btc = self.__generate_batch(btc_data, btsize) # {key: [], key2: []...}
            res.append(btc)

        left_len = self.data_len - int(self.data_len / btsize) * btsize

        if left_len != 0:
            btc_data = data[len(res)*btsize: ]
            btc = self.__generate_batch(btc_data, left_len)
            res.append(btc)        
        return res
    
    def __generate_batch(self, batch_data, batch_size):
        btsize = batch_size
        btc = {} # {key: [], key2: []...}
        for idx, key in enumerate(self.keys):
            btlist = [] #[[],[], ..., []]
            if self.tokenize[key] is not None:
                btc_max = max([len(batch_data[b][idx]) for b in range(btsize)])
                start_idx = self.vocab.stoi[key][self.start_token]
                end_idx = self.vocab.stoi[key][self.end_token]
                pad_idx = self.vocab.stoi[key]['<pad>']
                unk_idx = self.vocab.stoi[key]['<unk>']
                for b in range(btsize):
                    tknlist = [start_idx]
                    for tkn in batch_data[b][idx]:
                        tknlist.append(self.vocab.stoi[key][tkn] if tkn in self.vocab.stoi[key] else unk_idx)
                    tknlist.append(end_idx)
                    tknlist.extend([pad_idx]*(btc_max - len(batch_data[b][idx])))
                    btlist.append(tknlist)
            else:
                for b in range(btsize):
                    btlist.append(batch_data[b][idx])
            btc[key] = torch.LongTensor(btlist).to(self.device)
        return btc

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self.res):
            r = self.res[self.idx]
            self.idx += 1
            return r
        else:
            self.idx = 0
            raise StopIteration

    def get_length(self):
        return len(self.res)

    def __cut_word(self, x):
        res = []
        for idx, key in enumerate(self.keys):
            if self.tokenize[key] is not None:
                res.append(self.tokenize[key](x[idx]))
            else:
                res.append(x[idx])
        return tuple(res)



if __name__ == '__main__':
    print('dataset')

    from vocab import Vocab
    from util import load_data, get_time, cost_time, en_word_cut
    import jieba as jb
    import torch
    
    path = '/Users/kevinyang/bert_chinese/Bert_chinese_translation/translation2019zh_train.json'
    max_len = 15
    print('lodaing data...')
    t1 = get_time()
    data = load_data(path, max_len = max_len)
    print('data length:', len(data))
    t2 = get_time()
    print('loading data cost: %dh %dm %ds' % cost_time(t1, t2))

    tokenize = {'en': en_word_cut, 'zh': lambda x: jb.lcut(x)}
    # tokenize = {'en': en_word_cut, 'zh': None}

    vocab = Vocab(data, tokenize)
    vocab.build_vocab(min_freq = 1)
    t2 = get_time()

    # print('vocab itos:', vocab.itos)
    # print('vocab stoi:', vocab.stoi)
    # print('=========================')
    # print('vocab.freq:', vocab.freq)

    print('build vocab cost: %dh %dm %ds' % cost_time(t1, t2))
    # for d in data:
    #     print(d)
    #     print('====================================================================================')

    device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    t1 = get_time()
    dataset = Dataset(data, tokenize, vocab, device = device, sort_key=lambda x: len(x[0]))
    print('dataset len:', dataset.get_length())
    t2 = get_time()
    print('build dataset cost: %dh %dm %ds' % cost_time(t1, t2))

    print(list(dataset)[0]['en'].type())

    d1 = list(map(lambda x: (en_word_cut(x[0]), jb.lcut(x[1])), data))
    d1.sort(key=lambda x: len(x[0]))
    print('en:', [len(w[0]) for w in d1])
    print('zh:', [len(w[1]) for w in d1])

    for ds in dataset:
        en = ds['en']
        zh = ds['zh']
        ent = torch.Tensor(en)
        zht = torch.Tensor(zh)
        btcsize = ent.size(0)
        print('en size:', ent.size())
        for idx in range(btcsize):
            for tkn in ent[idx]:
                print(vocab.itos['en'][int(tkn.item())], end=' ')
            print()
        print('zh size:', zht.size())
        for idx in range(btcsize):
            for tkn in zht[idx]:
                print(vocab.itos['zh'][int(tkn.item())], end=' ')
            print()
        print('====================================================================================')
