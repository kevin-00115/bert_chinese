'''
data: 数据[(),(),()...]
tokenize: {key1: tokenize1, key2: tokenize2, ...}
'''
class Vocab:
    def __init__(self, data, tokenize, path = None):
        self.data = data
        self.path = path
        self.tokenize = tokenize
        self.tokenize_value = list(tokenize.values())
        self.stoi = {}
        self.itos = {}
        self.freq = {}
        self.init_tokens = ['<cls>', '<eos>', '<slp>', '<pad>', '<unk>']
        self.keys = list(tokenize.keys())
        for key in self.keys:
            self.stoi[key] = {}
            self.itos[key] = []
            self.freq[key] = {}

    def build_vocab(self, min_freq):
        for combine in self.data:
            for idx, sent in enumerate(combine):
                if self.tokenize_value[idx] != None:
                    tokens = self.tokenize_value[idx](sent)
                    for tkn in tokens:
                        if tkn in self.freq[self.keys[idx]]:
                            self.freq[self.keys[idx]][tkn] += 1
                        else:
                            self.freq[self.keys[idx]][tkn] = 1

        for key in self.keys:
            if self.tokenize[key] != None:
                for idx, token in enumerate(self.init_tokens):
                    self.itos[key].append(token)

        for key in self.keys:
            if self.tokenize[key] != None:
                self.itos[key].extend(list(filter(lambda x: self.freq[key][x] >= min_freq, self.freq[key].keys())))


        for key in self.keys:
            if self.tokenize[key] != None:
                for idx, token in enumerate(self.itos[key]):
                    self.stoi[key][token] = idx

if __name__ == '__main__':
    print('vocab')

    import jieba as jb
    from util import load_data, get_time, cost_time, en_word_cut
    import os
    path = 'translation2019zh_train.json'
    max_len = 15
    print('lodaing data...')
    print(os.getcwd())
    t1 = get_time()
    data = load_data(path, max_len = max_len)
    zh = list(map(lambda x: x[1], data))
    print('data length:', len(zh))
    t2 = get_time()
    print('loading data cost: %dh %dm %ds' % cost_time(t1, t2))

    t1 = get_time()

    vocab = Vocab(data, {'en': en_word_cut, 'zh': lambda x: jb.lcut(x)})
    vocab.build_vocab(1)
    t2 = get_time()
    print('vocab itos:', vocab.itos)
    print('vocab stoi:', vocab.stoi)
    print('=========================')
    print('vocab.freq:', vocab.freq)

    print('build vocab cost: %dh %dm %ds' % cost_time(t1, t2))
