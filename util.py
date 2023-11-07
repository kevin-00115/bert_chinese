import json
import time
import jieba as jb

from dataset import Dataset
from vocab import Vocab


def load_data(path, max_len=None):
    english = []
    chinese = []
    idx = 0
    f = open(path, 'r')
    for line in f.readlines():
        data = json.loads(line)
        print(data)
        english.append(data['english'])
        chinese.append(data['chinese'])
        idx += 1
        if max_len is not None and idx == max_len:
            break
    f.close()
    return list(zip(english, chinese))


def build_dataset(data, batch_size, device, min_freq=1):
    tokenize = {'en': en_word_cut, 'zh': lambda x: jb.lcut(x)}
    vocab = Vocab(data, tokenize)
    vocab.build_vocab(min_freq)
    dataset = Dataset(data, tokenize, vocab, batch_size=batch_size, device=device, sort_key=lambda x: len(x[0]))
    return dataset, vocab


def en_word_cut(x):
    return [w for w in jb.lcut(x) if w != ' ']


def get_time():
    return time.time()


def cost_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time / 60 / 60)
    elapsed_mins = int((elapsed_time - elapsed_hours * 60 * 60) / 60)
    elapsed_secs = int(elapsed_time - (elapsed_hours * 60 * 60) - (elapsed_mins * 60))
    return elapsed_hours, elapsed_mins, elapsed_secs


def get_parameters_num(model):
    res = sum(p.numel() for p in model.parameters())
    return res


train_path = '/Users/kevinyang/bert_chinese/Bert_chinese_translation/translation2019zh_train.json'
vaild_path = '/Users/kevinyang/bert_chinese/Bert_chinese_translation/translation2019zh_valid.json'

if __name__ == '__main__':
    print('util')
    path = '/Users/kevinyang/bert_chinese/Bert_chinese_translation/translation2019zh_train.json'
    max_len = 15000
    data = load_data(path, max_len)
    print(data[0])
    zh = list(map(lambda x: x[1], data))
    print(zh)
