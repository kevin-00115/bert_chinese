import warnings
# 忽视警告
warnings.filterwarnings('ignore')

def train(model, data, optim, lossfn, epoch = 0):
    model.train()
    loss_sum = 0
    for idx, btc in enumerate(data):
        optim.zero_grad()
        # [batch_size, seq_len]
        en = btc['en']
        zh = btc['zh']
        # output [batch_size, seq_len, output_dim]
        output = model(en, zh)
        # 去除掉序列第一个<cls>的影响
        predict = output[:, 1:].contiguous().view(-1, output.size(-1))
        zh_target = zh[:, 1:].contiguous().view(-1)
        loss = lossfn(predict, zh_target)
        loss.backward()
        optim.step()
        loss_sum += loss.item()
        if idx % 10 == 0:
            print("epcho: %d | idx: %d | btc_loss: %10f" % (epoch, idx, loss.item()))
    return loss_sum / data.get_length()



if __name__ == '__main__':
    pass
