
import torch
import torch.nn as nn
import torch.nn.functional as F

""""
attention公式：
sofrmax(W * embedding·enc)
*: 矩阵乘
·: 向量内积

模型中的应用
softmax(W * dec_embedding·enc_output)
dec_embedding: (1, 8, 256); (seq_len, batch_size, vector)
W: (256, 1024)matrix
enc_output: (seq_len, 8, 1024); (seq_len, batch_size, vector)


"""

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_emb_dim):
        super(Attention, self).__init__()
        # W 矩阵
        self.fc1 = nn.Linear(dec_emb_dim, enc_hid_dim*2)
        self.softmax = nn.Softmax(dim = 1)
    # dec_embedding [batch_size, 1 , emb_dim]
    # enc_output [batch_size, seq_len, enc_hid_dim*2]
    def forward(self, dec_embedding, enc_output):
        # W * dec_embedding; [batch_size, 1, enc_hid_dim*2]
        x = self.fc1(dec_embedding)
        x = torch.tanh(x)
        # W * dec_embedding·enc_output [batch_size, seq_len, 1]
        x = torch.einsum('ijk,ilk->ijl', enc_output, x)
        # 得到序列中每个单位的比重
        # softmax by seq dim; [batch_len, seq_len, 1]
        x = self.softmax(x)
        # 按序列比例乘以序列信息; [batch_size, seq_len , enc_hid_dim*2]
        x = torch.mul(x, enc_output)
        # 获得每个batch的序列信息，将整个序列的序列信息求和; 
        # [batch_size, 1, enc_hid_dim*2]
        x = torch.sum(x, dim = 1).unsqueeze(1)
        return x

if __name__ == '__main__':
    print('attention')
    x = torch.tensor([[0.1,0.2,0.7], [0.3,0.4,0.3]]).view(2,3,1)
    print('x:', x.size())
    print(x)

    y = torch.tensor(range(24)).view(2,3,4)
    print('y:', y.size())
    print(y)
    z = torch.mul(x, y)
    print('z:', z)
    print('res:', torch.sum(z, dim = 1))
    '''x = torch.tensor([range(36)]).view(3,3,4)
    print(x.size())
    print(x)

    y = torch.tensor([[[0.1],[0.2],[0.3]],[[0.4],[0.5],[0.6]],[[0.7],[0.8],[0.9]]])
    print(y.size())
    print(y)

    z = torch.mul(x, y)
    print(z.size())
    print(z)

    w = torch.sum(z, dim = 0).unsqueeze(0)
    print(w.size())
    print(w)'''
    '''
    x = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12]).view(1,3,4)

    y = x.expand(4,3,4)
    print(y.size())
    print(y)

    z = torch.tensor([range(12, 24)]).view(1,3,4)
    print(z.size())
    print(z)

    w = torch.einsum('ijk,ljk->ijl', y, z)
    print(w.size())
    print(w)

    print('=================================')
    x1 = torch.tensor([range(48)]).view(4,3,4)
    print(x1.size())
    print(x1)

    y1 = torch.tensor([range(48, 48+12)]).view(1,3,4)
    print(y1.size())
    print(y1)

    w1= torch.einsum('ijk,ljk->ijl', x1, y1)
    print(w1.size())
    print(w1)

    print('=================================')

    x2 = torch.tensor([range(48)], dtype = torch.float64).view(4, 3,4)
    print(x2.size())
    print(x2)

    softmax = nn.Softmax(dim = 0)
    y2 = softmax(x2)
    print(y2.size())
    print(y2)

    w3 = torch.sum(y2, 0)
    print(w3.size())
    print(w3)
    '''
