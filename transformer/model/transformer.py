from turtle import forward
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

d_model = 512 # Embedding Size
d_ff = 2048 # FeedForward Dimension
d_k = d_v = 64 # dimension of K(=Q), V
n_layers = 6 # number of Encoder or Decoder Layer
n_heads = 8 # number of heads in Multi-Head Attention
class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        ## 输出层 d_model 是我们解码层每个toekn输出的维度大小
        ## 之后会做一个tgt_vocab_size大小的softmax; 这个softmax产生的是选词表中某个词的概率
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias = False)
    
    def forward(self, enc_inputs, dec_inputs):
        ## enc_inputs: 形状为[batch_size, src_len]主要作为编码段（encoder）的输入
        ## dec_outputs: 作为主要输出， enc_self_attns这里没有记错就是Q与K^T相乘后softermax之后的矩阵值，代表的每个单词的相关性
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        ## dec_outputs是decoder主要输出，用于后续的Linear映射：
        ## dec_self_attns类比于enc_self_attns是查看每个单词对decoder中输入的其余单词的相关性： 
        ## dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性
        ## TODO: enc_inputs作用: 
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        ## dec_outputs做映射到词表大小
        dec_logits = self.projection(dec_outputs) ## dec_logits: [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model) ## 输入编码层： 定义生成一个矩阵， 大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(d_model) ## 位置编码层， 这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来

    def forward(self, enc_inputs):
        ## 这里我们的enc_inputs的形状是：[batch_size x source_len]
        
        ## 下面这个代码通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        ## 字符转换数字，数字转换为Embbedding
        enc_outputs = self.src_emb(enc_inputs)

        ## C1: 这里就是位置编码，把两者相加放入到这个函数里面，从这里可以去看一下位置编码函数的实现：
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        
        ## get_attn

class Decoder():
    pass


## C1. PositionalEncoding 代码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000) -> None:
        super(PositionalEncoding).__init__()

        ## 位置编码的实现其实很简单，照着公式敲代码就行了，下面的代码只是一种实现方式
        ## 从理解来讲，需要注意的就是偶数和奇数在公式上有个共同的部分，我们使用log函数把次方拿下来，方便计算
        ## 假设我的 d_model = 512, 那么公式里的pos代表的从0，1，2，3，...., 511的每一个位置，2i那个符号中i从0取到了255，那么2i对应的值就是0，2，4，...,510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)).float() * (-math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term) ## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，步长为2，其实代表的是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term) ## 这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，步长为2，其实代表的是奇数位置
        ## 上面代码获取之后得到的是pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是: [max_len*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        ## x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout

## C2. EncoderLayer: 包含两个部分，多头注意力和前馈神经网络
class EncoderLayer(nn.modules):
    pass