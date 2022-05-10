from turtle import forward
import numpy as np
import math
from pygame import K_s
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

        ## F1: 这里就是位置编码，把两者相加放入到这个函数里面，从这里可以去看一下位置编码函数的实现：
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        
        ## F3: get_attn_pad_mask 是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响
        ## 因为batch的长度都不一致，所以设置一个最大长度，去掉大于最大长度的信息，小于的pad填充
        ## 告诉模型后面的符号是被填充的：计算应该去掉字符与pad符号的相关性
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            ## F2: 去看EncoderLayer层函数
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder():
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs) ## [batch, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0. 1) ## [batch_size, tgt_len, d_model]

        ## get_attn_pad_mask 自注意力层的时候的pad部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        ## get_attn_subsequent_mask 这个做的事自主一层的mask部分，就是当前单词之后看不到，只用一个上三角的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        ## 两个矩阵相加，大于0的为1，不大于0的为0， 为1的之后就会被fill到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        ## 这个做的是交互注意力机制中的mask矩阵，enc的输入是k，我曲看这个k里面哪些是pad符号，给到后面的模型；注意哦，我q肯定也是有pad符号，但是这里我不在意
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


## F1. PositionalEncoding 代码实现
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

## F2. EncoderLayer: 包含两个部分，多头注意力和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention() ## F4
        self.pos_ffn = PoswiseFeedForwardNet() ## 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        ## F4下面这个就是做自注意力层，输入时enc_inputs,形状时[batch_size x seq_len_q x d_model] 需要注意的时最初始的QKV矩阵时等同于这个输入的，去看下enc_self_attn函数
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) ## 最原始的enc_inputs to same Q, K, V
        enc_outputs = self.pos_ffn(enc_outputs) ## enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


## F3. get_attn_pad_mask
## 比如说， 我现在句子的长度是5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状
## len_input * len * input 代表每个单词对其余包含自己的单词的影响力

## 所以这里我需要一个同等大小形状的矩阵，告诉我那个位置是PAD部分，之后再计算softmax之前会把这里置为无穷大

## 一定需要注意的是这里得到的矩阵形状是batch_size_x, len_q, x, len_k, 我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没有必要

## seq_q和seq_k不一定不一致，在自注意力里面一致，但是在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这本pad符号信息就可以，解码端的pad信息在交互注意力层是没有用的
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    ## eq(zero) is PAD token； seq_k里面哪些位置是pad符号（为零），把这个位置上的数字改成1
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) ## batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)

## F4: MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射Linear做一个映射得到参数矩阵Wq, Wk, Wv
        ## 最后需要保证QKV矩阵是相同的
        ## 使用词嵌入和参数矩阵相乘，编码端把词复制了3份进行，解码端
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ## 输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        ## (B, S, D) -proj -> (B, S, D) -split -> (B, S, H, W) -trans-> (B, H, S, W)

        ## 下面这个就是先映射，后分头：一定要注意的是q和k分头之后维度是一致的，所以这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2) ## q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2) ## q_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2) ## q_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k, 然后经过下面这个代码得到新的attn_mask: [batch_size x n_heads x len_q x len_k] 就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        ## 然后我们计算ScaledDotProductAttention 这个函数（F5）
        ## 得到的结果有两个 context:[batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) ## context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn ## output: [batch_size x len_q x d_model]

## F5 ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k] KL: [batch_size x n_heads x len_k x d_k] V: [batch_size x n_heads x len_k x dv]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        ## 然后关键词地方来了，下面就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) ## Fills elements of self tensor with value where mask is one
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn