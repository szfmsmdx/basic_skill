import torch
from torch import nn 
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_len, d_model):
        # vocab_len 是有多少个词，d_model 是词嵌入的维度
        super(TokenEmbedding, self).__init__(vocab_len, d_model, padding_idx=1)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        # max_len 是词的最大长度
        super(PositionEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # 不进行梯度更新

        pos = torch.arange(0, max_len).unsqueeze_(1).float()
        _2i = torch.arange(0, d_model, 2).float()   # 上标上的2i

        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]
    
# 整合一下
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(p = drop_prob)
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        # print("token shape:", tok_emb.shape)
        # print("pos shape:", pos_emb.shape)
        return self.dropout(tok_emb + pos_emb)
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.d_model)) # nn.Parameter 让参数可学习
        self.beta = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * norm + self.beta
        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # dropout 在前是为了减少神经元的连接，不然做完fc2连接完了再dropout就没有意义了
        x = self.fc2(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiheadAttention, self).__init__()
        
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.n_head = n_head
        self.d_model = d_model
        self.n_d = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, token_size, d_model = q.shape
        
        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.n_d)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.n_d)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.n_d)
        # shape is (batch_size, token_size, d_model)
        # print("q shape", q.shape)
        
        # Transpose for attention dot product
        q = q.transpose(1, 2)  # (batch_size, n_head, seq_len, n_d)
        k = k.transpose(1, 2)  # (batch_size, n_head, seq_len, n_d)
        v = v.transpose(1, 2)  # (batch_size, n_head, seq_len, n_d)
        # (batch_size, n_head, token_size, n_d)
        # print("q shape", q.shape)

        score = q @ k.permute(0, 1, 3, 2) / math.sqrt(self.n_d)
        # (batch_size, n_head, token_size, token_size)
        # print("score shape", score.shape)
        if mask is not None:
            # mask = torch.tril(torch.ones(token_size, token_size, dtype=bool))
            score = score.masked_fill(mask == 0, float('-inf'))
        # print("after mask score shape", score.shape, "v shape", v.shape)
        score = self.softmax(score) @ v
        # (batch_size, n_head, token_size, n_d)
        # print("after softmax score shape", score.shape)
        score = score.permute(0, 2, 1, 3).contiguous().view(batch_size, token_size, d_model)
        out = self.w_combine(score)
        return out
    
class EncodingLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(EncodingLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x  # resnet 提前保存一下
        x = self.attention(x, x, x, mask)
        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob=0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiheadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.drop3 = nn.Dropout(drop_prob)
    def forward(self, dec, enc, token_mask, dec_mask):
        _dec = dec
        dec = self.attention(dec, dec, dec, token_mask)
        dec = self.drop1(dec)
        dec = self.norm1(dec + _dec)

        if enc is not None:
            _dec = dec
            dec = self.cross_attention(dec, enc, enc, dec_mask)
            dec = self.drop2(dec)
            dec = self.norm2(dec + _dec)

        _dec = dec
        dec = self.ffn(dec)
        dec = self.drop3(dec)
        dec = self.norm3(dec + _dec)
        return dec
    
class Encoder(nn.Module):
    def __init__(
        self, 
        vocab_size,
        d_model, 
        max_len,
        ffn_hidden, 
        n_head, 
        drop_prob,
        n_layers
    ):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob)
        self.layers = nn.ModuleList(
            [
                EncodingLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layers)
            ]
        )
    
    def forward(self, x, enc_mask):
        x = self.embedding(x)
        # print("encoder embedding shape", x.shape)
        for layer in self.layers:
            x = layer(x, enc_mask)
        return x
    
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        max_len,
        ffn_hidden,
        n_head,
        drop_prob,
        n_layers
    ):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_len, drop_prob
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_head, ffn_hidden, drop_prob)
                for _ in range(n_layers)
            ]
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, dec, enc, token_mask, dec_mask):
        dec = self.embedding(dec)
        # print("decoder embedding shape", dec.shape)
        for layer in self.layers:
            dec = layer(dec, enc, token_mask, dec_mask)

        dec = self.fc(dec)
        
        return dec

class Transformer(nn.Module):
    def __init__(
        self, 
        src_pad_idx, 
        tgt_pad_idx,
        enc_voc_size, 
        dec_voc_size,
        max_len,
        d_model,
        n_head,
        ffn_hidden,
        n_layers,
        drop_prob
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            enc_voc_size, d_model, max_len, ffn_hidden, n_head, drop_prob, n_layers
        )
        self.decoder = Decoder(
            dec_voc_size, d_model, max_len, ffn_hidden, n_head, drop_prob, n_layers
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        '''
        attention padding mask, 用于屏蔽padding的token, 使其不参与attention计算
        pad_idx_q: query的padding token
        pad_idx_k: key的padding token
        '''
        len_q = q.shape[1]  # q shape is (batch_size, q_token_size)
        len_k = k.shape[1]  # k shape is (batch_size, k_token_size)
        # attention 结束的shape 是 batch_size, q_token_size, k_token_size
        
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask
    
    def causal_mask(self, q, k):
        len_q, len_k = q.shape[1], k.shape[1]
        mask = (torch.tril(torch.ones(len_q, len_k, dtype=bool)))
        return mask
    
    def forward(self, src, tgt):
        src_mask = self.pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        tgt_mask_1 = self.pad_mask(tgt, tgt, self.tgt_pad_idx, self.tgt_pad_idx)
        tgt_mask_2 = self.causal_mask(tgt, tgt)
        tgt_mask = tgt_mask_1 * tgt_mask_2
        src_tgt_mask = self.pad_mask(tgt, src, self.tgt_pad_idx, self.src_pad_idx)
        # print("src_mask shape:", src_mask.shape)
        # print("tgt mask_1 shape:", tgt_mask_1.shape)
        # print("tgt mask_2 shape:", tgt_mask_2.shape)
        # print("tgt_mask shape:", tgt_mask.shape)
        # print("src_tgt_mask shape:", src_tgt_mask.shape)
        enc = self.encoder(src, src_mask)
        ouput = self.decoder(tgt, enc, tgt_mask, src_tgt_mask)
        return ouput