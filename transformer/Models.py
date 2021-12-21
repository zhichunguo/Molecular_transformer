import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
import transformer.Constants as Constants


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1) #all pad mask is 1


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.
    
    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  #b * lq * lk

    return padding_mask


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)
    return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True)
        self.linear = nn.Linear(d_model, 512)

        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        for _ in range(n_layers)])
    
    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []
        
        #prepare mask ???
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        #forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        enc_output = self.linear(enc_output)
        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output,


class Decoder(nn.Module):
    def __init__(self, n_tgt_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        #farword
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
        
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, len_max_seq, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, tgt_emb_prj_weight_sharing=True, emb_src_tgt_weight_sharing=True):
        super().__init__()

        self.encoder = Encoder(n_src_vocab=n_src_vocab, len_max_seq=len_max_seq, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,dropout=dropout)
        self.decoder = Decoder(n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,dropout=dropout)

        self.mlp = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 2))
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
            'to judge the residual and outputs dimensions'
        
        if tgt_emb_prj_weight_sharing:
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            assert n_src_vocab == n_tgt_vocab, \
            'to judge the src and tgt emb'
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)

        return self.mlp(torch.sum(enc_output, dim=1))
        
        # remove if only using encoder
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seg_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        # enc_output is the last hidden with dimension (batch, length, 512)
        return seg_logit.view(-1, seg_logit.size(2)), enc_output

