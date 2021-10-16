import torch
import numpy as np
from torch import nn
from torch.nn.functional import softmax

class SingleHeadAttention(nn.Module):
  def __init__(self, inp_ch: int, inp_seq: int, key_dim: list, que_dim: list, out_ch: int, out_seq: int, verbos=False) -> None:
    super(SingleHeadAttention, self).__init__()
    self.N = inp_ch
    self.T = inp_seq
    assert len(key_dim)==2 and len(que_dim)==2 and key_dim[0]==que_dim[0], f"1'st dimention of key and querry should be the same but key is {key_dim[0]} and querry is que_dim[0]"
    self.k, self.s = key_dim
    self.k, self.u = que_dim
    self.m = out_ch
    self.o = out_seq
    self.w_key = nn.Parameter(torch.randn([self.s, self.T]))
    self.w_query = nn.Parameter(torch.randn([self.u, self.o]))
    self.w_value = nn.Parameter(torch.randn([self.m, self.N]))
    self.verbos = verbos

  def forward(self, input, key, query):
    x1, x2, x3 = input, key, query
    keys = x2 @ self.w_key
    querys = x3 @ self.w_query
    values = (x1 @ self.w_value.T).transpose(1,2)
    attn_scores = querys[:,:,None] * keys[:,:,:,None]
    attn_scores = attn_scores.sum(dim=1)  / np.sqrt(attn_scores.shape[1])
    attn_scores_softmax = softmax(attn_scores, dim=1)
    outputs = values @ attn_scores_softmax
    if self.verbos:
      print(f"key is:    {keys.shape}")
      print(f"query is:  {querys.shape}")
      print(f"value is:  {values.shape}")
      print(f"output is: {outputs.shape}")
    return outputs
