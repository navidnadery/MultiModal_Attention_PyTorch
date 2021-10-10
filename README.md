# MultiModal_Attention_PyTorch
A simple implementation of a multi-modal (3 sources) single head attention module in pytorch


# N=2 neurons in input & T=10 sequence
inp_ch=2
inp_seq=10
# k=4 neurons & s=3 neurons'
key_dim=[4, 3]
# u=2 neurons''
que_dim=[4, 2]
# o=6
out_seq=6
# m=5
out_ch=5

batch_size=16

att_module = SingleHeadAttention(inp_ch, inp_seq, key_dim, que_dim, out_ch, out_seq, True)
x1 = torch.rand([batch_size, inp_ch, inp_seq], dtype=torch.float32)
x2 = torch.rand([batch_size, key_dim[0], key_dim[1]], dtype=torch.float32)
x3 = torch.rand([batch_size, que_dim[0], que_dim[1]], dtype=torch.float32)
out = att_module(x1, x2, x3)


# single output (non-sequence)
att_module = SingleHeadAttention(inp_ch, inp_seq, key_dim, que_dim, out_ch, 1, True)
out = att_module(x1, x2, x3)
