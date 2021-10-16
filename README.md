# Multi-Modal Single-head attention based on key, query, value
A simple implementation of a multi-modal (3 different sources), single head attention module in pytorch using *key*, *query*, *value*.

## Different Input/Output size
The size of input/output can be different for the number of neurons or the sequence length. In the following, the size of input/output is shown:

1. `inp_ch=2` <p style="color:yellow;">the number of channels (neurons) in the main input</p>
2. `inp_seq=10` <p style="color:yellow;">the sequence length in the main input </p>
3. `out_ch=5` <p style="color:yellow;">the number of channels (neurons) in the output </p>
4. `out_seq=6` <p style="color:yellow;">the sequence length in the output </p>

## Input modal 2 and 3 as key and query (correspondingly)
The input for key, can have any dimention for both channels and sequence length (features).
The input for query, can have any dimention for the second dim (sequence length or features) but with the number of channels like key dim1.

5. `key_dim=[4, 3]` <p style="color:yellow;">[neuronsKQ & seq_lengthK] of Input features 2 </p>
6. `que_dim=[4, 2]` <p style="color:yellow;">[neuronsKQ & seq_lengthQ] of Input features 3 </p>

# How to use
Define the batch size

`batch_size=16`

<br/>
Initialize an object from the class *SingleHeadAttention* as follow:

```
att_module = SingleHeadAttention(inp_ch, inp_seq, key_dim, que_dim, out_ch, out_seq, True)
```
Initialize some random matrix as inputs or prepare your input such that:
* x1: main input (value)
* x2: key input
* x3: query input

**NOTE**

Input features should be as batch first, the output is the same.
Input shape is as follow:
`batch, sequence_length,input_channel(neurons)`
<p style="color:blue">Example input features</p>

```
x1 = torch.rand([batch_size, inp_seq, inp_ch], dtype=torch.float32)
x2 = torch.rand([batch_size, key_dim[0], key_dim[1]], dtype=torch.float32)
x3 = torch.rand([batch_size, que_dim[0], que_dim[1]], dtype=torch.float32)
out = att_module(x1, x2, x3)
```

Example of single output (non-sequence)

```
att_module = SingleHeadAttention(inp_ch, inp_seq, key_dim, que_dim, out_ch, 1, True)
out = att_module(x1, x2, x3)
```
