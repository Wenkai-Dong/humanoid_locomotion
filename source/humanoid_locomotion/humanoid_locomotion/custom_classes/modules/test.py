from torch.nn.attention import sdpa_kernel, SDPBackend

import mha
import torch
import time
input_dim_1d = 64
input_channels_1d = 1
input_dim_2d = 64
device = torch.device("cuda")
mha = mha.MHA(
    input_dim_q=input_dim_1d,
    input_channels_q=input_channels_1d,
    input_dim_kv=input_dim_2d,
    num_heads=16,
    dropout=0.0,
    bias=True,
    # if create mult mha, attention_type must writer type of all multihead attention
    attention_type=["cross", "self"],
    norm=["layer","none"],
    # if create mult norm, norm_position must writer all type of norm.
    norm_position=["pre_norm","none"],
    activation=None,
    flatten=True
).to(device)
print(f"MHA Model: {mha}")

optimizer = torch.optim.AdamW(mha.parameters())

map = torch.rand(4096*4, 176, input_dim_2d).to(device)
pripro = torch.rand(4096*4, input_channels_1d, input_dim_1d).to(device)
print("start")
start = time.time()
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
# while True:
    for i in range(8):
        with sdpa_kernel(backends=[SDPBackend.MATH ]):
            x = mha(pripro, map)
        loss = x.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # break
time = time.time() - start
print(f"Time: {time}")