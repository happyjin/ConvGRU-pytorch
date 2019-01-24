# ConvLSTM_pytorch
**[This](https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py)** file **contains 
the implementation of Convolutional LSTM in PyTorch**

### How to Use
The `ConvGRU` module derives from `nn.Module` so it can be used as any other PyTorch module.

The ConvGRU class supports an arbitrary number of stacked hidden layers in GRU. In this case, it can be specified 
the hidden dimension (that is, the number of channels) and the kernel size of each layer. In the case more layers 
are present but a single value is provided, this is replicated for all the layers. For example, in the following 
snippet each of the three layers has a different hidden dimension but the same kernel size.

Example usage:
```
# set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# detect if CUDA is available or not
use_gpu = torch.cuda.is_available()
if use_gpu:
    dtype = torch.cuda.FloatTensor # computation in GPU
else:
    dtype = torch.FloatTensor

height = width = 6
channels = 256
hidden_dim = [32, 64]
kernel_size = [(3,3), (3,3)] # for two stacked hidden layer
num_layers = 2 # number of stacked hidden layer
model = ConvGRU(input_size=(height, width),
                input_dim=channels,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                dtype=dtype,
                batch_first=True,
                bias = True,
                return_all_layers = False)

batch_size = 1
time_steps = 1
input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)
layer_output_list, last_state_list = model(input_tensor)
```



### Disclaimer

This is still a work in progress and is far from being perfect: if you find any bug please don't hesitate to open an issue.

### License
Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Acknowledgment
This repo borrows some codes from 
- [ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)

