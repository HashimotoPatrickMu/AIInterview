## Convolutions

> **Depthwise Conv**
>
> > For an input tensor with $C$ channels, depthwise convolution applies a separate single-channel filter to each input channel. Thus, if the input tensor has $C$ channels, there will be $C$​ filters, and each filter is applied to its respective input channel.
> >
> > ```python
> > import torch
> > import torch.nn as nn
> > 
> > in_channels = 3
> > kernel_size = 3
> > stride = 1
> > padding = 1
> > 
> > depth_conv = nn.Conv2d(in_channels=in_channels,
> >                            out_channels=in_channels,    # Equal to input channels
> >                            kernel_size=kernel_size,
> >                            stride=stride,
> >                            padding=padding,
> >                            groups=in_channels)
> > ```
>
> **Pointwise Conv & $1\times1$ Conv & Separable Conv**
>
> > $1\times1$ represents kernel size. 
> >
> > ```python
> > pointwise_conv = nn.Conv2d(in_channels=in_channels,
> >                            out_channels=out_channels_after_pointwise,
> >                            kernel_size=1,  # 1x1 convolution
> >                            stride=1,
> >                            padding=0,
> >                            groups=1)  # Normal convolution, no grouping
> > ```
> >
> > Pointwise Conv is often applied after the Depthwise Conv. 
> >
> > For visualization, please refer [HERE](https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec). 
>
> **Deconvolution (ConvTranspose2d) & PixelShuffle**
>
> > Shortcomings of Deconvolution
> >
> > > Checkerboard Artifacts: Due to the overlap and uneven coverage of the convolutional filters across the input space during the upsampling process. The carelessness choice on stride and kernel size would lead to some pixels being updated more frequently than others, resulting in visible patterns or artifacts in the upsampled images. 
> > >
> > > Uneven Overlap: Due to the stride does not evenly divide the kernel size, leading to uneven distribution of the output values, exacerbating the checkerboard effect. 
> >
> > PixelShuffle: Upscales by rearranging the tensor. 
>
> **Kernel Size**
>
> > Empirically $(3,3)$ and $(5,5)$ are common choices for image related tasks. While in deep neural networks inputs high-resolution images, a series of descending kernel size perform better, for instance $(7,7)$ to $(5,5)$ to $(3,3)$. 
>
> **Factorized Convolution**
>
> > A $(n,n)$ convolution kernel can be decomposed into $(1,n),(n,1)$​, determining horizontal and vertical features, respectively. 
>
> **Dropout in Conv**
>
> > Dropout the output feature map. 
> >
> > Dropout some channels of input feature map. 
> >
> > Randomly zero-mask the kernel. 
> >
> > Mask a contiguous region on the input feature map. 
>
> **Upsampling (non-leanable) Followed by Convolution**
>
> > Smooths out the upscaled image.



---

## Pooling

> **MaxPool**
>
> > Only the neurons that contributed the maximum value during the forward pass receive the gradient, while all other neurons in the pooling region receive a gradient of zero.
> >
> > Remember the maximum location => Apply the gradient to the maximum => Zero-out the other gradients. 
>
> **Functionality**
>
> > Reduce the computational complexity.
> >
> > Increase receptive field. 
> >
> > Prevent overfitting. 
>

---

## Architecture

> [**ShuffleNet**](https://medium.com/syncedreview/shufflenet-an-extremely-efficient-convolutional-neural-network-for-mobile-devices-72c6f5b01651)
>
> **Attention**
>
> **ResNet**
>
> > Residual connection / Skip connection. 
> >
> > Bottleneck structure. 
> >
> > Solves the gradient vanishing problem and allows the neural network to go deeper. 
>
> [**Graph Neural Networks**](https://distill.pub/2021/gnn-intro/)
>
> **UNet** 
>
> > Contracting Path (Downsampling):  It focuses on capturing high-level, contextual information about the image content.
> >
> > Expanding Path (Upsampling): It uses the high-level context from the contracting path and combines it with precise location information to accurately segment the objects.
>
> **TO BE CONTINUED~** :hugs:
