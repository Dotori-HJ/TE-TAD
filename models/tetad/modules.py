import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.

    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397

    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
    .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.

    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py

    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.

    Check pytorch's BatchNorm1d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

    def forward(self, inp, inp_mask):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        # mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        n = inp_mask.sum()
        mask = inp_mask / n
        mask = mask.unsqueeze(1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 2]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp, inp_mask


class MaskedConv1D(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#10
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0
        mask = ~mask.unsqueeze(1)
        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, ~out_mask.squeeze(1)

class ConvBlock(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#L735
    A simple conv block similar to the basic block used in ResNet
    """
    def __init__(
        self,
        n_channels,            # dimension of the input features
        kernel_size=3,         # conv kernel size
        stride=1,         # downsampling stride for the current layer
        expansion_factor=2,    # expansion factor of feat dims
        n_out=None,            # output dimension, if None, set to input dim
        act_layer=nn.ReLU,     # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_channels

         # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = n_channels * expansion_factor
        self.conv1 = MaskedConv1D(
            n_channels, width, kernel_size, stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_channels, n_out, 1, stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


class GroupNorm(nn.GroupNorm):
    def forward(self, x, mask):
        out = super().forward(x)
        return out, mask

class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x, mask):
        out = super().forward(x)
        return out, mask


class LayerNorm(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/libs/modeling/blocks.py#L63
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x, mask):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out, mask

class Sequential(nn.Sequential):
    def forward(self, x, mask):
        for module in self:
            x, mask = module(x, mask)
        return x, mask

class Identity(nn.Identity):
    def forward(self, x, mask):
        out = super().forward(x)
        return out, mask

class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
        from https://github.com/happyharrycn/actionformer_release/libs/modeling/backbones.py#L168
    """
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        kernel_size,
        arch=(2, 2),
        num_feature_levels=5,
        scale_factor=2,
        with_ln=False,
    ):
        super().__init__()
        # assert num_feature_levels > 1
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.arch = arch
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            feature_dim = hidden_dim if idx > 0 else feature_dim
            self.embd.append(
                MaskedConv1D(
                    feature_dim, hidden_dim, kernel_size,
                    stride=1, padding=kernel_size//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(hidden_dim))
                # self.embd_norm.append(BatchNorm1d(hidden_dim))
            else:
                self.embd_norm.append(Identity())

        for layer in self.embd:
            nn.init.xavier_uniform_(layer.conv.weight, gain=1)
            if not with_ln:
                nn.init.zeros_(layer.conv.bias)

        # stem network using convs
        self.stem = nn.ModuleList([
            ConvBlock(hidden_dim, kernel_size=3, stride=1)
            for _ in range(arch[1])
        ])

        # main branch using convs with pooling
        self.branch = nn.ModuleList([
            Sequential(
                MaskedConv1D(hidden_dim, hidden_dim, kernel_size=3, stride=self.scale_factor, padding=1),
                LayerNorm(hidden_dim),
            )
            # ConvBlock(hidden_dim, kernel_size=3, stride=self.scale_factor)
            for _ in range(num_feature_levels-1)
        ])
        for layer in self.branch:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            if not with_ln:
                nn.init.zeros_(layer[0].conv.bias)
        # init weights
        self.apply(self.__init_weights__)



    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.embd_norm[idx](x, mask)[0]
            # x = self.activation(self.embd_norm[idx](x, mask)[0])

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, output, prev_output):
        if self.drop_prob > 0 and self.training:
            p = torch.rand(output.size(0), dtype=torch.float32, device=output.device)
            p = (p > self.drop_prob)[:, None, None]
            output = torch.where(p, output, prev_output)
        else:
            output = output

        return output

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'