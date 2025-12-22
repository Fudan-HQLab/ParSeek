import numpy as np
import torch

def powerSpectrum(m: torch.tensor):
    a = torch.fft.fft2(m)
    b = torch.fft.fftshift(a)
    c = torch.log(1 + torch.abs(b))
    # plt.imshow(torch.log(1+ torch.abs(b)),cmap='gray')
    # plt.show()
    c = (c - c.min()) / (c.max() - c.min())
    return c


def zscore(x: np.array):
    if len(x.shape)==2 :
        m = x.mean()
        s = x.std()
        return (x - m) / s , m, s
    elif len(x.shape)==3:
        c, h ,w= x.shape
        if c ==1:
            m = x.mean()
            s = x.std()
            return (x-m)/s ,m ,s
        else:
            raise NotImplementedError("can't do zscore across channel")
    elif len(x.shape)==4:
        n, c,h,w = x.shape
        if n==1 and c==1:
            m = x.mean()
            s = x.std()
            return (x-m)/s, m,s
        else:
            raise NotImplementedError("can't do zscore across channel or batch")
    else:
            raise NotImplementedError("dims of tensor.shape is larger than 5!")


def reverse_zscore(x, x_mean, x_sigma):
    return (x * x_sigma) + x_mean


def normalize(x: np.array):
    n,c,h,w = x.shape
    x_flat = x.view(n,-1)
    min_vals = x_flat.min(dim=1,keepdim=True)[0]
    max_vals = x_flat.max(dim=1,keepdim=True)[0]

    range_vals = max_vals - min_vals
    range_vals = torch.where(
        range_vals == 0.0,
        torch.ones_like(range_vals),
        range_vals
    )

    x_flat_norm = (x_flat - min_vals)/range_vals

    m = x_flat.mean(dim=1, keepdim=True)
    s = x_flat.std(dim=1, keepdim=True)

    return x_flat_norm.view(x.shape), m, s


def aiparse_pad(x, padding_size=32):
    """
    padding to n x 32
    """
    if len(x.shape) == 2:
        h, w = x.shape
    elif len(x.shape) == 3:
        _, h, w = x.shape
    elif len(x.shape) == 4:
        _, _, h, w = x.shape
    else:
        raise NotImplementedError("input array is not image!")

    ph = 0
    pw = 0
    if not (h % padding_size) == 0:
        ph = padding_size - (h % padding_size)

    if not (w % padding_size) == 0:
        pw = padding_size - (w % padding_size)

    return torch.nn.functional.pad(x, (0, pw, 0, ph), "constant", 0), h, w, ph, pw


norm_functions = {"zscore": zscore, "normalize": normalize}
