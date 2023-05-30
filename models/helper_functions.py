import functools
import torch
from torch import nn
from network import ResnetGenerator, UnetGenerator


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError(f"normalization layer {norm_type} is not found")
    return norm_layer


def init_weights(network, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        network (nn.Module)   -- network to be initialized
        init_type (str)       -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)     -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"initialization method {init_type} is not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}...")
    network.apply(init_func)


def init_network(network, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        network (network)      -- the network to be initialized
        init_type (str)        -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list)     -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        network.to(gpu_ids[0])
        network = nn.DataParallel(network, gpu_ids)
    init_weights(network, init_type, init_gain=init_gain)
    return network


def create_generator(
    in_channels,
    out_channels,
    last_layer_filters,
    net_type="resnet_9blocks",
    norm_layer="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
):
    """Create a generator model."""
    network = None
    norm_layer = get_norm_layer(norm_type=norm_layer)

    if net_type == "resnet_9blocks":
        network = ResnetGenerator(
            in_channels,
            out_channels,
            last_layer_filters,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
        )
    elif net_type == "resnet_6blocks":
        network = ResnetGenerator(
            in_channels,
            out_channels,
            last_layer_filters,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=6,
        )
    elif net_type == "unet_128":
        network = UnetGenerator(
            in_channels,
            out_channels,
            7,
            last_layer_filters,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )
    elif net_type == "unet_256":
        network = UnetGenerator(
            in_channels,
            out_channels,
            8,
            last_layer_filters,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )
    else:
        raise NotImplementedError(
            f"Generator model name [{net_type}] is not recognized"
        )
    return init_network(network, init_type, init_gain, gpu_ids)


def create_discriminator():
    # TODO
    pass


class GANLoss(nn.Module):
    # TODO
    def __init__(self) -> None:
        super().__init__()
