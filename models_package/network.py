import torch
from torch import nn

import functools


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        last_layer_filters=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels, last_layer_filters, kernel_size=7, padding=0, bias=use_bias
            ),
            norm_layer(last_layer_filters),
            nn.ReLU(True),
        ]

        down_samp_layers = 2
        for i in range(down_samp_layers):
            multiplier = 2**i
            model += [
                nn.Conv2d(
                    last_layer_filters * multiplier,
                    last_layer_filters * multiplier * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(last_layer_filters * multiplier * 2),
                nn.ReLU(True),
            ]
        multiplier = 2**down_samp_layers
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    last_layer_filters * multiplier,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]
        for i in range(down_samp_layers):
            multiplier = 2 ** (down_samp_layers - i)
            model += [
                nn.ConvTranspose2d(
                    last_layer_filters * multiplier,
                    int(last_layer_filters * multiplier / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(last_layer_filters * multiplier / 2)),
                nn.ReLU(True),
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(last_layer_filters, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    def load_from_dict(self, network_file_path: str, device: str = "cpu"):
        print(f"Loading model from state dict: {network_file_path}")
        import os
        print(os.listdir('../'))
        print(os.listdir('../../'))
        print(os.listdir("/".join(network_file_path.split("/")[:-1])))
        state_dict = torch.load(network_file_path, map_location=device)
        self.load_state_dict(state_dict)
        print("\tModel loaded!")


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias) -> None:
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def get_padding(self, padding_type):
        p = 0
        padding = None
        if padding_type in ["reflect", "replicate"]:
            padding = nn.ReflectionPad2d(1)
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        return p, padding

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p, padding = self.get_padding(padding_type)
        if padding:
            conv_block += [padding]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if padding:
            conv_block += [padding]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
