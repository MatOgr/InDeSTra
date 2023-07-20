import torch


class CycleGAN(torch.nn.Module):
    """Implements the CycleGAN model.

    Args:
        BaseModel (BaseModel class): Base class for all models providing set of common (also abstract) methods.
    """

    def __init__(self, generator, discriminator, lambda_cycle=10.0):
        super().__init__()
        self.loss_names = [
            "D_A",
            "G_A",
            "cycle_A",
            "idt_A",
            "D_B",
            "G_B",
            "cycle_B",
            "idt_B",
        ]
        visual_names = ["real_", "fake_", "rec_"]
        if self.want_train and self.opt.lambdas_identity > 0.0:
            visual_names += ["idt_"]

        ### Network definition
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_cycle = lambda_cycle

    def forward(self, x, y):
        pass


class Generator(torch.nn.Module):
    def __init__(
        self, input_channels, output_channels, num_filters=64, architecture="resnet"
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.architecture = architecture

    def forward(self, x):
        pass


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels, num_filters=64, architecture="resnet"):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.architecture = architecture

    def forward(self, x):
        pass


class TestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
