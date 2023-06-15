import torch
import itertools
from base_model import BaseModel
from . import network


class CycleGANModel(BaseModel):
    """Implements the CycleGAN model.

    Args:
        BaseModel (BaseModel class): Base class for all models providing set of common (also abstract) methods.
    """

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
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
        self.netG_A = network.ResnetGenerator()
