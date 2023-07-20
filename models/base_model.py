import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from network import get_scheduler


class BaseModel(ABC):
    def __init__(self, opt) -> None:
        self.opt = opt
        self.device = torch.device("cuda:0" if opt.gpu_ids else "cpu")
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.optimizers = []
        self.image_paths = []
        self.model_names = []

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def update_lr(self):
        """Update learning rate of all networks"""
        prev_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]["lr"]
        print(f"Learning rate updated from {prev_lr:.7f} to {lr:.7f}")

    def setup(self, opt):
        """Load networks settings and print teir details.

        Args:
            opt (Option class): all flags and values set for this experiment
        """
        if self.want_train:
            self.schedulers = [
                get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]
        if not self.isTrain or opt.continue_training:
            self.load_network(opt.model_iteration or "last_checkpoint")
        self.print_networks(opt.verbose)

    def eval(self):
        """Turn the eval mode during test time."""
        net = getattr(self, "netG_A")
        net.eval()

    def test(self):
        """Wrapper function for the <forward> method using no_grad() context."""
        with torch.no_grad():
            self.forward()
            # do sth else???

    def save_networks(self, epoch):
        """Save all networks with the current epoch number.

        Args:
            epoch (int): current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f"{epoch}_net_{name}.pth"
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                # net.to(self.device)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{epoch}_net_{name}.pth"
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print(f"loading the model from {load_path}")
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(
                #     state_dict.keys()
                # ):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(
                #         state_dict, net, key.split(".")
                #     )
                net.load_state_dict(state_dict)

    def print_networks(self, verbose=True):
        """Print details of the used network architecture.

        Args:
            verbose (bool, optional): if vrebose: print all the details. Defaults to True.
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    f"[Network {name}] Total number of parameters : {num_params / 1e6} M"
                )
        print("-----------------------------------------------")

    # def set_requires_grad(self, nets, requires_grad=False):
    #         """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    #         Parameters:
    #             nets (network list)   -- a list of networks
    #             requires_grad (bool)  -- whether the networks require gradients or not
    #         """
    #         if not isinstance(nets, list):
    #             nets = [nets]
    #         for net in nets:
    #             if net is not None:
    #                 for param in net.parameters():
    #                     param.requires_grad = requires_grad
