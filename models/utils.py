import numpy as np
from PIL import Image
import functools
from helper_functions import Identity

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import io, transforms

import network


class InteriorDataset(Dataset):
    def __init__(
        self, img_dir, annotations_file=None, transform=None, target_transform=None
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_paths = []
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = io.read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image


def get_norm_layer(normalization_type):
    if normalization_type == "instance":
        norm_layer = functools.partial(
            torch.nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    else:
        print(__name__, f"not implemented for argument #_{normalization_type}_#")
        norm_layer = Identity()
    return norm_layer


def get_model_architecture_instance(
    model_arch: str = "resnet",
    normalization_type: str = "instance",
    action="test",
    init_type="normal",
):
    if model_arch == "resnet":
        model_instance = network.ResnetGenerator(
            in_channels=3,
            out_channels=3,
            last_layer_filters=64,
            norm_layer=get_norm_layer(normalization_type),
            use_dropout=action != "test",
            n_blocks=9,
        )
    else:
        print(f"Creating a model for option |{model_arch}| not implemented")
        raise NotImplemented(model_arch)

    # init net for <device>
    model_instance.to(("cuda" and torch.cuda.is_available()) or "cpu")
    # init weights of the model if <action> == 'training'
    # if init_type == 'normal':
    return model_instance


def image_to_tensor(
    image,
    img_size=256,
    crop_size=256,
    normalize_values=(0.5, 0.5, 0.5),
    interpolation_mode=transforms.InterpolationMode.BICUBIC,
):
    composed_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation_mode),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(normalize_values, normalize_values),
        ]
    )
    image_tensor = composed_transforms(image)
    return image_tensor


def tensor_to_image(input_image, imtype=np.uint8):
    """Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = (
            image_tensor.cpu().float().numpy()
        )  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def infere_image(model, image):
    with torch.no_grad():
        result = model(image)
    return result


def save_img(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    print(f"Image saved to {image_path}")
