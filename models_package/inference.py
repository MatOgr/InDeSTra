from .utils import (
    InteriorDataset,
    get_model_architecture_instance,
    image_to_tensor,
    infere_image,
    save_img,
    tensor_to_image,
)
from PIL import Image

from torchvision import transforms


def create_model(
    model_arch="resnet",
    model_type="generator",
    network_file_path: str = "latest_net_G.pth",
):
    """Creates an instance of chosen model imported from the appripriate module.

    Args:
        model_arch (str, optional): network's architecture wanted to use. Defaults to 'resnet'.
        model_type (str, optional): type of network wanted to use. Available options: ['generator', 'segmenter']. Defaults to 'generator'.
    """
    model_instance = get_model_architecture_instance(model_arch)
    # setup the model for a specific action (TEST for now)
    model_instance.load_from_dict(network_file_path)
    return model_instance


def prepare_input(image, model_type="generator", device="cpu"):
    """Universal method for preparing input.

    Args:
        input (obj): object to be passed to the model.
        model_type (str, optional): type of model from the available options {'segmenter', 'generator}. Defaults to 'generator'.
    """

    tensor_image = image_to_tensor(image).to(device)
    return tensor_image


def perform_inference(
    network_file_path="../models/ArtDeco/latest_net_G.pth",
    image_path: str = "./demo-app/images/tests/test-image-real.png",
    model_type="generator",
    autosave=False,
):
    """Universal method for inference.

    Args:
        input (obj): object to be passed to the model.
        model (torch.nn.Model): model to be used for inference.
        model_type (str, optional): type of from the available options {'segmenter', 'generator}. Defaults to 'generator'.
    """
    model = create_model(network_file_path=network_file_path, model_type=model_type)
    print(model)

    # prepare input image
    input = Image.open(image_path).convert("RGB")
    tensor_input = prepare_input(input)
    print(tensor_input)

    result = infere_image(model, tensor_input)
    image_array = tensor_to_image(result)
    new_image_path = "_fake.".join((image_path[:-4], image_path[-5:].split(".")[-1]))
    if autosave:
        save_img(image_array, new_image_path)
    return image_array, new_image_path


if __name__ == "__main__":
    perform_inference()
    print("That's all I wanted – bye!")
