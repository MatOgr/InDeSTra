from model import get_model_architecture


def create_model(model_arch="resnet", model_type="generator"):
    """Creates a model imported from the appripriate module.

    Args:
        model_arch (str, optional): network's architecture wanted to use. Defaults to 'resnet'.
        model_type (str, optional): type of network wanted to use. Available options: ['generator', 'segmenter']. Defaults to 'generator'.
    """
    if model_type == "generator":
        model = get_model_architecture(model_type, model_arch)

    pass


def perform_inference(input, model, model_type="generator"):
    """Universal method for inference.

    Args:
        input (obj): object to be passed to the model.
        model (torch.nn.Model): model to be used for inference.
        model_type (str, optional): type of from the available options {'segmenter', 'generator}. Defaults to 'generator'.
    """
    pass


def prepare_input(input, model_type="generator"):
    """Universal method for preparing input.

    Args:
        input (obj): object to be passed to the model.
        model_type (str, optional): type of model from the available options {'segmenter', 'generator}. Defaults to 'generator'.
    """
    pass
