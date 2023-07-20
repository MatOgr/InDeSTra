import streamlit as st
from io import BytesIO
from yaml import load as load_yaml, SafeLoader
from PIL import Image

import os
import sys
import subprocess


# SCRIPT_DIR = os.path.dirname(os.path.abspath('/home/tukut/Documents/University/seminary/PBR/InDeSTra/demo-app/models'))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
print(os.path.join(os.path.dirname(__file__), '../models_package'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import models_package.inference as models_inference


@st.cache_resource
def get_config():
    config_path = os.environ.get("CONFIG_FILE_PATH", "config.yml")
    with open(config_path, "r") as f:
        config = load_yaml(f, Loader=SafeLoader)
    return config


@st.cache_resource
def create_images_dict(
    styles_list: list = ["ArtDeco", "Rustic"],
    img_test_path: str = "../datasets/test_data",
):
    images_dict = {"Select Image": None, "Upload Image": "upload"}
    for i, style in enumerate(sorted(styles_list * 4)):
        images_dict[f"{style}_{i % 4 + 1}"] = (
            f"{img_test_path}/{style}/{i % 4 + 1}.jpg",
            f"{img_test_path}/{style}/{i % 4 + 1}_mask.jpg",
        )
    return images_dict


def choose_file():
    uploaded_file = st.file_uploader(
        "Choose an image",
        help="Browse your files and upload an interior image of your choice",
    )
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        return image, uploaded_file.name
    return None


### TODO: Inherit methods for creating the segmentation mask
def create_mask(
    image_path, segmentation_script, result_mask_path="", mask_file_name="", state=None
) -> str:
    if state:
        state.text("Creating segmentation mask (this may take up to ~1 minute)...")
    try:
        result = subprocess.run(
            [
                segmentation_script,
                "/".join(segmentation_script.split("/")[:-1]),
                "-i",
                image_path,
                "-r",
                result_mask_path,
                "-f",
                mask_file_name,
            ]
        )
    except:
        st.error("Segmentation script failed to run.")
    print(f"Result of inference through bash script in web client: {result}")
    if state:
        state.text('Done! (You can now click "Generate Design" to see the result.)')
    return result


def create_new_design(image_path: str, style: str, model_path: str):
    subprocess.call(
        [
            "python3",
            "../test_GAN.py",
            "--dataroot",
            image_path,
            "--name",
            style,
            "--model",
            "test",
            "--no_dropout",
            "--checkpoints_dir",
            model_path,
            "--results_dir",
            "./results/",
            "--num_test",
            "1",
            "--gpu_ids",
            "-1",
        ]
    )


def generate_image(
    style, image_path, test_image: bool = False, config=None, state=None
):
    #################### Segmentation mask part
    print(f"Image path: {image_path}")
    file_name = image_path.split("/")[-1]
    mask_path = config["images-masks-path"]
    if test_image:  # We can used already existing segmentation masks
        if state:
            state.text("Using existing segmentation mask...")
        Image.open(image_path.split(".")[0] + "_mask.jpg").save(
            f"{mask_path}/{file_name}"
        )
    else:  # We need to create a segmentation mask
        # mask_path = config["images-masks-path"]
        # mask_file_name = "/segmentation.jpg"
        create_mask(
            image_path,
            config["segmentation-script"],
            mask_path,
            file_name,
            state,
        )
    mask_image = Image.open(f"{mask_path}/{file_name}")
    st.image(mask_image, caption="Generated segmentation mask.", use_column_width=True)
    # image.save(f'{config["images-received-path"]}/{mask_file_name}_mask.jpg')

    #################### Style transfer part
    state.text("Generating new design...")
    # create_new_design(mask_path, style, config["transfer-model-base-path"])
    image, image_path = models_inference.perform_inference(
        network_file_path=f'{config["transfer-model-base-path"]}/{style}/latest_net_G.pth',
        image_path=f"{mask_path}/{file_name}",
    )
    print("Image path from inference.py is:", image_path)
    state.text("Done! Enjoy the results!")
    # image = Image.open(
    #     # f"{config['images-result-path']}/{style}/test_latest/images/{file_name.split('.')[0]}_fake.png"
    #     image_path
    # )
    os.remove(f"{mask_path}/{file_name}")
    return image


def call_for_inference(data):
    image = generate_image(
        data["model_style"],
        data["image"],
        data["test_image"],
        data["config"],
        data["state"],
    )
    return image


### Load config params
config = get_config()
IMAGES_DICT = create_images_dict(config["styles-list"], config["images-test-path"])


######################          website elements          ######################
st.title("Interior Design Style Transfer")
image_loaded = None

uploaded_file = st.selectbox(
    "Choose an image",
    help="Browse your files and upload an interior image of your choice",
    options=IMAGES_DICT.keys(),
)
if IMAGES_DICT[uploaded_file] not in [None, "upload"]:
    print(uploaded_file)
    image_path, mask_path = IMAGES_DICT[uploaded_file]
    print("\n", image_path, "\n")
    image_loaded = Image.open(image_path)
    test_image = True

elif IMAGES_DICT[uploaded_file] == "upload":
    try:
        image_loaded, uploaded_file = choose_file()
        test_image = False
    except:
        print("Nothing loaded yet")

if image_loaded is not None:
    print("\nFile uploaded:", uploaded_file)
    if test_image:
        image_path = f'{config["images-test-path"]}/{uploaded_file.split("_")[0]}/{uploaded_file.split("_")[1]}.jpg'
    else:
        image_path = f'{config["images-received-path"]}/{uploaded_file}'
    print(f"\n\nSave image path: {image_path}\n\n")
    image_loaded.save(image_path)
    st.image(image_loaded, caption="Chosen test image.", use_column_width=True)

    chosen_style = st.selectbox(
        "What interior style are you interested in?", config["styles-list"]
    )
    if st.button("Generate Design"):
        data_load_state = st.text("Generating design...")
        generated_image = call_for_inference(
            {
                "image": image_path,
                "model_style": chosen_style,
                "test_image": test_image,
                "config": config,
                "state": data_load_state,
            }
        )
        data_load_state.text("Done!")
        st.image(generated_image)
