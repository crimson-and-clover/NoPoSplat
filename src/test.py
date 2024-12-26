import os
from src.utils import read_write_model
from PIL import Image
import json

def crop_to_square_and_resize(image_path, output_dir):
    """
    Reads an image, crops it to a square, resizes it to 256x256, and saves it to the specified directory.

    :param image_path: Path to the input image file
    :param output_dir: Directory where the processed image will be saved
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Calculate dimensions for square crop
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim

            # Crop the image
            img_cropped = img.crop((left, top, right, bottom))

            # Resize the image to 256x256
            img_resized = img_cropped.resize((256, 256))

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Save the processed image
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            img_resized.save(output_path)

            print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


image_names = ["DSCF4679.JPG", "DSCF4673.JPG", "DSCF4686.JPG"]
os.makedirs("./datasets/room", exist_ok=True)
cameras, images, points = read_write_model.read_model("/root/hdd/datasets/room/sparse/0/")

image_params = {}

for image_name in image_names:
    crop_to_square_and_resize(os.path.join("/root/hdd/datasets/room/images/", image_name), "./datasets/room")
    image = next(filter(lambda x: x.name == image_name, images.values()), None)
    camera = next(filter(lambda x: x.id == image.camera_id, cameras.values()), None)
    image_params[image_name] = {
        "image_name" : image_name,
        "camera": {
            "focal_x" : camera.params[0] * 256 / min(camera.width, camera.height),
            "focal_y" : camera.params[1] * 256 / min(camera.width, camera.height),
            "center_x": camera.params[2] * 256 / camera.width,
            "center_y": camera.params[3] * 256 / camera.height,
        },
        "pose": {
            "rotation": image.qvec.tolist(),
            "translation": image.tvec.tolist(),
        },
    }

with open("./datasets/room/image_params.json", mode="w", encoding="utf-8") as f:
    json.dump(image_params,f, indent=2)   


pass