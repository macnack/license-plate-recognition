import argparse
import json
from pathlib import Path
import time
import cv2

from processing.utils import perform_processing
from processing.utils_cv import load_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    char_paths = load_images("dane/fonts")
    char_images = [(cv2.imread(char_path, cv2.IMREAD_GRAYSCALE), char_path)
                   for char_path in char_paths]
    images_paths = sorted([image_path for image_path in images_dir.iterdir(
    ) if image_path.name.endswith('.jpg') or image_path.name.endswith('.JPG')])
    results = {}
    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue
        results[image_path.name] = perform_processing(
            image, char_images)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
