import numpy as np
from processing.license_plate_detection import detect_license_plate
from processing.license_plate_sing_recognitions import sign_recognitions


def perform_processing(image: np.ndarray, chars_tamplate) -> str:
    print(f'image.shape: {image.shape}')
    char_images, char_paths = chars_tamplate
    pts, image = detect_license_plate(image)
    # print(pts)
    if pts is not None:
        text = sign_recognitions(image, pts, char_images, char_paths)
        if text is not None:
            if len(text) > 7:
                return text[:7]
            return text
    return 'PO12345'
