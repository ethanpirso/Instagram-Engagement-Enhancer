from google.cloud import vision
import os

def authenticate_vision_api():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials/instagram-engagement-enhancer-604d40340a18.json'

def get_image_labels(image_path):
    """Detect labels in the image using Google Vision API."""
    authenticate_vision_api()
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    return [label.description for label in labels]

def label_images(image_paths):
    """Return a dictionary with image paths and their corresponding labels."""
    labels_dict = {}
    for image_path in image_paths:
        labels = get_image_labels(image_path)
        labels_dict[image_path] = labels
    return labels_dict

if __name__ == '__main__':
    image_paths = ['data/aashnashroff_969148_3000403601659402518_25980_65/2022-12-24_15-33-23_UTC_1.jpg']
    labels_dict = label_images(image_paths)
    for image_path, labels in labels_dict.items():
        print(f'Image: {image_path}')
        print(f'Labels: {", ".join(labels)}\n')