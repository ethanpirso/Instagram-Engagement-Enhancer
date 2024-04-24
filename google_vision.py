from google.cloud import vision
import os

def authenticate_vision_api():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/My-Project-74932-95eea7d0bd8b.json'

def get_image_labels(image_path):
    """Detect labels in the image using Google Vision API."""
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
