import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
from itertools import cycle
from labels import COCO_LABELS_ENGLISH, COCO_LABELS_SPANISH



def load_model():
    model_url = "https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/TensorFlow2/variations/d7/versions/1"
    return hub.load(model_url)  


def load_image(path):
    image = Image.open(path).convert('RGB')
    return image


def preprocess_image(image, target_size=(512, 512)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor


def run_detector(detector, image_tensor):
    detector_output = detector(image_tensor)
    result = {key: value.numpy() for key, value in detector_output.items()}
    return result


def get_class_names(class_ids, labels):
    class_names = [labels.get(id, "Unknown") for id in class_ids]
    return class_names


def draw_boxes(image, boxes, class_names, scores, threshold=0.5):
    draw = ImageDraw.Draw(image)
    #font = ImageFont.load_default()
    font = ImageFont.truetype("./static/fonts/Roboto-Regular.ttf", 15)
    colors = cycle(["blue", "green", "red", "purple", "orange", "pink", "yellow"])

    for i in range(boxes.shape[0]):
        if scores[i] >= threshold:
            color = next(colors)
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image.width, xmax * image.width,
                                          ymin * image.height, ymax * image.height)
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)
            text = f"{class_names[i]}: {scores[i]:.2f}"
            #text = f"{class_names[i]}"
            draw.text((left, top), text, fill=color, font=font)

    return image


def annotate_image(detector, image_path, labels=COCO_LABELS_SPANISH, threshold=0.5):
    image = load_image(image_path)
    image_tensor = preprocess_image(image)
    result = run_detector(detector, image_tensor)
    boxes = result["detection_boxes"][0]
    class_ids = result["detection_classes"][0].astype(int)
    scores = result["detection_scores"][0]
    class_names = get_class_names(class_ids, labels)
    annotated_image = draw_boxes(image, boxes, class_names, scores, threshold)
    return annotated_image



def main():
    if len(sys.argv) < 2:
        print("Usage: python3 project.py <image_path>")
    else:
        image_path = sys.argv[1]

    detector = load_model()
    image_with_boxes = annotate_image(detector, image_path)
    image_with_boxes.save("./static/images/processed/annotated_image.jpg")
    image_with_boxes.show()


if __name__ == "__main__":
    main()
