import numpy as np
import tensorflow as tf
from PIL import Image
import os
from project import preprocess_image, load_image, get_class_names, draw_boxes

def test_preprocess_image():
    # Creating a dummy image
    dummy_image = Image.new('RGB', (100, 100), color = 'red')
    dummy_image_path = './static/images/dummy_image.jpg'
    dummy_image.save(dummy_image_path)

    # Expected values
    expected_shape = (1, 512, 512, 3)
    expected_dtype = tf.uint8

    # Calling the function under test
    preprocessed_image = preprocess_image(load_image(dummy_image_path))

    # Checking the shape
    assert preprocessed_image.shape == expected_shape, "Shape does not match expected shape."

    # Checking the data type
    assert preprocessed_image.dtype == expected_dtype, "Data type does not match expected dtype."

    # Cleaning up dummy file
    os.remove(dummy_image_path)


def test_get_class_names():
    class_ids = np.array([1, 2, 2, 3, 99])
    dummy_labels = {1: "Person", 2: "Bicycle", 3: "Car"}
    
    # Expected result
    expected_class_names = ["Person", "Bicycle", "Bicycle", "Car", "Unknown"]

    # Calling the function under test
    class_names = get_class_names(class_ids, dummy_labels)

    # Checking if the output matches the expected result
    assert class_names == expected_class_names, "Class names do not match expected values."


def test_draw_boxes():
    # Creating a dummy image
    dummy_image = Image.new('RGB', (100, 100), color = 'white')
    boxes = np.array([[0.1, 0.1, 0.5, 0.5]])
    class_names = ["Test"]
    scores = np.array([0.9])

    # Calling the function under test
    image_with_boxes = draw_boxes(dummy_image, boxes, class_names, scores, threshold=0.5)

    # Converting the image to a numpy array for testing
    image_array = np.array(image_with_boxes)

    # Checking if there are any red pixels (bounding box color) in the image
    red_pixels = np.any(image_array[:, :, 0] == 255)
    assert red_pixels, "No red boxes were drawn in the image."
