# Advanced Object Detection Web Application

#### Video Demo: [Watch Here](https://youtu.be/UBxP_5zdRHY?si=f2HUHMFWTy9eEmZZ)

#### Description

This project is a comprehensive object detection application accessible via command line and a web interface powered by Flask. Leveraging TensorFlow, TensorFlow Hub, and the EfficientDet d7 model, it processes images, detects objects, and annotates them with bounding boxes and labels.
For example, if this image is fed into the program:
![image](https://github.com/josedanieljarap/AdvancedObjectDetectionApp/assets/50277190/8f7e1fa5-ce60-42fc-8718-61fe25842e11)
The program will output this processed image:
![image](https://github.com/josedanieljarap/AdvancedObjectDetectionApp/assets/50277190/4307e586-5c95-425f-85a3-01a63eec7dd2)


## Project Structure

- **project.py**: The core of the application. It uses TensorFlow, TensorFlow Hub, PIL, and NumPy to process images and detect objects. The main logic resides here, and it includes a `main()` function for command line execution:
  ```bash
  python3 project.py <image_path>

This processes the image, annotates detected objects, and saves the output.

Key functions:

- `annotate_image()`: Central function calling:
  - `load_image()`: Reads the image using PIL.
  - `preprocess_image()`: Resizes and converts the image into a tensor.
  - `run_detector()`: Runs the detection model to get bounding boxes, class IDs, and scores.
  - `get_class_names()`: Maps class IDs to object names.
  - `draw_boxes()`: Draws bounding boxes and annotations on the image.

- **app.py**: The Flask web application allowing image upload and processing through a web interface. It handles both GET and POST methods, serving two HTML files:
  - `upload.html`: For image upload.
  - `result.html`: Displays the annotated image with an option to upload another image.

- **test_project.py**: Contains unit tests for key functions:
  - `preprocess_image()`
  - `get_class_names()`
  - `draw_boxes()`

- **labels.py**: Contains two dictionaries, `COCO_LABELS_ENGLISH` and `COCO_LABELS_SPANISH`, mapping class numbers to labels in English and Spanish.

## Flask Project Structure

- **static/**: Contains subfolders for CSS, fonts, images, and JavaScript.
  - `css/`
  - `fonts/`
  - `images/`: Stores processed, sample, and uploaded images.
  - `js/`

- **templates/**: Contains HTML files served by Flask.
  - `upload.html`: Form for image upload.
  - `result.html`: Displays the processed image and provides a link to upload another image.

## Additional Files

- **.gitignore**: Specifies files and folders to exclude from the repository, including the virtual environment (`venv`).
- **requirements.txt**: Lists third-party libraries required for the project:
  - TensorFlow
  - TensorFlow Hub
  - NumPy
  - PIL (Pillow)
  - Flask
  - pytest

## Design Choices

I chose the EfficientDet d7 model for its superior accuracy, despite its slower processing time. The trade-off was made to ensure the most precise object detection possible, which is critical for this application.
