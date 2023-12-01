from flask import Flask, render_template, request
import os
from project import annotate_image, load_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        print(file)
        if file.filename == '':
            return 'No selected file'
        if file:
            filepath = os.path.join('static/images/uploads', file.filename)
            file.save(filepath)
            
            detector = load_model()
            annotated_image = annotate_image(detector, filepath)
            output_path = 'static/images/processed/processed_image.jpg'
            annotated_image.save(output_path)

            return render_template('result.html', image_path=output_path)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
