
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import json
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Load trained CNN model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Load disease class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            error = "Please select a leaf image."

        else:
            filename = secure_filename(file.filename)
            extension = os.path.splitext(filename)[1].lower()

            if extension not in [".jpg", ".jpeg", ".png"]:
                error = "Please upload JPG, JPEG or PNG."

            else:
                try:
                    image = Image.open(file.stream).convert("RGB")
                    image = image.resize((128, 128))

                    image_array = np.array(image)
                    image_array = np.expand_dims(image_array, axis=0)

                    result = model.predict(image_array, verbose=0)

                    predicted_index = int(np.argmax(result[0]))

                    prediction = class_names[predicted_index]

                    confidence = round(
                        float(result[0][predicted_index]) * 100, 2
                    )

                    unique_name = str(uuid.uuid4()) + extension
                    save_path = os.path.join(
                        UPLOAD_FOLDER, unique_name
                    )

                    image.save(save_path)
                    image_path = "/" + save_path

                except (UnidentifiedImageError, OSError):
                    error = "Invalid image. Please upload a valid leaf image."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="Image must be smaller than 5 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
