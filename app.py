from flask import Flask, render_template, request
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

model = load_model("model/saved_model.h5")

def predict_logo(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
    label = "Fake" if prediction > 0.5 else "Real"
    return label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            result, confidence = predict_logo(file_path)
            return render_template("index.html", result=result, confidence=confidence, image=file.filename)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
