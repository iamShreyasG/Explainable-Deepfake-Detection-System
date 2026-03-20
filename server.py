from flask import Flask, render_template, request, redirect, url_for
from final_app import inference
import os
import uuid

app = Flask(__name__)

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------

UPLOAD_FOLDER = os.path.join('static', 'videos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ----------------------------------------
# ROUTES
# ----------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    # -------------------------------
    # Validate File Upload
    # -------------------------------
    if "video" not in request.files:
        return redirect(url_for("upload_page"))

    video_file = request.files["video"]

    if video_file.filename == "":
        return redirect(url_for("upload_page"))

    # -------------------------------
    # Save Uploaded Video
    # -------------------------------
    video_id = str(uuid.uuid4())
    filename = f"{video_id}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    video_file.save(video_path)

    print(f"[INFO] Saved video to {video_path}")

    # -------------------------------
    # Run AI Inference
    # -------------------------------
    try:
        result = inference(video_path)

        if result.get("prediction") == "Error":
            return (
                "<h2>Error during AI analysis</h2>"
                f"<pre>{result.get('error')}</pre>",
                500,
            )

    except Exception as e:
        print(f"[Inference Exception] {e}")
        return (
            "<h2>Unexpected Server Error</h2>"
            f"<pre>{e}</pre>",
            500,
        )

    print("[INFO] Inference successful")

    # -------------------------------
    # Render Result Page
    # -------------------------------
    return render_template(
        "result.html",
        video_preview=f"/static/videos/{filename}",
        **result
    )


# ----------------------------------------
# MAIN
# ----------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)