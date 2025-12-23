from flask import Flask, render_template, request
from detector import process_video
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    ds_video_url = None
    bt_video_url = None

    if request.method == "POST":
        video_file = request.files.get("video")

        if not video_file or video_file.filename == "":
            return "No file selected"

        input_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(input_path)

        # 🔥 Run processing
        result = process_video(input_path)

        # 🔗 Static URLs (Chrome compatible)
        ds_video_url = "/static/output/deepsort_result.mp4"
        bt_video_url = "/static/output/bytetrack_result.mp4"

    return render_template(
        "index.html",
        result=result,
        ds_video_url=ds_video_url,
        bt_video_url=bt_video_url
    )


if __name__ == "__main__":
    app.run(debug=False)
