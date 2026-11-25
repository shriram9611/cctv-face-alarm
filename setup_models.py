"""Download the official OpenCV Zoo models used for local face matching."""

from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        destination = MODEL_DIR / filename
        if destination.exists() and destination.stat().st_size > 1024:
            print(f"Already present: {destination}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {filename} from the OpenCV Zoo…")
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size <= 1024:
                raise RuntimeError("Downloaded file is unexpectedly small")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"Saved {destination}")
    print("Models are ready. Runtime face processing stays on this laptop.")


if __name__ == "__main__":
    main()
