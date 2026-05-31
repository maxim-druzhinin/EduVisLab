import os
import sys
import subprocess


def run(cmd):
    subprocess.run(cmd, check=True)


def setup_dover():
    if os.path.exists("/tmp/DOVER/pretrained_weights/DOVER.pth"):
        print("DOVER уже установлен ✅")
        return

    run(["rm", "-rf", "/tmp/DOVER"])
    run(["git", "clone", "https://github.com/QualityAssessment/DOVER.git", "/tmp/DOVER", "-q"])

    run(["sed", "-i", "/^torch/d", "/tmp/DOVER/requirements.txt"])

    run([sys.executable, "-m", "pip", "install", "-r", "/tmp/DOVER/requirements.txt", "-q"])
    run([sys.executable, "-m", "pip", "install", "-e", "/tmp/DOVER", "--no-deps", "-q"])

    os.makedirs("/tmp/DOVER/pretrained_weights", exist_ok=True)

    run([
        "wget", "-q",
        "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth",
        "-O", "/tmp/DOVER/pretrained_weights/DOVER.pth",
    ])

    print("DOVER установлен ✅")


def setup_mediapipe_face_detector():
    if os.path.exists("/tmp/face_detector.tflite"):
        print("MediaPipe face detector уже скачан ✅")
        return

    run([
        "wget", "-q",
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
        "-O", "/tmp/face_detector.tflite",
    ])

    print("MediaPipe face detector скачан ✅")


def setup_all_models():
    setup_dover()
    setup_mediapipe_face_detector()


if __name__ == "__main__":
    setup_all_models()