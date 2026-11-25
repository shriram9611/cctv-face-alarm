# Laptop Face Recognition Alarm

A semester prototype that uses the laptop's built-in camera to recognize locally enrolled people. The app plays an alarm through the laptop's built-in speakers when it sees an unknown face. It does not use a microphone or identify sounds.

## Requirements

- macOS, Windows, or Linux laptop with a built-in or already available camera
- Python 3.10 or newer (Python 3.12 is a good default for this OpenCV version)
- Internet for the one-time package and model downloads
- Laptop speaker or audio output for the alarm

After setup, face detection, face matching, and enrollment run locally. The app does not send camera frames or face data to a server.

## Setup on macOS

```sh
cd "/Users/mac/Documents/ChatGPT/cctv project"
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_models.py
```

The first camera run may prompt macOS for camera permission. Allow access to the terminal or app that runs Python.

## Enroll a known person

```sh
python camera_alarm.py enroll "Your Name"
```

Face the camera and press **S** when exactly one face appears. Press **Q** to cancel. Repeat enrollment in a different light or with a slightly different pose to add another sample for the same person.

## Start monitoring

```sh
python camera_alarm.py scan
```

The preview labels each face. A known person is shown by name. An unknown face must remain visible across multiple frames before the app sounds the alarm. A cooldown prevents repeated alerts. Press **Q** in the preview to stop.

The default match threshold is a starting point, not a measured accuracy claim. Increase it to make matching stricter, then record both false alarms and missed unknown detections during your evaluation.

## Local data and privacy

- Enrolled face embeddings stay in `data/known_faces.json` on this laptop.
- The generated alarm tone stays in `data/alarm.wav`.
- Do not enroll someone without their permission. Delete `data/known_faces.json` to remove enrolled faces.
- The camera preview is processed in memory and is not saved.

## Semester implementation stages

1. Confirm camera access and display a live preview.
2. Enroll local known-face references.
3. Compare camera faces with enrolled references.
4. Add the unknown-person status and laptop-speaker alarm.
5. Evaluate with consenting participants under different lighting and distances; record false accepts and false rejects.
6. Tune the threshold and prepare a demonstration and results report.

External buzzers, extra cameras, phones, cloud storage, and sound recognition are outside this project.
