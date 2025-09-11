<h1 align="center">Whistle Drone</h1>
<h2 align="center">Real‑time whistle detection (DSP + AI) controlling a DJI Tello.</h2>

<!-- Action buttons -->
<p align="center">
  <a href="https://youtu.be/UR_oug2gfhU">
    <img alt="Watch demo on YouTube"
         src="https://img.shields.io/badge/Watch%20demo-YouTube-red?logo=youtube">
  </a>
  <a href="docs/Poster.pdf">
    <img alt="Poster (PDF)"
         src="https://img.shields.io/badge/Poster-PDF-blue?logo=adobeacrobatreader">
  </a>
  <a href="docs/Slides.pdf">
    <img alt="Slides (PDF)"
         src="https://img.shields.io/badge/Slides-PDF-green?logo=microsoftpowerpoint">
  </a>
</p>

## 🚀 Overview
Control a **DJI Tello** using your **whistle**.  
The system uses:
- **DSP** (fast, low-latency) for continuous pitch/volume analysis.
- **Optional AI** (TensorFlow Lite) as a periodic “authenticator” to ensure the drone reacts only to the pilot’s whistle.

This combo keeps control **responsive** while reducing **false triggers**.

---

## 🧠 How it works (short)
- Microphone audio → band-pass filter → pitch & volume detection (Aubio + SciPy).
- DSP decides motion (takeoff, altitude via loudness, forward/yaw via pitch & slope).
- AI gate (TFLite) runs periodically; if it matches your whistle, commands are allowed.

---

## 🎥 Demo
- Full video: **[YouTube – Whistle Drone](https://youtu.be/UR_oug2gfhU)**

---

## 📄 Docs
- **[Poster (PDF)](docs/Poster.pdf)**
- **[Slides (PDF)](docs/Slides.pdf)**

> Make sure the filenames above match your `docs/` folder.

---

## 📦 Folder Structure
Whistle_Drone/
├─ src/
│ └─ whistle_flight.py
├─ model/
│ └─ model.tflite 
├─ docs/
│ ├─ Poster.pdf
│ ├─ Slides.pdf
├─ requirements.txt
├─ .gitignore
└─ LICENSE

2) Install deps
pip install -r requirements.txt


If TensorFlow is not needed, you can comment it out in requirements.txt.

▶️ Run
# SIM mode (HUD only)
python src/whistle_flight.py --mode sim

# REAL drone (Tello only)
python src/whistle_flight.py --mode real

# Both (simulate + real feed)
python src/whistle_flight.py --mode sim+real


Keyboard

A = Active mode

S = Sleep mode (lands/holds)

E = Emergency stop

Q = Quit

🎚️ Audio tips

Use a decent mic.

If input is quiet, increase MIC_GAIN (in code) or your OS mic level.

Tune VOL_GATE, PITCH_MIN/MAX to your whistle.

⚙️ Requirements (tested)

Python 3.12

Windows 10/11

DJI Tello, djitellopy

See requirements.txt for exact package versions.

🔒 License

This project is released under the MIT License. See LICENSE.
