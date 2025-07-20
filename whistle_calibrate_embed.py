#!/usr/bin/env python3
"""
whistle_calibrate_embed.py
–––––––––––––––––––––––––––
Interactive recorder that

1.  creates    users/<USER>/{recordings,embeddings}/
2.  lets you   (re)record short WAV clips for six whistle commands
3.  extracts   a simple MFCC-mean embedding from every clip
4.  saves      users/<USER>/embeddings/embeddings.npy  (shape: N×D)
               users/<USER>/embeddings/labels.json     (parallel list)

Dependencies
  pip install sounddevice scipy librosa
"""
import argparse, json, os, pathlib, time, sys, wave, uuid
import numpy as np, sounddevice as sd, librosa

SR          = 22_050      # Hz – small model, good enough for whistle
DURATION    = 1.5         # seconds per example
N_MFCC      = 20          # length of each embedding
MIN_VOL_RMS = 0.005       # gate so silence isn't stored
COMMANDS = ["forward","back","left","right","fly","land"]

# ---------------------------------------------------------------------------

def record_clip(path: pathlib.Path):
    print(f"▶  Whistle NOW… ({DURATION}s)")
    audio = sd.rec(int(DURATION*SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    rms = np.sqrt(np.mean(audio**2))
    if rms < MIN_VOL_RMS:
        print("⚠  Very quiet – discarded.")
        return None
    wav = (audio * 32767).astype("<i2").tobytes()
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(wav)
    print(f"✔  Saved {path.name}")
    return path

def mfcc_mean(wav_path: pathlib.Path):
    y, _ = librosa.load(wav_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    return mfcc.mean(axis=1)        # shape (20,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="username / folder name")
    ap.add_argument("--reuse", action="store_true",
        help="If recordings exist, keep them & just rebuild embeddings")
    args = ap.parse_args()

    user_dir       = pathlib.Path("users")/args.user
    rec_dir        = user_dir/"recordings"
    emb_dir        = user_dir/"embeddings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # STEP 1 – (optionally) make new recordings
    if not args.reuse:
        # drop old recordings
        for p in rec_dir.glob("*.wav"): p.unlink()
        print(f"\n🎙  Recording whistle examples for user: {args.user}")
        for cmd in COMMANDS:
            print(f"\n=== {cmd.upper()} ===")
            for i in range(3):                       # 3 clips per command
                input("Press ⏎ then whistle…")
                fname = f"{cmd}_{i}_{uuid.uuid4().hex[:8]}.wav"
                _ = record_clip(rec_dir/fname)

    # ---------------------------------------------------------------------
    # STEP 2 – build embedding matrix
    print("\n🔎  Extracting embeddings…")
    X, y = [], []
    for wav_path in sorted(rec_dir.glob("*.wav")):
        cmd = wav_path.stem.split('_')[0]
        if cmd not in COMMANDS: continue
        try:
            X.append(mfcc_mean(wav_path)); y.append(cmd)
        except Exception as e:
            print(f"   skipped {wav_path.name}: {e}")

    if not X:
        print("❌  No usable recordings – abort.")
        sys.exit(1)

    np.save(emb_dir/"embeddings.npy", np.vstack(X))
    (emb_dir/"labels.json").write_text(json.dumps(y))
    print(f"✅  Saved {len(y)} embeddings to {emb_dir}")

if __name__ == "__main__":
    main()
