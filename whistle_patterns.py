import numpy as np
import librosa
from collections import deque

SR = 22050

class RollingBuffer:
    """Maintain a rolling audio buffer of fixed length in samples."""
    def __init__(self, max_seconds: float, sr: int = SR):
        self.max_samples = int(max_seconds * sr)
        self.sr = sr
        self.buf = deque()
        self.n = 0

    def append(self, data: np.ndarray):
        self.buf.append(data)
        self.n += len(data)
        while self.n > self.max_samples:
            item = self.buf.popleft()
            self.n -= len(item)

    def get(self) -> np.ndarray:
        if not self.buf:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self.buf))

def detect_onsets(y: np.ndarray, sr: int = SR):
    """Return onset times (in samples) of detected whistles."""
    if len(y) == 0:
        return []
    # use simple energy-based onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
    return onset_frames

def pitch_slope(y: np.ndarray, sr: int = SR):
    """Return slope of dominant pitch in Hz/s."""
    if len(y) < sr // 4:
        return 0.0
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    idx = mags.argmax(axis=0)
    pitch_track = pitches[idx, range(pitches.shape[1])]
    valid = pitch_track[pitch_track>0]
    if len(valid) < 2:
        return 0.0
    t = np.arange(len(valid)) / (len(pitch_track) / len(y) / sr)
    p1, p2 = valid[0], valid[-1]
    return (p2 - p1) / (t[-1] - t[0])


def classify_pattern(buffer: RollingBuffer, sr: int = SR):
    """Classify whistle pattern from audio buffer. Returns command or None."""
    y = buffer.get()
    if len(y) < sr:
        return None
    onsets = detect_onsets(y, sr)
    if len(onsets) == 0:
        return None
    # heuristics
    gaps = np.diff(onsets) / sr
    n_whistles = len(onsets)
    if n_whistles >=3 and np.all(gaps[:2] < 0.5):
        return "emergency"
    # analyze single whistle / glide using pitch slope
    first = onsets[0]
    end = onsets[1] if len(onsets)>1 else len(y)
    clip = y[first:end]
    slope = pitch_slope(clip, sr)
    if slope < -200:   # falling
        return "land"
    if slope > 200:    # rising
        return "rotate"
    if n_whistles==1:
        return "single"
    return None
