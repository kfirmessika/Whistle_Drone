# ──────────────────────────────────────────────────────────────────────────────
#  Whistle Drone Control 
#
#  Description
#  -----------
#  • Real‑time control of a DJI Tello using human whistles.
#  • Low‑latency DSP (pitch / volume / confidence) detects whistle frames.
#  • A TensorFlow Lite model periodically authenticates that the whistle
#    belongs to the pilot; this opens/closes an “AI gate”.
#  • Drone responds only when DSP detects a whistle AND the AI gate is open.
#
#  Features
#  --------
#  • Volume → target altitude (PID stabilized) with soft/hard ceilings.
#  • Pitch → forward speed; pitch slope → yaw rate (with smoothing/decay).
#  • Auto‑land on long silence or audio stall; battery checks; emergency stop.
#  • HUD showing audio stats, AI status, and telemetry (in sim and sim+real).
#
#  Modes
#  -----
#  --mode sim       : HUD only (no drone)
#  --mode real      : Real Tello; minimal window for keyboard input
#  --mode sim+real  : Real Tello + HUD
#
#  Controls
#  --------
#  A = Active   •   S = Sleep   •   E = Emergency stop   •   Q = Quit
#
# ──────────────────────────────────────────────────────────────────────────────


# (imports unchanged)
import time, sys, math, logging
from collections import deque
import numpy as np
import cv2
import sounddevice as sd
from aubio import pitch as AubioPitch
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.interpolate import interp1d  # resampling utility for AI buffer
from djitellopy import Tello

# TensorFlow (optional — only required for AI gate; DSP works without it)
try:
    import tensorflow as tf
    logging.info("TensorFlow imported successfully.")
except ImportError:
    logging.warning("TensorFlow not found. Install it (`pip install tensorflow`) to enable AI authentication.")
    tf = None


# ───────── USER LIMITS ─────────
ALT_BASE = 0.50             # m: desired base altitude with quiet whistle
ALT_MAX = 1.10              # m: soft ceiling (force descent if exceeded)
ABS_CEIL = 1.50             # m: hard ceiling (emergency stop if reached)
#  NOTE: If Tello's default takeoff hover > ABS_CEIL, emergency will trigger immediately.
#        Consider ABS_CEIL > 1.0–1.2 m to allow descent into (ALT_BASE..ALT_MAX).


# ───────── CONSTANTS ──────────
# Audio processing (latency vs. stability tradeoff)
SR, BUF, HOP = 44_100, 4096, 2048
VOL_GATE = 0.0005                 # RMS threshold to consider “whistle present”
PITCH_MIN, PITCH_MAX = 100.0, 3_000.0  # Hz, valid whistle pitch range
CONF_GATE, SILENCE_DB, TOLERANCE = 0.90, -80, 0.05  # aubio gates
MIC_GAIN = 30                     # input gain applied to mic frames
MAX_EXPECTED_WHISTLE_VOLUME = 0.5 # RMS estimate for “loud whistle” (tune for mic)

# AI model (TFLite) configuration
AI_MODEL_PATH = "whistle_ai_model/model.tflite"
AI_WHISTLE_CLASS_INDEX = 1        # matches labels.txt: class index for “My Whistle”
AI_PROBABILITY_THRESHOLD = 0.95   # AI confidence to open the gate
AI_INFERENCE_INTERVAL_SEC = 0.5   # run AI every N seconds on the rolling buffer
AI_GATE_FORGIVENESS_COUNT = 6     # negatives required to close gate after it opened

# Populated dynamically based on model input tensor
AI_MODEL_INPUT_SAMPLE_RATE = None
AI_MODEL_INPUT_SAMPLES = None
AI_MODEL_INPUT_DURATION_SEC = 1.0  # assumed window length the model expects (in seconds)

# Flight control mapping / rates
RATE_DZ = 20.0                    # Hz/s deadzone for pitch slope → yaw
K_FWD, K_YAW = 0.0002, 0.8        # proportional gains
MAX_FWD = 0.4                     # m/s cap
MAX_YAW = 2000                    # deg/s cap
CTRL_HZ = 10                      # control loop frequency
BAT_MIN = 25                      # % minimum battery
YAW_SMOOTH = 0.001                # yaw smoothing when slope ~ 0 but whistle continues

DECEL_FWD = 0.5                   # exponential decay factor (no‑whistle → slow to 0)
DECEL_YAW = DECEL_FWD * 1.5

# PID (altitude) — tuned for gentle response
PID_KP = 60
PID_KI = 10
PID_KD = 30
PID_OUT_MIN, PID_OUT_MAX = -60, 60
PID_I_MAX = 30.0

# Timeouts
LAND_TOUT = 100.0                 # s: silence duration before auto‑land
CALLBACK_TOUT = 100.0             # s: audio callback stall → auto‑land
IMU_WAIT = 5                      # s: initial IMU settle (kept small)

# Whistle persistence filter
WHISTLE_OFF_THRESHOLD = 3         # consecutive no‑whistle frames required to mark “off”


# ───────── LOGGING ────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)


# ───────── AUDIO PIPE ─────────
# Band‑pass around whistle range, aubio config, rolling state for pitch/volume
SOS_FILTER = butter(4, [PITCH_MIN / (SR / 2), PITCH_MAX / (SR / 2)], btype='band', output='sos')
sos_filter_state = sosfilt_zi(SOS_FILTER)
pitch_detector = AubioPitch("yin", BUF, HOP, SR)
pitch_detector.set_unit("Hz")
pitch_detector.set_silence(SILENCE_DB)
try:
    pitch_detector.set_tolerance(TOLERANCE)
except AttributeError:
    # some aubio versions expose set_threshold instead of set_tolerance
    pitch_detector.set_threshold(TOLERANCE)

pitch_history = deque(maxlen=10)
time_history = deque(maxlen=10)
current_pitch = 0.0
current_pitch_rate = 0.0
current_volume = 0.0
last_audio_time = time.time()
whistle_off_counter = 0  # counts consecutive frames without a whistle

# Operating mode: start safely in sleep
operating_mode = 'sleep'


# ───────── AI Model Globals ─────────
ai_interpreter = None
ai_input_details = None
ai_output_details = None

# AI rolling buffer + book‑keeping
ai_audio_buffer = np.array([], dtype=np.float32)
ai_buffer_fill_idx = 0
last_ai_inference_time = time.time()
ai_whistle_probability = 0.0  # last AI probability (for HUD)

# AI gate state
ai_whistle_gate_open = False
ai_negative_count = 0


def load_ai_model():
    """Load TFLite model and infer input sizing from its tensor shape."""
    global ai_interpreter, ai_input_details, ai_output_details
    global AI_MODEL_INPUT_SAMPLES, AI_MODEL_INPUT_SAMPLE_RATE, ai_audio_buffer

    try:
        if tf is None:
            logging.warning("TensorFlow not available. Skipping AI model; DSP‑only mode.")
            return

        logging.info(f"Loading AI model from {AI_MODEL_PATH}...")
        ai_interpreter = tf.lite.Interpreter(model_path=AI_MODEL_PATH)
        ai_interpreter.allocate_tensors()
        ai_input_details = ai_interpreter.get_input_details()
        ai_output_details = ai_interpreter.get_output_details()

        if len(ai_input_details) == 0:
            logging.error("AI model has no input tensors. Exiting.")
            sys.exit(1)

        # Determine required input length from model shape, e.g., (1, 44032) or (1, 44032, 1)
        input_shape = ai_input_details[0]['shape']
        if len(input_shape) >= 2:
            AI_MODEL_INPUT_SAMPLES = input_shape[1]
        else:
            logging.error(f"Unexpected AI input shape: {input_shape}. Cannot infer sample length.")
            ai_interpreter = None
            return

        # Initialize ring buffer to the correct size
        ai_audio_buffer = np.zeros(AI_MODEL_INPUT_SAMPLES, dtype=np.float32)

        # Infer sample‑rate from the assumed window duration
        if AI_MODEL_INPUT_DURATION_SEC > 0:
            AI_MODEL_INPUT_SAMPLE_RATE = int(AI_MODEL_INPUT_SAMPLES / AI_MODEL_INPUT_DURATION_SEC)
        else:
            logging.error("AI_MODEL_INPUT_DURATION_SEC must be > 0.")
            ai_interpreter = None
            return

        logging.info("AI model loaded.")
        logging.info(f"AI input shape: {ai_input_details[0]['shape']}")
        logging.info(f"Inferred samples: {AI_MODEL_INPUT_SAMPLES}, fs≈{AI_MODEL_INPUT_SAMPLE_RATE} Hz")
        logging.info(f"AI output shape: {ai_output_details[0]['shape']}")

    except Exception as e:
        logging.error(f"Failed to load AI model from '{AI_MODEL_PATH}': {e}")
        logging.error("Continuing without AI (DSP‑only).")
        ai_interpreter = None


def audio_callback(indata, frames, time_info, status):
    """Audio ISR: updates volume/pitch, maintains AI buffer, and computes gate state."""
    global sos_filter_state, current_pitch, current_pitch_rate, current_volume, last_audio_time, whistle_off_counter
    global ai_audio_buffer, ai_buffer_fill_idx, last_ai_inference_time, ai_whistle_probability
    global ai_whistle_gate_open, ai_negative_count

    if status:
        logging.warning(f"Audio callback status: {status}")

    # Gain + band‑pass → aubio pitch
    signal = indata[:, 0] * MIC_GAIN
    signal_filtered, sos_filter_state = sosfilt(SOS_FILTER, signal, zi=sos_filter_state)
    signal_float32 = signal_filtered.astype(np.float32)
    current_volume = float(np.sqrt(np.mean(signal_float32**2)))
    now = time.time()
    last_audio_time = now

    pitch_candidate = float(pitch_detector(signal_float32)[0])
    confidence = pitch_detector.get_confidence()

    # Fast DSP whistle decision for responsiveness
    is_dsp_whistle_this_frame = (
        PITCH_MIN <= pitch_candidate <= PITCH_MAX and
        confidence >= CONF_GATE and
        current_volume > VOL_GATE
    )

    # Periodic AI inference: authenticates the pilot’s whistle and controls the gate
    if ai_interpreter and AI_MODEL_INPUT_SAMPLE_RATE is not None and AI_MODEL_INPUT_SAMPLES is not None:
        # Resample audio to model’s expected fs if needed (linear, adequate for gating)
        resampled_signal = signal_float32
        if SR != AI_MODEL_INPUT_SAMPLE_RATE:
            interp_func = interp1d(np.arange(len(signal_float32)), signal_float32, kind='linear',
                                   bounds_error=False, fill_value=0.0)
            num_resampled_samples = int(len(signal_float32) * (AI_MODEL_INPUT_SAMPLE_RATE / SR))
            resampled_x = np.linspace(0, len(signal_float32) - 1, num_resampled_samples)
            resampled_signal = interp_func(resampled_x).astype(np.float32)

        # Append to rolling buffer (drop oldest if necessary)
        samples_to_copy = len(resampled_signal)
        if ai_buffer_fill_idx + samples_to_copy > AI_MODEL_INPUT_SAMPLES:
            shift_amount = (ai_buffer_fill_idx + samples_to_copy) - AI_MODEL_INPUT_SAMPLES
            ai_audio_buffer = np.roll(ai_audio_buffer, -shift_amount)
            ai_buffer_fill_idx = max(0, ai_buffer_fill_idx - shift_amount)

        if ai_buffer_fill_idx + samples_to_copy <= AI_MODEL_INPUT_SAMPLES:
            ai_audio_buffer[ai_buffer_fill_idx: ai_buffer_fill_idx + samples_to_copy] = resampled_signal
            ai_buffer_fill_idx += samples_to_copy
        else:
            logging.warning("AI buffer overflow condition handled (roll + clamp).")
            ai_audio_buffer[ai_buffer_fill_idx:] = resampled_signal[:AI_MODEL_INPUT_SAMPLES - ai_buffer_fill_idx]
            ai_buffer_fill_idx = AI_MODEL_INPUT_SAMPLES

        # Run AI occasionally on a full buffer
        if (now - last_ai_inference_time) >= AI_INFERENCE_INTERVAL_SEC and ai_buffer_fill_idx >= AI_MODEL_INPUT_SAMPLES:
            last_ai_inference_time = now

            input_tensor = ai_audio_buffer.reshape(ai_input_details[0]['shape'])
            ai_interpreter.set_tensor(ai_input_details[0]['index'], input_tensor)
            ai_interpreter.invoke()
            output_data = ai_interpreter.get_tensor(ai_output_details[0]['index'])
            probabilities = np.squeeze(output_data)

            ai_whistle_probability = 0.0
            if AI_WHISTLE_CLASS_INDEX < len(probabilities):
                ai_whistle_probability = probabilities[AI_WHISTLE_CLASS_INDEX]
                if ai_whistle_probability >= AI_PROBABILITY_THRESHOLD:
                    ai_whistle_gate_open = True
                    ai_negative_count = 0
                    logging.info(f"AI: My‑Whistle detected (p={ai_whistle_probability:.2f}) → Gate OPEN")
                else:
                    ai_negative_count += 1
                    logging.info(f"AI: Not My‑Whistle (p={ai_whistle_probability:.2f}) → neg={ai_negative_count}")
                    if ai_negative_count >= AI_GATE_FORGIVENESS_COUNT:
                        ai_whistle_gate_open = False
                        logging.warning(f"AI: {AI_GATE_FORGIVENESS_COUNT} consecutive negatives → Gate CLOSED")
            else:
                logging.warning(f"AI_WHISTLE_CLASS_INDEX {AI_WHISTLE_CLASS_INDEX} out of bounds (output size={len(probabilities)}).")

            # Reset buffer for next window
            ai_audio_buffer.fill(0)
            ai_buffer_fill_idx = 0

    # Final decision that drives the drone: DSP whistle gated by AI (if present)
    is_current_frame_whistle_for_drone = is_dsp_whistle_this_frame and ai_whistle_gate_open
    if ai_interpreter is None:
        # Fallback: DSP‑only control if AI is unavailable
        is_current_frame_whistle_for_drone = is_dsp_whistle_this_frame

    logging.info(
        f"Vol={current_volume:.4f}, pitch={pitch_candidate:.1f}, conf={confidence:.2f}, "
        f"DSP_Whistle={is_dsp_whistle_this_frame}, AI_Prob={ai_whistle_probability:.2f}, "
        f"AI_Gate={ai_whistle_gate_open}, Final_Whistle={is_current_frame_whistle_for_drone}"
    )

    # Update histories / counters for the control loop
    if is_current_frame_whistle_for_drone:
        whistle_off_counter = 0
        if len(time_history) > 0 and len(pitch_history) > 0:
            dt = now - time_history[-1]
            if dt > 1e-6:
                rate = (pitch_candidate - pitch_history[-1]) / dt
                current_pitch_rate = rate if abs(rate) >= RATE_DZ else 0.0
            else:
                current_pitch_rate = 0.0
        else:
            current_pitch_rate = 0.0
        pitch_history.append(pitch_candidate)
        time_history.append(now)
        current_pitch = pitch_candidate
    else:
        whistle_off_counter += 1
        if whistle_off_counter >= WHISTLE_OFF_THRESHOLD:
            pitch_history.clear()
            time_history.clear()
            current_pitch = 0.0
            current_pitch_rate = 0.0


# ───────── PID CONTROLLER ─────────
class PID:
    """Simple PID with clamped integral and output limits."""
    def __init__(self, kp, ki, kd, output_limits, integral_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output, self.max_output = output_limits
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

    def __call__(self, setpoint, process_variable):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt < 1e-6:
            dt = 1e-6
        error = setpoint - process_variable
        p_term = self.kp * error
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.ki * self.integral
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        self.last_error = error
        self.last_time = current_time
        return max(self.min_output, min(self.max_output, output))


# ───────── DRONE CONTROL ───────
class Drone:
    """Facade around Tello + a light simulation for HUD/testing."""
    def __init__(self, simulate=True):
        self.simulate = simulate
        self.is_flying = False
        self.last_fwd_speed = 0.0      # m/s
        self.last_yaw_rate  = 0.0      # deg/s
        self.yaw_smooth     = YAW_SMOOTH
        self.last_whistle_time = time.time()
        self.sim_x, self.sim_y, self.sim_z, self.sim_yaw = 0.0, 0.0, ALT_BASE, 0.0
        self.pid_altitude = PID(PID_KP, PID_KI, PID_KD, (PID_OUT_MIN, PID_OUT_MAX), PID_I_MAX)
        self.tello = None

        # Attempt to connect if in real mode; fallback to sim if unstable
        if not self.simulate:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.tello = Tello()
                    logging.info(f"Connecting to Tello (attempt {attempt + 1}/{max_retries})…")
                    self.tello.connect()
                    logging.info("Tello connected.")
                    self.tello.streamoff()

                    time.sleep(1)  # small grace after connect
                    battery = self.tello.get_battery()
                    if battery is None:
                        logging.warning("Battery query failed; retrying…")
                        time.sleep(1)
                        battery = self.tello.get_battery()

                    if battery is not None and battery < BAT_MIN:
                        logging.error(f"Battery too low ({battery}%). Exiting.")
                        self.close()
                        sys.exit(1)
                    elif battery is None:
                        logging.error("Battery status unavailable; retry connect.")
                        raise RuntimeError("Battery status unavailable")

                    logging.info(f"Battery: {battery}%")
                    logging.info("Waiting briefly for IMU…")
                    time.sleep(2)

                    current_height_cm = self.tello.get_height()
                    if current_height_cm is None:
                        logging.error("Initial height unavailable; retry connect.")
                        raise RuntimeError("Initial height unavailable")
                    logging.info(f"Initial height: {current_height_cm} cm")
                    break
                except Exception as e:
                    logging.error(f"Init failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        logging.warning("Retrying Tello connection…")
                        time.sleep(5)
                    else:
                        logging.error("Max retries reached; falling back to simulation.")
                        self.tello = None
                        self.simulate = True

    def takeoff(self):
        if self.is_flying:
            return
        logging.info("Takeoff initiated.")
        self.is_flying = True
        self.last_whistle_time = time.time()
        if not self.simulate and self.tello:
            try:
                self.tello.takeoff()
            except Exception as e:
                self.is_flying = False
                logging.exception(f"Takeoff failed: {e}")
        elif self.simulate:
            self.sim_z = ALT_BASE

    def land(self):
        if not self.is_flying and not self.simulate:
            if self.tello:
                logging.info("Ensuring motors off; attempting land.")
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)
                    self.tello.land()
                except Exception as e:
                    logging.exception(f"Land command failed: {e}")
            return
        logging.info("Landing initiated.")
        self.is_flying = False
        if not self.simulate and self.tello:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.1)
                self.tello.land()
            except Exception as e:
                logging.exception(f"Landing failed: {e}")
        elif self.simulate:
            self.sim_z = 0.0

    def step(self, pitch, rate, volume, whistle_frames_off):
        """One control step: handles takeoff/land, altitude PID, and RC mapping."""
        now = time.time()
        is_whistle_active = whistle_frames_off < WHISTLE_OFF_THRESHOLD

        if is_whistle_active:
            self.last_whistle_time = now

        if is_whistle_active and not self.is_flying:
            self.takeoff()

        if self.is_flying and (now - self.last_whistle_time) > LAND_TOUT:
            logging.info(f"Silence > {LAND_TOUT}s → landing.")
            self.land()
            return
        if not self.is_flying:
            return

        # altitude (sim vs real)
        measured_altitude_m = 0.0
        if self.simulate:
            measured_altitude_m = self.sim_z
        elif self.tello:
            height_cm = self.tello.get_height()
            if height_cm is not None:
                measured_altitude_m = height_cm / 100.0
            else:
                logging.warning("Height unavailable; using last sim_z/ALT_BASE.")
                measured_altitude_m = self.sim_z if hasattr(self, 'sim_z') else ALT_BASE

        # Map whistle volume → target altitude within [ALT_BASE..ALT_MAX]
        volume_range = MAX_EXPECTED_WHISTLE_VOLUME - VOL_GATE
        if volume_range > 1e-3:
            normalized_volume = np.clip((volume - VOL_GATE) / volume_range, 0, 1)
        else:
            normalized_volume = 0 if volume < VOL_GATE else 1

        desired_altitude_m = ALT_BASE + normalized_volume * (ALT_MAX - ALT_BASE)
        desired_altitude_m = np.clip(desired_altitude_m, ALT_BASE, ALT_MAX)
        vz_command = int(self.pid_altitude(desired_altitude_m, measured_altitude_m))

        # Soft ceiling: force descent if above ALT_MAX (and reset integral)
        if measured_altitude_m > ALT_MAX:
            logging.warning(f"Altitude {measured_altitude_m:.2f}m > ALT_MAX {ALT_MAX:.2f}m → forcing descent.")
            vz_command = PID_OUT_MIN
            self.pid_altitude.integral = 0.0

        # Hard ceiling: emergency stop
        if measured_altitude_m >= ABS_CEIL:
            logging.critical(f"EMERGENCY: {measured_altitude_m:.2f}m ≥ ABS_CEIL {ABS_CEIL}m.")
            self.emergency_stop()
            return

        # Forward/Yaw from pitch & slope; decay when whistle inactive
        if not is_whistle_active:
            raw_fwd = 0.0
            raw_yaw = 0.0
        else:
            raw_fwd = np.clip(pitch * K_FWD, 0.0, MAX_FWD)
            raw_yaw = np.clip(rate * K_YAW, -MAX_YAW, MAX_YAW)

        if not is_whistle_active:
            alpha_fwd = DECEL_FWD
            alpha_yaw = DECEL_YAW
        else:
            alpha_fwd = 1.0
            alpha_yaw = self.yaw_smooth if raw_yaw == 0.0 else 1.0

        # Exponential smoothing
        self.last_fwd_speed += alpha_fwd * (raw_fwd - self.last_fwd_speed)
        self.last_yaw_rate  += alpha_yaw * (raw_yaw - self.last_yaw_rate)

        fwd_speed_mps = self.last_fwd_speed
        yaw_rate_dps  = self.last_yaw_rate

        # RC mapping (−100..100)
        rc_left_right = 0
        rc_fwd_bwd    = int(np.clip((fwd_speed_mps / MAX_FWD) * 100, -100, 100))
        rc_up_down    = int(np.clip(vz_command, -100, 100))
        rc_yaw        = int(np.clip(yaw_rate_dps, -100, 100))

        if self.simulate:
            dt_sim = 1.0 / CTRL_HZ
            self.sim_yaw = (self.sim_yaw + yaw_rate_dps * dt_sim) % 360
            sim_yaw_rad = math.radians(self.sim_yaw)
            actual_fwd_sim = (rc_fwd_bwd / 100.0) * MAX_FWD
            self.sim_x += actual_fwd_sim * dt_sim * math.cos(sim_yaw_rad)
            self.sim_y += actual_fwd_sim * dt_sim * math.sin(sim_yaw_rad)
            MAX_VZ_SPEED_MPS_SIM = 0.5
            actual_vz_sim = (rc_up_down / 100.0) * MAX_VZ_SPEED_MPS_SIM
            self.sim_z += actual_vz_sim * dt_sim
            self.sim_z = np.clip(self.sim_z, 0, ABS_CEIL)
            if self.sim_z < ALT_BASE and is_whistle_active:
                self.sim_z = max(self.sim_z, ALT_BASE)
        elif self.tello:
            if self.tello.get_battery() < BAT_MIN:
                logging.warning(f"Battery low ({self.tello.get_battery()}%). Landing.")
                self.land()
                return
            try:
                self.tello.send_rc_control(rc_left_right, rc_fwd_bwd, rc_up_down, rc_yaw)
            except Exception as e:
                logging.exception(f"RC control failed: {e}. Emergency stop.")
                self.emergency_stop()

    def emergency_stop(self):
        """Stop motion immediately; attempt land and emergency motor cut‑off."""
        logging.critical("EMERGENCY STOP ACTIVATED!")
        if not self.simulate and self.tello:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
            except Exception as e:
                logging.error(f"Failed to send zero RC during emergency: {e}")
        self.is_flying = False
        if not self.simulate and self.tello:
            try:
                logging.info("Attempting land() during emergency…")
                self.tello.land(); time.sleep(1)
            except Exception as e:
                logging.error(f"land() failed during emergency: {e}")
            try:
                logging.info("Attempting emergency() motor cut‑off…")
                self.tello.emergency()
            except Exception as e:
                logging.error(f"emergency() failed: {e}")
        elif self.simulate:
            logging.info("Sim emergency: setting altitude to 0.")
            self.sim_z = 0.0

    def close(self):
        """Graceful shutdown; land if needed; release resources."""
        logging.info("Closing down…")
        if self.is_flying and not self.simulate and self.tello:
            logging.info("Drone is flying → landing first.")
            self.land(); time.sleep(2)
        if not self.simulate and self.tello:
            try:
                self.tello.end()
            except Exception as e:
                logging.exception(f"Error during Tello end(): {e}")
        logging.info("Drone closed.")


# ───────── MAIN APPLICATION ────────
def main():
    """Entry point: parse args, start audio, init HUD/drone, run control loop."""
    global current_pitch, current_pitch_rate, current_volume, last_audio_time, operating_mode, whistle_off_counter
    global ai_interpreter, ai_whistle_probability, ai_whistle_gate_open

    # Args ─────────────────────────────────────────────────────────────────────
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["sim", "real", "sim+real"],
        default="sim",
        help="sim=HUD only, real=Tello only, sim+real=both"
    )
    args = ap.parse_args()
    mode = args.mode

    # Mode flags
    simulate = (mode == "sim")                    # simulation only when exactly 'sim'
    show_full_hud_content = (mode in ("sim", "sim+real"))  # HUD shown for sim & sim+real

    # Load AI model before audio starts
    load_ai_model()

    # Audio stream ─────────────────────────────────────────────────────────────
    try:
        audio_stream = sd.InputStream(
            channels=1,
            samplerate=SR,
            blocksize=HOP,
            callback=audio_callback
        )
        audio_stream.start()
        logging.info("Audio stream started.")
    except Exception as e:
        logging.error(f"Failed to start audio stream: {e}. Check microphone device.")
        if ai_interpreter:
            logging.error("AI model loaded but audio unavailable. Exiting.")
        return

    # Window for keyboard input (always present, even in 'real')
    cv2.namedWindow("Drone Control")
    if show_full_hud_content:
        cv2.resizeWindow("Drone Control", 800, 600)
        logging.info(f"Running in '{mode}' → Full HUD.")
    else:
        cv2.resizeWindow("Drone Control", 300, 100)
        logging.info(f"Running in '{mode}' → Minimal window for keys only.")

    # Drone controller ─────────────────────────────────────────────────────────
    drone_controller = Drone(simulate=simulate)
    last_loop_time = time.time()

    try:
        while True:
            # Pull latest audio features
            pitch = current_pitch
            rate  = current_pitch_rate
            vol   = current_volume
            whistle_frames_off_for_main_loop = whistle_off_counter

            # Sleep mode forces zero commands and lands if needed
            if operating_mode == "sleep":
                pitch = rate = vol = 0.0
                whistle_frames_off_for_main_loop = WHISTLE_OFF_THRESHOLD + 1
                if drone_controller.is_flying:
                    logging.info("Sleep mode while flying → landing.")
                    drone_controller.land()

            # Step control
            drone_controller.step(pitch, rate, vol, whistle_frames_off_for_main_loop)

            # HUD / minimal window
            if show_full_hud_content:
                hud_display = np.zeros((600, 800, 3), np.uint8)

                # Text layout
                label_x, value_x = 20, 160
                y, dy = 40, 28
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs, th = 0.6, 1

                # Audio stats (cyan)
                audio_color = (0, 255, 255)
                audio_stats = [
                    ("Mode",  operating_mode.upper()),
                    ("Pitch", f"{current_pitch:5.1f} Hz"),
                    ("Rate",  f"{current_pitch_rate:5.1f} Hz/s"),
                    ("Vol",   f"{current_volume:5.3f}"),
                    ("Whistle Off", f"{whistle_off_counter} frames"),
                ]
                for lbl, val in audio_stats:
                    cv2.putText(hud_display, f"{lbl}:", (label_x, y), font, fs, audio_color, th)
                    cv2.putText(hud_display, val,       (value_x, y), font, fs, audio_color, th)
                    y += dy

                # AI status (orange)
                ai_status_color = (0, 165, 255)
                cv2.putText(hud_display, "AI Model:", (label_x, y), font, fs, ai_status_color, th)
                if ai_interpreter and AI_MODEL_INPUT_SAMPLE_RATE is not None:
                    cv2.putText(hud_display, "LOADED", (value_x, y), font, fs, ai_status_color, th); y += dy
                    cv2.putText(hud_display, "AI Prob:", (label_x, y), font, fs, ai_status_color, th)
                    cv2.putText(hud_display, f"{ai_whistle_probability:.2f}", (value_x, y), font, fs, ai_status_color, th); y += dy
                    cv2.putText(hud_display, "AI Gate:", (label_x, y), font, fs, ai_status_color, th)
                    cv2.putText(hud_display, "OPEN" if ai_whistle_gate_open else "CLOSED", (value_x, y), font, fs, ai_status_color, th); y += dy
                else:
                    cv2.putText(hud_display, "NOT LOADED", (value_x, y), font, fs, ai_status_color, th); y += dy

                y += 10  # spacer

                # Flight stats (green)
                flight_color = (0, 255, 0)
                flight_stats = [
                    ("Flying",  "YES" if drone_controller.is_flying else "NO"),
                    ("Altitude", f"{drone_controller.sim_z:.2f} m" if drone_controller.simulate else f"{drone_controller.tello.get_height()/100.0:.2f} m"),
                    ("Fwd Speed", f"{drone_controller.last_fwd_speed:.2f} m/s"),
                    ("Yaw Rate", f"{drone_controller.last_yaw_rate:.1f} °/s"),
                    ("Sim Pos", f"X:{drone_controller.sim_x:.2f} Y:{drone_controller.sim_y:.2f}" if drone_controller.simulate else "N/A"),
                ]
                for lbl, val in flight_stats:
                    cv2.putText(hud_display, f"{lbl}:", (label_x, y), font, fs, flight_color, th)
                    cv2.putText(hud_display, val,       (value_x, y), font, fs, flight_color, th)
                    y += dy

                # Battery (red if low)
                battery_color = (0, 0, 255) if (not drone_controller.simulate and drone_controller.tello and drone_controller.tello.get_battery() < BAT_MIN + 5) else (255, 255, 0)
                if not drone_controller.simulate and drone_controller.tello:
                    cv2.putText(hud_display, "Battery:", (label_x, y), font, fs, battery_color, th)
                    cv2.putText(hud_display, f"{drone_controller.tello.get_battery()}%", (value_x, y), font, fs, battery_color, th)
                    y += dy

                # Help
                help_color = (255, 255, 255)
                y = 500
                cv2.putText(hud_display, "S: Sleep | A: Active | Q: Quit | E: Emergency", (label_x, y), font, fs, help_color, th)

            else:
                # Minimal window in real mode
                hud_display = np.zeros((100, 300, 3), np.uint8)
                help_color = (255, 255, 255)
                cv2.putText(hud_display, "Drone Control - Real Mode", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, help_color, 1)
                cv2.putText(hud_display, "S: Sleep | A: Active",       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, help_color, 1)
                cv2.putText(hud_display, "Q: Quit | E: Emergency",     (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, help_color, 1)

            cv2.imshow("Drone Control", hud_display)

            # Keyboard (always active)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("Quit requested.")
                break
            elif key == ord('s'):
                if operating_mode != "sleep":
                    operating_mode = "sleep"
                    logging.info("Sleep mode requested.")
            elif key == ord('a'):
                if operating_mode != "active":
                    operating_mode = "active"
                    logging.info("Active mode requested.")
            elif key == ord('e'):
                logging.warning("Emergency stop requested.")
                drone_controller.emergency_stop()
                break

            # Maintain loop rate
            now = time.time()
            elapsed = now - last_loop_time
            to_sleep = (1.0 / CTRL_HZ) - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            last_loop_time = time.time()

            # Auto‑land on audio timeout (active mode only)
            if (drone_controller.is_flying
                and operating_mode == "active"
                and (time.time() - last_audio_time) > CALLBACK_TOUT):
                logging.error(f"Audio timeout ({CALLBACK_TOUT}s). Landing.")
                drone_controller.land()

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt — exiting.")
    except Exception as e:
        logging.exception(f"Unhandled exception: {e}")
        if hasattr(drone_controller, "emergency_stop"):
            drone_controller.emergency_stop()
    finally:
        logging.info("Cleaning up…")
        if 'audio_stream' in locals() and audio_stream.active:
            audio_stream.stop()
            audio_stream.close()
            logging.info("Audio stream stopped.")
        if 'drone_controller' in locals():
            drone_controller.close()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()
