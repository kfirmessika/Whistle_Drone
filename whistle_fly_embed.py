#!/usr/bin/env python3
"""
whistle_fly_embed.py – v1.2 (15 Jul 2025)

* Instant-hover the moment you stop whistling.
* Restored robust in-flight battery and height telemetry safety checks.
* Adds manual 'l' (land) key.
* Ensures command redundancy checks are robust.
"""
import argparse, json, os, pathlib, time, math, sys, queue
import numpy as np, sounddevice as sd, librosa, cv2
from scipy.spatial.distance import cdist

try:
    from djitellopy import Tello
except ImportError:
    Tello = None
    print("⚠  djitellopy not found – running SIM only")

# ---------- Parameters you might tweak ----------
SR             = 22_050       # Hz - Sample Rate. Must match calibration.
WIN            = 1.0          # seconds analysed each time. Length of audio window for classification.
HOVER_M        = 0.60         # fixed hover height in meters.
MAX_FWD_MPS    = 0.4          # Max forward/backward speed in meters/second.
MAX_YAW_DPS    = 90           # Max yaw speed in degrees/second.
BATT_MIN       = 25           # % - Minimum battery percentage to allow flight.
CEIL_HARD_M    = 1.50         # meters - Hard ceiling: emergency land if drone reaches this.
NO_CMD_TOUT    = 10           # seconds - If no valid command for this duration, drone will hover.
VOL_GATE_RMS   = 0.005        # RMS - Minimum volume to consider an audio window for classification.
N_MFCC         = 20           # Number of MFCCs to extract. Must match calibration.

RC_SEND_INTERVAL = 0.1        # seconds - Minimum time between sending RC commands to Tello (approx 10Hz)

# Safety timeouts for telemetry
HEIGHT_TELEMETRY_TIMEOUT = 2.0 # s, Max time without valid height data before emergency landing
MAX_HEIGHT_STALENESS     = 5    # Count, Max consecutive invalid height readings before emergency
IMU_WAIT                 = 5.0  # s, Time to wait after Tello connect for IMU/sensors to stabilize
# -------------------------------------------------

# Mapping of classified commands to Tello RC control values (lr, fb, ud, yaw)
# These are RC values from -100 to 100.
CMD2RC = {
    "forward": dict(lr=0,  fb=  60, ud=0, yaw=0),
    "back"   : dict(lr=0,  fb= -60, ud=0, yaw=0),
    "left"   : dict(lr=0,  fb=   0, ud=0, yaw=-60), # Counter-clockwise yaw
    "right"  : dict(lr=0,  fb=   0, ud=0, yaw= 60), # Clockwise yaw
    # "fly" and "land" are handled as discrete actions in the Drone class
}

# ------------------ simple k-NN ----------------------------------------------
class SimpleKNN:
    """
    A simple K-Nearest Neighbors classifier for classifying whistle embeddings.
    Uses Euclidean distance to find the closest pre-recorded whistle.
    """
    def __init__(self, X, y):
        self.X = X # Embeddings (N x D matrix)
        self.y = np.array(y) # Labels (N-length array)

    def predict(self, v):
        """
        Predicts the command for a given input embedding vector 'v'.
        Finds the closest embedding in the training data.
        """
        # Calculate Euclidean distance from input vector 'v' to all stored embeddings 'X'
        d = cdist(self.X, v.reshape(1,-1), metric="euclidean").ravel()
        # Find the index of the minimum distance
        idx = np.argmin(d)
        # Return the label corresponding to the closest embedding
        return self.y[idx]

def load_user_embeddings(user):
    """
    Loads the pre-saved embeddings and labels for a specific user.
    """
    root = pathlib.Path("users")/user/"embeddings"
    embeddings_path = root/"embeddings.npy"
    labels_path = root/"labels.json"

    if not embeddings_path.exists() or not labels_path.exists():
        print(f"❌ Error: Embeddings not found for user '{user}'. Please run calibrate_whistle_embed.py first.")
        sys.exit(1)

    X = np.load(embeddings_path)
    y = json.loads(labels_path.read_text())
    return SimpleKNN(X, y)

# ------------------ audio thread ---------------------------------------------
audio_q : "queue.Queue[np.ndarray]" = queue.Queue()

def audio_cb(indata, frames, time_info, status):
    """
    Callback function for the sounddevice audio stream.
    Puts incoming audio chunks into a thread-safe queue.
    """
    if status:
        print(f"Audio stream status: {status}", file=sys.stderr)
    audio_q.put(indata.copy())

def next_window():
    """
    Collects WIN seconds of audio from the queue and returns it as a mono numpy array.
    This function blocks until enough audio data is available.
    """
    needed = int(WIN * SR)
    buf, got = [], 0
    while got < needed:
        chunk = audio_q.get()
        buf.append(chunk); got += len(chunk)
    return np.concatenate(buf)[:needed,0]

def classify_window(knn: SimpleKNN):
    """
    Captures an audio window, checks its volume, extracts MFCCs, and classifies it using k-NN.
    Returns the predicted command string or None if below volume gate.
    """
    raw = next_window()
    if np.sqrt(np.mean(raw**2)) < VOL_GATE_RMS:
        return None
    mfcc = librosa.feature.mfcc(y=raw.astype(float), sr=SR, n_mfcc=N_MFCC)
    return knn.predict(mfcc.mean(axis=1))

# ------------------ drone wrapper -------------------------------------------
class Drone:
    """
    Manages the drone's state (real or simulated) and sends commands.
    Includes safety checks and command redundancy handling.
    """
    def __init__(self, real=True):
        self.real = real and (Tello is not None)
        self.tello = None
        self.last_rc_send = 0.0 # To rate-limit RC commands
        
        # Telemetry tracking for safety
        self.last_battery_read_time = 0.0
        self.cached_battery = 100
        self.last_height_check_time = 0.0
        self.cached_height_cm = int(HOVER_M * 100) # Initial cached height
        self.last_valid_height_time = time.time()
        self.height_staleness_count = 0

        if self.real:
            try:
                self.tello = Tello()
                print("Connecting to Tello drone...")
                self.tello.connect()
                self.tello.streamoff() # Ensure video stream is off to save bandwidth/resources

                # Initial battery check
                batt = self.tello.get_battery()
                print(f"Tello Battery: {batt}%")
                if batt < BATT_MIN:
                    raise RuntimeError(f"Battery too low ({batt}%) to fly. Requires {BATT_MIN}%.")
                
                print(f"Waiting {IMU_WAIT}s for Tello IMU to stabilize...")
                time.sleep(IMU_WAIT) # Give time for IMU to settle

                # Initial height read
                initial_height = self.tello.get_height()
                if initial_height is None:
                    raise RuntimeError("Failed to get initial Tello height (IMU error?).")
                self.cached_height_cm = initial_height
                self.last_height_check_time = time.time() # Update time for height
                self.last_valid_height_time = time.time() # Mark valid telemetry
                print("Tello connected and ready.")

            except Exception as e:
                print(f"❌ Tello connection failed: {e} - switching to SIMULATION ONLY.")
                if self.tello: # Try to end connection if it was partially established
                    try: self.tello.end()
                    except: pass
                self.real = False # Fallback to simulation if connection fails
        
        self.flying = False # Internal state: True if drone is currently airborne
        self.last_cmd_t = time.time() # Timestamp of the last valid whistle command

    # ------------- basic actions -----------------
    def takeoff(self):
        """
        Initiates drone takeoff. Ignores command if already flying.
        """
        if self.flying:
            print("✈  Already flying, ignoring takeoff command.")
            return
        print("✈  Takeoff initiated!")
        self.flying = True
        if self.real:
            self.tello.takeoff()
            time.sleep(2) # Give Tello time to execute takeoff

    def land(self):
        """
        Initiates drone landing. Ignores command if already landed.
        """
        if not self.flying:
            print("⬇  Already landed, ignoring land command.")
            return
        print("⬇  Landing initiated!")
        if self.real:
            self.tello.land()
        self.flying = False

    def emergency_stop(self):
        """
        Performs immediate emergency stop (hard motor cut-off).
        """
        print("🚨 EMERGENCY STOP!")
        if self.real and self.tello:
            try: self.tello.emergency() # Force motors off
            except Exception as e: print("  error during emergency:", e)
        self.flying = False # Drone is no longer flying

    # ------------- low-level RC -------------------
    def rc(self, lr=0, fb=0, ud=0, yaw=0):
        """
        Sends RC control commands to the Tello drone (or simulates them).
        Rate-limits commands to prevent spamming the Tello.
        """
        now = time.time()
        if (now - self.last_rc_send) < RC_SEND_INTERVAL:
            return # Skip sending if too soon since last command

        if self.real:
            try:
                self.tello.send_rc_control(lr, fb, ud, yaw)
            except Exception as e:
                print(f"RC error: {e}")
                # Consider triggering emergency_stop if RC commands consistently fail
                # self.emergency_stop() 
        self.last_rc_send = now # Update time only if command was actually sent

    # ------------- Telemetry & Safety Checks -------------------
    def get_cached_battery(self):
        """Fetches Tello battery, caches for 1 second to reduce API calls."""
        now = time.time()
        if self.real and (now - self.last_battery_read_time) > 1.0:
            try:
                new_batt = self.tello.get_battery()
                if new_batt is not None:
                    self.cached_battery = new_batt
                    self.last_battery_read_time = now
            except Exception as e:
                print(f"DEBUG: Failed to get Tello battery (cached): {e}") # Use print for quick debug
        return self.cached_battery

    def get_current_height_m(self):
        """
        Gets current height from Tello (if real) or simulation.
        Includes staleness check for real drone telemetry.
        """
        if not self.real:
            return HOVER_M if self.flying else 0.0 # Sim doesn't have dynamic height from Tello

        now = time.time()
        # Update cached height if enough time has passed
        if (now - self.last_height_check_time) > 0.1: # Check height roughly 10 times a second
            try:
                height_cm = self.tello.get_height()
                if height_cm is not None:
                    self.cached_height_cm = height_cm
                    self.last_height_check_time = now
                    self.last_valid_height_time = now # Update time of last valid telemetry
                    self.height_staleness_count = 0 # Reset staleness counter
                else:
                    self.height_staleness_count += 1 # Increment if height is None
                    print(f"⚠ Tello height returned None. Staleness count: {self.height_staleness_count}")
            except Exception as e:
                self.height_staleness_count += 1
                print(f"⚠ Error getting Tello height: {e}. Staleness count: {self.height_staleness_count}")
            self.last_height_check_time = now

        # Check for overall telemetry staleness timeout
        if self.height_staleness_count >= MAX_HEIGHT_STALENESS or \
           (now - self.last_valid_height_time) > HEIGHT_TELEMETRY_TIMEOUT:
            print("❌ Tello height telemetry is stale/missing. Initiating emergency land!")
            self.emergency_stop() # Trigger emergency stop if telemetry is unreliable
            sys.exit(1) # Exit after emergency

        return self.cached_height_cm / 100.0 # Return height in meters

    # ------------- main FSM (Finite State Machine) -----------------------
    def update(self, cmd):
        now = time.time()

        # --- In-flight safety checks (only if real drone is flying) ---
        if self.real and self.flying:
            current_battery = self.get_cached_battery()
            if current_battery < BATT_MIN:
                print(f"⚠ Low battery ({current_battery}%) mid-flight! Initiating auto-landing.")
                self.land() # Auto-land on low battery
                return # Stop processing this update cycle

            current_height = self.get_current_height_m() # This also handles staleness
            if current_height > CEIL_HARD_M:
                print(f"⚠ Hard ceiling hit ({current_height:.2f}m) – initiating landing.")
                self.land() # Force land if ceiling is hit
                return # Stop processing this update cycle

        # -------- immediate hover on silence -------
        if cmd is None:
            if self.flying:
                self.rc(0,0,0,0) # Send zero RC commands instantly to stop horizontal/yaw movement
            
            # Start / continue the inactivity timer
            if now - self.last_cmd_t > NO_CMD_TOUT and self.flying:
                # After NO_CMD_TOUT (10s) of silence, it's already hovering.
                # If silence continues for another 5 seconds, initiate auto-land.
                if now - self.last_cmd_t > NO_CMD_TOUT + 5:
                    print("Auto-land after extended silence"); self.land()
            return

        # We have a valid command – reset timer
        self.last_cmd_t = now

        # Handle discrete commands
        if cmd == "fly":
            self.takeoff(); return
        if cmd == "land":
            self.land(); return

        # Handle motion commands (only if flying)
        if self.flying and cmd in CMD2RC:
            self.rc(**CMD2RC[cmd])
        elif not self.flying:
            # If a motion command is received while landed, ensure drone is idle.
            self.rc(0,0,0,0)

    def close(self):
        """
        Cleans up drone resources, ensuring landing if flying and Tello disconnect.
        """
        print("Closing drone connection...")
        if self.flying:
            self.land() # Ensure drone lands before disconnecting
            time.sleep(2) # Give it time to land
        if self.real and self.tello:
            try:
                self.tello.end() # Disconnect from Tello
                print("Tello drone disconnected.")
            except Exception as e:
                print(f"Error ending Tello connection: {e}")

# ------------------ main loop -----------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Whistle-controlled Tello drone flight (or simulation).")
    ap.add_argument("--user", required=True, help="Username whose calibration embeddings to load.")
    ap.add_argument("--sim", action="store_true", help="Run in simulation-only mode (no real Tello).")
    args = ap.parse_args()

    knn = load_user_embeddings(args.user)
    drone = Drone(real=not args.sim)

    # Start the audio input stream
    try:
        stream = sd.InputStream(channels=1, samplerate=SR, blocksize=1024,
                                callback=audio_cb, dtype="float32"); stream.start()
        print("🟢 Listening for whistle commands… (q=quit, e=EMERGENCY, l=land)")
    except Exception as e:
        print(f"❌ Audio stream initialization failed: {e}")
        print("Please ensure your microphone is connected, recognized by the OS, and sounddevice is configured correctly.")
        sys.exit(1)

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),27): break # 'q' or ESC to quit
            if key==ord('e'): drone.emergency_stop(); break # 'e' for emergency stop
            if key==ord('l'): drone.land() # 'l' for manual land

            cmd = classify_window(knn)
            if cmd: print(f"🎧 {cmd}")
            drone.update(cmd)
            time.sleep(0.05) # Control loop rate

    except KeyboardInterrupt:
        print("Ctrl-C – exiting")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}")
    finally:
        stream.stop(); stream.close()
        drone.close()
        cv2.destroyAllWindows()
        print("Application finished.")

if __name__ == "__main__":
    main()
