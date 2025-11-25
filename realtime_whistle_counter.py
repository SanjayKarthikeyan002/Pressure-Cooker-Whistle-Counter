import numpy as np
import sounddevice as sd
import time

# ---------- CONFIG ----------
SAMPLE_RATE = 16000          # mic sample rate (Hz)
WINDOW_DURATION = 0.05       # 50 ms per analysis window
MIN_SILENCE_BETWEEN = 0.5    # seconds of quiet to separate whistles
THRESHOLD_MULTIPLIER = 3.0   # sensitivity: lower = more sensitive
CALIBRATION_SECONDS = 3.0    # listen to background noise for this long
# ----------------------------


def compute_energy_block(block):
    """
    Compute energy of a single audio block.
    Energy = mean of absolute values.
    """
    return float(np.mean(np.abs(block)))


def calibrate_threshold():
    """
    Record a few seconds of background audio (no whistles),
    and compute an energy-based threshold.
    """
    print(f"Calibrating for {CALIBRATION_SECONDS} seconds...")
    print("Stay quiet, no cooker whistle yet.")

    num_samples = int(CALIBRATION_SECONDS * SAMPLE_RATE)
    recording = sd.rec(
        num_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    recording = recording.flatten()
    window_size = int(WINDOW_DURATION * SAMPLE_RATE)
    num_frames = len(recording) // window_size
    recording = recording[:num_frames * window_size]
    frames = recording.reshape(num_frames, window_size)

    frame_energy = np.mean(np.abs(frames), axis=1)
    avg_energy = float(np.mean(frame_energy))
    threshold = avg_energy * THRESHOLD_MULTIPLIER

    print(f"\nCalibration complete.")
    print(f"Average background energy: {avg_energy:.6f}")
    print(f"Using threshold:          {threshold:.6f}\n")

    return threshold


def run_realtime_whistle_counter():
    """
    Run real-time whistle detection using the microphone.
    """
    threshold = calibrate_threshold()

    state = {
        "whistles": 0,
        "in_whistle": False,
        "silence_frames": 0,
        "start_time": time.time()
    }

    min_silence_frames = int(MIN_SILENCE_BETWEEN / WINDOW_DURATION)
    blocksize = int(WINDOW_DURATION * SAMPLE_RATE)

    def audio_callback(indata, frames, time_info, status):
        """
        This function is called automatically by sounddevice
        every time a new block of audio is available.
        """
        if status:
            print(status)

        # Take mono channel
        block = indata[:, 0]
        energy = compute_energy_block(block)

        # Update state based on energy vs threshold
        if energy > threshold:
            # loud region (potential whistle)
            if not state["in_whistle"]:
                state["whistles"] += 1
                state["in_whistle"] = True
                # current elapsed time
                t = time.time() - state["start_time"]
                print(f"Whistle #{state['whistles']} detected at {t:.2f} s")
            state["silence_frames"] = 0
        else:
            # quiet region
            if state["in_whistle"]:
                state["silence_frames"] += 1
                if state["silence_frames"] >= min_silence_frames:
                    state["in_whistle"] = False

    print("Starting real-time whistle counter...")
    print("Put the laptop near the cooker.")
    print("Press Ctrl+C in this window to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=blocksize,
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(0.1)  # keep main thread alive
        except KeyboardInterrupt:
            print("\nStopped by user.")
            print(f"Total whistles detected: {state['whistles']}")


if __name__ == "__main__":
    run_realtime_whistle_counter()
