import numpy as np
import sounddevice as sd
import time

# ---------- CONFIG (you can tweak these) ----------
SAMPLE_RATE = 16000          # mic sample rate (Hz)
WINDOW_DURATION = 0.05       # 50 ms analysis window
MIN_SILENCE_BETWEEN = 0.5    # seconds of quiet to separate whistles

THRESHOLD_MULTIPLIER = 3   # loudness sensitivity (lower = more sensitive)

# Generic pressure-cooker whistle band (fairly lenient)
WHISTLE_LOW_HZ = 1500        # lower bound of band
WHISTLE_HIGH_HZ = 4500       # upper bound of band

# How "whistle-like" it must be:
# ratio = energy in whistle band / total energy
RATIO_THRESHOLD = 0.9      # lower = more lenient, higher = stricter

CALIBRATION_SECONDS = 3.0    # background calibration duration
# ---------------------------------------------------


def compute_energy_block(block):
    """Simple loudness measure: mean abs value."""
    return float(np.mean(np.abs(block)))


def compute_whistle_ratio(block):
    """
    Compute how much of the sound's energy is in the whistle frequency band.
    Returns a ratio between 0 and 1.
    """
    # Hann window to reduce spectral leakage
    window = np.hanning(len(block))
    block_win = block * window

    # FFT (real)
    fft = np.fft.rfft(block_win)
    mag2 = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(block_win), d=1.0 / SAMPLE_RATE)

    total_energy = float(np.sum(mag2)) + 1e-9  # avoid divide by zero

    # Energy in whistle band
    band_mask = (freqs >= WHISTLE_LOW_HZ) & (freqs <= WHISTLE_HIGH_HZ)
    band_energy = float(np.sum(mag2[band_mask]))

    ratio = band_energy / total_energy
    return ratio


def calibrate_threshold():
    """
    Record background audio (no whistles) and compute the loudness threshold.
    """
    print(f"Calibrating for {CALIBRATION_SECONDS} seconds...")
    print("Stay quiet, no cooker whistle yet.\n")

    num_samples = int(CALIBRATION_SECONDS * SAMPLE_RATE)
    recording = sd.rec(
        num_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
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

    print(f"Average background energy: {avg_energy:.6f}")
    print(f"Energy threshold:          {threshold:.6f}\n")

    return threshold


def run_realtime_whistle_counter():
    """
    Real-time whistle detection based on loudness + whistle-band ratio.
    Works for many different cooker whistles (lenient).
    """
    energy_threshold = calibrate_threshold()

    state = {
        "whistles": 0,
        "in_whistle": False,
        "silence_frames": 0,
        "start_time": time.time(),
    }

    min_silence_frames = int(MIN_SILENCE_BETWEEN / WINDOW_DURATION)
    blocksize = int(WINDOW_DURATION * SAMPLE_RATE)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)

        block = indata[:, 0]  # mono

        energy = compute_energy_block(block)
        ratio = compute_whistle_ratio(block)

        # Conditions for whistle:
        is_loud = energy > energy_threshold
        is_whistle_like = ratio > RATIO_THRESHOLD

        # Debug line if you want to see values:
        # print(f"Energy={energy:.6f}, ratio={ratio:.2f}")

        if is_loud and is_whistle_like:
            if not state["in_whistle"]:
                state["whistles"] += 1
                state["in_whistle"] = True
                t = time.time() - state["start_time"]
                print(
                    f"Whistle #{state['whistles']} detected at {t:.2f} s "
                    f"(ratio={ratio:.2f})"
                )
            state["silence_frames"] = 0
        else:
            if state["in_whistle"]:
                state["silence_frames"] += 1
                if state["silence_frames"] >= min_silence_frames:
                    state["in_whistle"] = False

    print("Starting REAL-TIME GENERIC whistle counter...")
    print("It should react to most pressure cooker whistles (more lenient).")
    print("It may occasionally react to other high-pitched tones.")
    print("Press Ctrl+C in this window to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=audio_callback,
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            print(f"Total whistles detected: {state['whistles']}")


if __name__ == "__main__":
    run_realtime_whistle_counter()
