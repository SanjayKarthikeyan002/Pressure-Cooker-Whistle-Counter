import numpy as np
import sounddevice as sd
import time
import numpy.fft as fft

# ---------- CONFIG ----------
SAMPLE_RATE = 16000          # mic sample rate (Hz)
WINDOW_DURATION = 0.05       # 50 ms
MIN_SILENCE_BETWEEN = 0.5    # seconds of quiet to separate whistles

THRESHOLD_MULTIPLIER = 3.0   # loudness threshold multiplier (for background)
SIMILARITY_THRESHOLD = 0.8   # cosine similarity (0–1), higher = stricter

CALIBRATION_SECONDS = 3.0    # for background noise
PROFILE_FILE = "whistle_profile.npy"
# ----------------------------


def compute_energy_block(block):
    return float(np.mean(np.abs(block)))


def compute_block_spectrum(block):
    # apply window
    window = np.hanning(len(block))
    block_win = block * window
    spec = np.abs(fft.rfft(block_win))
    # normalize
    norm = np.linalg.norm(spec) + 1e-9
    return spec / norm


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def calibrate_threshold():
    print(f"Calibrating for {CALIBRATION_SECONDS} seconds...")
    print("Stay quiet, no cooker whistle yet.\n")

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

    print(f"Average background energy: {avg_energy:.6f}")
    print(f"Energy threshold:          {threshold:.6f}\n")

    return threshold


def run_realtime_whistle_counter():
    # Load trained whistle profile
    whistle_profile = np.load(PROFILE_FILE)
    print(f"Loaded whistle profile from {PROFILE_FILE}")
    print(f"Profile length: {len(whistle_profile)}\n")

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

        block = indata[:, 0]

        # 1) Loudness check
        energy = compute_energy_block(block)
        is_loud = energy > energy_threshold

        # 2) Whistle similarity check
        block_spec = compute_block_spectrum(block)

        # Ensure same length (in case of mismatch)
        L = min(len(block_spec), len(whistle_profile))
        sim = cosine_similarity(block_spec[:L], whistle_profile[:L])

        is_whistle_like = sim >= SIMILARITY_THRESHOLD

        # Debug if you want:
        # print(f"Energy={energy:.6f}, sim={sim:.2f}")

        if is_loud and is_whistle_like:
            if not state["in_whistle"]:
                state["whistles"] += 1
                state["in_whistle"] = True
                t = time.time() - state["start_time"]
                print(f"Whistle #{state['whistles']} detected at {t:.2f} s (sim={sim:.2f})")
            state["silence_frames"] = 0
        else:
            if state["in_whistle"]:
                state["silence_frames"] += 1
                if state["silence_frames"] >= min_silence_frames:
                    state["in_whistle"] = False

    print("Starting REAL-TIME TRAINED whistle counter...")
    print("It will react mainly to sounds that match your cooker whistle profile.")
    print("Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=blocksize,
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            print(f"Total whistles detected: {state['whistles']}")


if __name__ == "__main__":
    run_realtime_whistle_counter()
