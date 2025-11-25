import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
FILE_PATH = "PressureCookerWhistleData.mp3"  # Your audio file
WINDOW_DURATION = 0.05      # 50 ms window for energy calculation
MIN_SILENCE_BETWEEN = 0.5   # Seconds of silence to separate whistles
THRESHOLD_MULTIPLIER = 2.0  # Sensitivity (lower = more sensitive)
# ----------------------------


def load_audio(file_path):
    """Load audio file and convert to mono if needed."""
    audio, sr = sf.read(file_path)

    # Convert stereo to mono
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)

    return audio, sr


def compute_energy(audio, sr, window_duration):
    """Compute short-time energy for the audio signal."""
    window_size = int(window_duration * sr)
    if window_size <= 0:
        window_size = 1

    num_frames = len(audio) // window_size
    trimmed = audio[:num_frames * window_size]
    frames = trimmed.reshape(num_frames, window_size)

    frame_energy = np.mean(np.abs(frames), axis=1)
    frame_times = (np.arange(num_frames) + 0.5) * window_duration

    return frame_energy, frame_times


def count_whistles(frame_energy, window_duration, min_silence_between, multiplier):
    """Count number of whistles based on energy and threshold."""
    avg_energy = np.mean(frame_energy)
    threshold = avg_energy * multiplier

    print(f"Average energy = {avg_energy:.6f}")
    print(f"Threshold      = {threshold:.6f}")

    whistles = 0
    in_whistle = False
    silence_frames = 0
    min_silence_frames = int(min_silence_between / window_duration)

    for energy in frame_energy:
        if energy > threshold:
            if not in_whistle:
                whistles += 1
                in_whistle = True
            silence_frames = 0
        else:
            if in_whistle:
                silence_frames += 1
                if silence_frames >= min_silence_frames:
                    in_whistle = False

    return whistles, threshold


def get_whistle_timestamps(frame_energy, frame_times, threshold, window_duration, min_silence_between):
    """Return start timestamps of each whistle."""
    timestamps = []
    in_whistle = False
    silence_frames = 0
    min_silence_frames = int(min_silence_between / window_duration)

    for i, energy in enumerate(frame_energy):
        if energy > threshold:
            if not in_whistle:
                timestamps.append(frame_times[i])
                in_whistle = True
            silence_frames = 0
        else:
            if in_whistle:
                silence_frames += 1
                if silence_frames >= min_silence_frames:
                    in_whistle = False

    return timestamps


def main():
    # 1) Load audio
    audio, sr = load_audio(FILE_PATH)
    duration = len(audio) / sr
    print(f"Loaded file: {FILE_PATH}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration:    {duration:.2f} seconds")

    # 2) Compute energy
    frame_energy, frame_times = compute_energy(audio, sr, WINDOW_DURATION)

    # 3) Count whistles
    whistles, threshold = count_whistles(
        frame_energy,
        WINDOW_DURATION,
        MIN_SILENCE_BETWEEN,
        THRESHOLD_MULTIPLIER
    )

    print("\n----------------------")
    print(f"Whistles detected: {whistles}")
    print("----------------------")

    # 3b) Get whistle timestamps
    timestamps = get_whistle_timestamps(
        frame_energy,
        frame_times,
        threshold,
        WINDOW_DURATION,
        MIN_SILENCE_BETWEEN
    )

    print("\nWhistle start times (seconds):")
    for i, t in enumerate(timestamps, start=1):
        print(f"Whistle {i}: {t:.2f} s")

    # 4) Plot energy, threshold, and whistle markers
    plt.figure(figsize=(12, 4))
    plt.plot(frame_times, frame_energy, label="Energy")
    plt.axhline(threshold, color='red', linestyle='--', label="Threshold")

    # Mark whistles
    for t in timestamps:
        plt.axvline(t, color='green', linestyle=':', alpha=0.7)

    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Whistle Detection Using Short-Time Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
# End of whistle_counter.py