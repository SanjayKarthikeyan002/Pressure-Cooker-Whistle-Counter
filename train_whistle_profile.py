import numpy as np
import soundfile as sf
import numpy.fft as fft

# ---------- CONFIG ----------
FILE_PATH = "PressureCookerWhistleData.mp3"
WINDOW_DURATION = 0.05       # 50 ms
THRESHOLD_MULTIPLIER = 2.5   # used to pick whistle segments
# ----------------------------


def load_audio(file_path):
    audio, sr = sf.read(file_path)
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def compute_energy(audio, sr, window_duration):
    window_size = int(window_duration * sr)
    if window_size <= 0:
        window_size = 1

    num_frames = len(audio) // window_size
    trimmed = audio[:num_frames * window_size]
    frames = trimmed.reshape(num_frames, window_size)

    frame_energy = np.mean(np.abs(frames), axis=1)
    return frame_energy, frames


def build_whistle_profile(frames, frame_energy, sr, window_duration, threshold_multiplier):
    """
    Build an average spectrum ("fingerprint") of whistle frames.
    """
    avg_energy = np.mean(frame_energy)
    threshold = avg_energy * threshold_multiplier

    print(f"Average energy = {avg_energy:.6f}")
    print(f"Energy threshold for whistle frames = {threshold:.6f}")

    window_size = frames.shape[1]
    hann = np.hanning(window_size)

    sum_spectrum = None
    count = 0

    for e, frame in zip(frame_energy, frames):
        if e > threshold:
            # treat this frame as part of a whistle
            win_frame = frame * hann
            spec = np.abs(fft.rfft(win_frame))  # magnitude spectrum

            if sum_spectrum is None:
                sum_spectrum = spec
            else:
                sum_spectrum += spec

            count += 1

    if count == 0:
        raise RuntimeError("No frames above threshold found. Try lowering THRESHOLD_MULTIPLIER.")

    avg_spectrum = sum_spectrum / count

    # Normalize (so only shape matters, not loudness)
    norm = np.linalg.norm(avg_spectrum) + 1e-9
    avg_spectrum_normalized = avg_spectrum / norm

    print(f"Used {count} frames to build whistle profile.")
    return avg_spectrum_normalized


def main():
    audio, sr = load_audio(FILE_PATH)
    print(f"Loaded {FILE_PATH}, sample rate = {sr} Hz")

    frame_energy, frames = compute_energy(audio, sr, WINDOW_DURATION)

    whistle_profile = build_whistle_profile(
        frames,
        frame_energy,
        sr,
        WINDOW_DURATION,
        THRESHOLD_MULTIPLIER
    )

    # Save the profile to a .npy file
    np.save("whistle_profile.npy", whistle_profile)
    print("\nSaved whistle profile to whistle_profile.npy")


if __name__ == "__main__":
    main()
