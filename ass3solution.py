from typing import Tuple
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import os.path
from glob import glob


T = np.ndarray  # for autocomplete engine, will be deleted before submission
BLOCK_SIZE = 1 << 10
HOP_SIZE = BLOCK_SIZE >> 2
DB_TRUNCATION_THRESHOLD = -100


# proudly plagiarized myself, again, by simply copying from assg 1.
def block_audio(x, blockSize, hopSize, fs) -> Tuple[T, T]:
    """plain implementation of spliting input signal into blocks of frames."""

    # equivalent to num_of_blocks = ceil(x / hopSize)
    num_of_blocks = len(x) // hopSize
    if len(x) % hopSize:
        num_of_blocks += 1
    x = np.pad(x, (0, (num_of_blocks - 1) * hopSize + blockSize - len(x)))

    xb = np.zeros((num_of_blocks, blockSize))
    time_in_sec = np.zeros((num_of_blocks,))

    for i in range(num_of_blocks):
        # i-th block
        l = i * hopSize
        time_in_sec[i] = l / fs
        xb[i] = x[l : l + blockSize]

    return xb, time_in_sec


# plagiarized from alex
def tool_read_audio(cAudioFilePath):
    samplerate, x = scipy.io.wavfile.read(cAudioFilePath)

    if x.dtype == "float32":
        audio = x
    else:
        # change range to [-1,1)
        nbits = {
            np.dtype("uint8"): 8,
            np.dtype("int16"): 16,
            np.dtype("int32"): 32,
        }[x.dtype]

        audio = x / (1 << (nbits - 1))

    # special case of unsigned format
    if x.dtype == "uint8":
        audio = audio - 1.0

    return (samplerate, audio)


def fft_block(xb):
    """Helper function that fft (one) block of audio.

    Args:
        xb (T): blocked signal, with shape (..., BLOCK_SIZE)

    Returns:
        T: corresponding magnitude response of `xb`
    """
    # hann window
    # - https://en.wikipedia.org/wiki/Hann_function
    # - $w[n] = \sin^2\frac{\pi n}{N}$
    w = np.sin(np.pi * np.arange(xb.shape[-1]) / xb.shape[-1]) ** 2

    # lets hope it will boardcast from (BLOCK_SIZE,) to (..., BLOCK_SIZE)
    windowed_xb = xb * w

    # fft, magnitude response
    X_b = np.abs(np.fft.rfft(windowed_xb))  # (..., (BLOCK_SIZE >> 1) + 1)

    return X_b


def amplitude_to_db(xb):
    """Convert amplitude into dB scale. Truncated at -100dB."""
    return np.maximum(20 * np.log10(xb), DB_TRUNCATION_THRESHOLD)


def magnitude_to_db(X_b):
    # see https://dsp.stackexchange.com/questions/47173/theoretical-maximum-of-dft
    # for the reason why divide by `BLOCK_SIZE` here.
    return amplitude_to_db(X_b / BLOCK_SIZE)


# --- A.1 ---
def compute_spectrogram(xb: T, fs: int):
    # xb.shape = (num_of_blocks, block_size)
    block_size = xb.shape[-1]
    Xb = fft_block(xb)  # shape: (num_of_blocks, (block_size >> 1) + 1)
    f_in_hz = np.fft.rfftfreq(
        block_size, d=1 / fs
    )  # shape: ((block_size >> 1) + 1, )
    return Xb, f_in_hz


# --- A.2 ---
def track_pitch_fftmax(x: T, blockSize: int, hopSize: int, fs: int):
    xb, time_in_sec = block_audio(x, blockSize, hopSize, fs)
    Xb, f_in_hz = compute_spectrogram(xb, fs)
    f0 = f_in_hz[np.argmax(Xb, axis=-1)]
    return f0, time_in_sec  # both have shape (num_of_block, )


# --- B.1 ---
def get_f0_from_Hps(X, fs, order):
    hps = X.copy()
    for i in range(1, order):
        Xd = X[:, :: i + 1]
        hps = hps[:, : Xd.shape[1]]
        hps *= Xd
    f0 = f_in_hz[np.argmax(hps, axis=-1)]  # f_in_hz is outside variable
    return f0


# --- B.2 ---
def track_pitch_hps(x, blockSize, hopSize, fs):
    xb, time_in_sec = block_audio(
        x, blockSize=blockSize, hopSize=hopSize, fs=fs
    )
    Xb, f_in_hz = compute_spectrogram(xb, fs=fs)
    f0 = get_f0_from_Hps(Xb, fs, order=4)
    return f0, time_in_sec


# --- C.1 ---
def extract_rms(xb):
    rms = np.maximum(
        20 * np.log10(np.sqrt(np.mean(xb ** 2, axis=-1))),
        DB_TRUNCATION_THRESHOLD,
    )
    return rms


# --- C.2 ---
def create_voicing_mask(rmsDb, thresholdDb):
    return 0 + rmsDb >= thresholdDb


# --- C.3 ---
def apply_voicing_mask(f0, mask):
    f0Adj = f0 * mask
    return f0Adj


# --- D.1 ---
def eval_voiced_fp(estimation, annotation):
    # false positive: Actually negative, predicted positive
    return np.sum(estimation != 0) / np.sum(annotation == 0)


# --- D.2 ---
def eval_voiced_fn(estimation, annotation):
    # false negative
    return np.sum(estimation == 0) / np.sum(annotation != 0)


# --- D.3 ---
def eval_pitchtrack_v2(estimation, annotation):
    # return RMS of cent
    err_cent_rms = 0
    for a, b in zip(estimation, annotation):
        if a and b:
            err_cent_rms += (1200 * np.log2(a / b)) ** 2
    err_cent_rms = (err_cent_rms / len(estimation)) ** 0.5
    pfp = eval_voiced_fp(estimation, annotation)
    pfn = eval_voiced_fn(estimation, annotation)
    return err_cent_rms, pfp, pfn


# --- E.1 ---


if __name__ == "__main__":
    for full_filename in glob("./trainData/*.wav"):
        filepath, filename_ext = os.path.split(full_filename)

        fs, x = tool_read_audio(full_filename)
        xb, time_in_sec = block_audio(x, BLOCK_SIZE, HOP_SIZE, fs)
        Xb, f_in_hz = compute_spectrogram(xb, fs)
        f0_fftmax, _ = track_pitch_fftmax(x, BLOCK_SIZE, HOP_SIZE, fs)
        f0_hps, _ = track_pitch_hps(x, BLOCK_SIZE, HOP_SIZE, fs)
        rms = extract_rms(xb)
        voicing_mask = create_voicing_mask(rms, -30)
        plt.plot(time_in_sec, rms)
        plt.plot(time_in_sec, voicing_mask)
        plt.show()
        plt.clf()
        print(Xb.shape, f0_fftmax.shape, f0_hps.shape)

        plt.figure(figsize=(36, 6))
        # plt.subplot(2, 1, 1)
        plt.imshow(
            magnitude_to_db(Xb.T),
            cmap="inferno",
            origin="lower",
            extent=[0, time_in_sec[-1], 0, fs / 2],
        )
        plt.plot(time_in_sec, f0_fftmax, label="estimated f0 (fft max)")
        plt.plot(time_in_sec, f0_hps, label="estimated f0 (hps)")
        plt.ylim(bottom=10)
        plt.yscale("log")
        plt.ylabel("Magnitude Response [Hz]")
        plt.xlabel("Time [s]")
        plt.legend(loc="upper left")
        plt.colorbar(format="%+2.0f dB", pad=0.01)
        plt.title("block size = %d, hop size = %d" % (BLOCK_SIZE, HOP_SIZE))
        plt.show()
