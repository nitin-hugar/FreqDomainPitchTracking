from typing import Tuple
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import os.path
from glob import glob


T = np.ndarray  # for autocomplete engine, will be deleted before submission
BLOCK_SIZE = 1 << 10
HOP_SIZE = BLOCK_SIZE >> 4
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
    Xb = fft_block(xb)  # shape: (num_of_blocks, (block_size >> 1) + 1)
    f_in_hz = np.fft.rfftfreq(BLOCK_SIZE, d=1/fs)  # shape: ((block_size >> 1) + 1, )
    return Xb, f_in_hz


# -- A.2 ---
def track_pitch_fftmax(x: T, blockSize: int, hopSize: int, fs: int):
    xb, time_in_sec = block_audio(x, blockSize, hopSize, fs)
    Xb, f_in_hz = compute_spectrogram(xb, fs)
    f0 = f_in_hz[np.argmax(Xb, axis=-1)]
    return f0, time_in_sec  # both have shape (num_of_block, )


if __name__ == "__main__":
    for full_filename in glob("./trainData/*.wav"):
        filepath, filename_ext = os.path.split(full_filename)

        fs, x = tool_read_audio(full_filename)
        xb, time_in_sec = block_audio(x, BLOCK_SIZE, HOP_SIZE, fs)
        Xb, f_in_hz = compute_spectrogram(xb, fs)
        f0, _ = track_pitch_fftmax(x, BLOCK_SIZE, HOP_SIZE, fs)
        print(Xb.shape, f0.shape)

        plt.figure(figsize=(36, 6))
        # plt.subplot(2, 1, 1)
        plt.imshow(
            magnitude_to_db(Xb.T),
            cmap="inferno",
            origin="lower",
            extent=[0, time_in_sec[-1], 0, fs / 2],
        )
        plt.plot(time_in_sec, f0, label="estimated f0")
        plt.ylim(bottom=10)
        plt.yscale("log")
        plt.ylabel("Magnitude Response [Hz]")
        plt.xlabel("Time [s]")
        plt.legend(loc="upper left")
        plt.colorbar(format="%+2.0f dB", pad=0.01)
        plt.title("block size = %d, hop size = %d" % (BLOCK_SIZE, HOP_SIZE))
        plt.show()
