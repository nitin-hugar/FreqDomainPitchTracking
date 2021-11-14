from typing import Tuple
import numpy as np
import scipy.io.wavfile
import scipy.signal
from matplotlib import pyplot as plt
import os.path
from glob import glob

T = np.ndarray  # for autocomplete engine, will be deleted before submission
BLOCK_SIZE = 1 << 10  # SHOULD NOT BE USED IN FUNCTION DEFINIION
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
        xb[i] = x[l: l + blockSize]

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
    X = fft_block(xb)  # shape: (num_of_blocks, (block_size >> 1) + 1)
    fInHz = np.fft.rfftfreq(
        block_size, d=1 / fs
    )  # shape: ((block_size >> 1) + 1, )
    return X, fInHz


# --- A.2 ---
def track_pitch_fftmax(x: T, blockSize: int, hopSize: int, fs: int):
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    Xb, f_in_hz = compute_spectrogram(xb, fs)
    f0 = f_in_hz[np.argmax(Xb, axis=-1)]
    return f0, timeInSec  # both have shape (num_of_block, )


# --- B.1 ---
def get_f0_from_Hps(X, fs, order):
    hps = X.copy()
    fInHz = np.fft.rfftfreq((X.shape[-1] - 1) << 1, d=1 / fs)
    for i in range(1, order):
        Xd = X[:, :: i + 1]
        hps = hps[:, : Xd.shape[1]]
        hps *= Xd ** 2
    f0 = fInHz[np.argmax(hps, axis=-1)]
    return f0


# --- B.2 ---
def track_pitch_hps(x, blockSize, hopSize, fs):
    xb, time_in_sec = block_audio(x, blockSize, hopSize, fs)
    Xb, f_in_hz = compute_spectrogram(xb, fs)
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
    mask = 0 + (rmsDb >= thresholdDb)
    return mask


# --- C.3 ---
def apply_voicing_mask(f0, mask):
    f0Adj = f0 * mask
    return f0Adj


# --- D.1 ---
def eval_voiced_fp(estimation, annotation):
    # false positive: Actually negative, predicted positive
    fp = np.sum(estimation != 0) / np.sum(annotation == 0)
    return fp


# --- D.2 ---
def eval_voiced_fn(estimation, annotation):
    # false negative
    fn = np.sum(estimation == 0) / np.sum(annotation != 0)
    return fn


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
def executeassign3():
    def gen_sin(f, t, fs):
        return np.sin(2 * np.pi * f * np.arange(t * fs) / fs)

    fs = 44100
    sin_ = np.concatenate((gen_sin(441, 1, fs), gen_sin(882, 1, fs)))
    ref_ = np.concatenate(
        (np.ones(shape=(fs,)) * 441, np.ones(shape=(fs,)) * 882)
    )
    ref_in_block = np.max(
        block_audio(ref_, BLOCK_SIZE, HOP_SIZE, fs)[0], axis=1
    )

    f0_fftmax, time_in_sec = track_pitch_fftmax(sin_, BLOCK_SIZE, HOP_SIZE, fs)
    f0_hps, _ = track_pitch_hps(sin_, BLOCK_SIZE, HOP_SIZE, fs)
    # plot
    fig, ax = plt.subplots()
    ax.plot(time_in_sec, f0_fftmax, label="estimated f0")
    ax.plot(time_in_sec, f0_hps, label="estimated f0")
    ax.plot(time_in_sec, ref_in_block, label="ground truth")
    ax.set_ylim(0)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("estimated frequency (hz)")
    ax2 = ax.twinx()
    ax2.plot(time_in_sec, ref_in_block - f0_fftmax, "--", label="error fftmax")
    ax2.plot(time_in_sec, ref_in_block - f0_hps, "--", label="error hps")
    ax2.set_ylabel("error (hz)")
    a, b = ax.get_legend_handles_labels()
    c, d = ax2.get_legend_handles_labels()
    ax.legend(a + c, b + d, loc=2)
    plt.savefig("sig_error.png")


# --- E.3 ---
# rename run_evaluation function in the end

# Report the results in the document
def run_evaluation_fftmax(complete_path_to_data_folder):
    wavpath = np.array([])
    txtpath = np.array([])
    for full_filepath in os.listdir(complete_path_to_data_folder):
        if full_filepath.endswith(".wav"):
            wav_path = np.append(wavpath, complete_path_to_data_folder + full_filepath)
            txt_path = np.append(txtpath, complete_path_to_data_folder + full_filepath.split(".")[0]
                                 + ".f0.Corrected.txt")

    blockSize = 1024
    hopSize = 512

    all_estimates = np.array([])
    all_groundtruths = np.array([])
    for wavfile, txtfile in zip(wav_path, txt_path):
        fs, x = tool_read_audio(wavfile)
        file = np.loadtxt(txtfile)
        groundtruths = file[:, 2]

        estimates, timestamps = track_pitch_fftmax(x, blockSize, hopSize, fs)
        all_estimates = np.append(all_estimates, estimates)
        all_groundtruths = np.append(all_groundtruths, groundtruths)

    errCentRms = eval_pitchtrack_v2(all_estimates, all_groundtruths)
    return errCentRms


# Report the results
def run_evaluation_hps(complete_path_to_data_folder):
    wavpath = np.array([])
    txtpath = np.array([])
    for full_filepath in os.listdir(complete_path_to_data_folder):
        if full_filepath.endswith(".wav"):
            wav_path = np.append(
                wavpath, complete_path_to_data_folder + full_filepath
            )
            txt_path = np.append(
                txtpath,
                complete_path_to_data_folder
                + full_filepath.split(".")[0]
                + ".f0.Corrected.txt",
            )

    blockSize = 1024
    hopSize = 512

    all_estimates = np.array([])
    all_groundtruths = np.array([])
    for wavfile, txtfile in zip(wav_path, txt_path):
        fs, x = tool_read_audio(wavfile)
        file = np.loadtxt(txtfile)
        groundtruths = file[:, 2]

        estimates, timestamps = track_pitch_hps(x, blockSize, hopSize, fs)
        all_estimates = np.append(all_estimates, estimates)
        all_groundtruths = np.append(all_groundtruths, groundtruths)

    errCentRms = eval_pitchtrack_v2(all_estimates, all_groundtruths)
    return errCentRms


# Alex implementation of ACF
# def comp_acf(inputVector, bIsNormalized=True):
#     if bIsNormalized:
#         norm = np.dot(inputVector, inputVector)
#     else:
#         norm = 1
#     afCorr = np.correlate(inputVector, inputVector, "full") / norm
#     afCorr = afCorr[np.arange(inputVector.size - 1, afCorr.size)]
#     return afCorr


# def get_f0_from_acf(r, fs):
#     eta_min = 1
#     afDeltaCorr = np.diff(r)
#     eta_tmp = np.argmax(afDeltaCorr > 0)
#     eta_min = np.max([eta_min, eta_tmp])
#     f = np.argmax(r[np.arange(eta_min + 1, r.size)])
#     f = fs / (f + eta_min + 1)
#     return f

# yet another E.5
def get_f0_from_acf(corr, fs):
    # takes the output of comp_acf and computes and returns the fundamental
    # frequency f0 of that block in Hz.
    # TODO what if corr is all zero?
    # TODO what if fs % f0 != 0?

    # still use monotonic stack to solve this.
    def calc_view_range(x):
        stack = []  # (index, value)
        view_range_l = [0] * len(corr)  # leftmost index of samples < cur
        view_range_r = [len(corr)] * len(
            corr
        )  # rightmost index of samples < cur

        for i, val in enumerate(x):
            while stack and val > stack[-1][1]:
                j, _ = stack.pop()
                view_range_r[j] = i
            view_range_l[i] = 0 if not stack else stack[-1][0] + 1  # [l, r)
            stack.append(
                (i, val)
            )  # the stack would be monotonically decreasing

        view_range = [
            (i, l, r)
            for i, (l, r) in enumerate(zip(view_range_l, view_range_r))
        ]
        view_range.sort(
            key=lambda x: x[0]
        )  # the lower frequency, the more likely
        view_range.sort(
            key=lambda x: x[2] - x[1], reverse=True
        )  # the bigger view the more likely.
        return view_range

    # get first local minimum, and the first peak after this sample would be
    # viewed as f0.
    local_min = np.logical_and(
        ((-corr)[:-1] > (-corr)[1:])[1:], ((-corr)[:-1] < (-corr)[1:])[:-1]
    )
    first_local_minimum = np.argmax(local_min, axis=0)

    view_range = calc_view_range(corr)
    for i, _, _ in view_range:
        if i > first_local_minimum:
            return fs / i

    return 0


def pad_with_trailing_zero(x: T, target_len: int):
    new_x = np.zeros((*x.shape[:-1], target_len))
    new_x[..., : x.shape[-1]] = x
    return new_x


def comp_acf(inputVector):
    n = len(inputVector)
    x = pad_with_trailing_zero(inputVector, n << 1)
    X = np.fft.fft(x)
    A = X * np.conj(X)
    a = np.abs(np.fft.ifft(A)[..., :n])
    return a


def comp_acf_direct(inputVector, bIsNormalized):
    n = len(inputVector)
    result = np.fromiter(
        (np.sum(inputVector[k:] * inputVector[: n - k]) for k in range(n)),
        dtype=float,
    )
    if bIsNormalized and result[0] > 0:
        result /= result[0]
    return result


def track_pitch_acf(x, blockSize, hopSize, fs):
    xb, time_in_sec = block_audio(x, blockSize, hopSize, fs)
    f0 = np.zeros_like(time_in_sec)
    for i, row in enumerate(xb):
        corr = comp_acf(row)
        f0[i] = get_f0_from_acf(corr, fs)
    return f0, time_in_sec


# --- E.6 ---
def track_pitch(x, blockSize, hopSize, fs, method, voicingThres=-40):
    func = {
        "acf": track_pitch_acf,
        "max": track_pitch_fftmax,
        "hps": track_pitch_hps,
    }
    if method not in func:
        print("warning: parameter `method` is not set, using 'hps' as default")
        method = "hps"
    f0, _ = func[method](x, blockSize, hopSize, fs)

    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    rmsDb = extract_rms(xb)
    mask = create_voicing_mask(rmsDb, voicingThres)
    f0Adj = apply_voicing_mask(f0, mask)
    return f0Adj, timeInSec

# --- BONUS ---

def centre_clip(x , clip_threshold=1e-3):
    x_clipped = np.clip(x, -clip_threshold, clip_threshold)
    return x_clipped

def comp_amdf(inputVector):
    y = np.zeros(inputVector.shape[0] * 2)
    y[:inputVector.shape[0]] = inputVector
    g = np.zeros(inputVector.shape[0])
    for n in range(inputVector.shape[0]):
        g[n] = np.sum(y - np.roll(y, -n))
    return g


def amdf_weighted_acf(inputVector, bIsNormalized=True):
    alpha = 1
    r = comp_acf(inputVector)
    g = comp_amdf(inputVector)
    r_weighted = r/(g + alpha)
    if bIsNormalized:
        r_weighted_max = np.max(r_weighted)
        if r_weighted_max:
            r_weighted /= r_weighted_max
    return r_weighted


def eval_track_pitch(complete_path_to_data_folder):
    wavpath = np.array([])
    txtpath = np.array([])
    for full_filepath in os.listdir(complete_path_to_data_folder):
        if full_filepath.endswith(".wav"):
            wav_path = np.append(
                wavpath, complete_path_to_data_folder + full_filepath
            )
            txt_path = np.append(
                txtpath,
                complete_path_to_data_folder
                + full_filepath.split(".")[0]
                + ".f0.Corrected.txt",
            )

    blockSize = 1024
    hopSize = 512
    errCentRms = np.zeros((3, 2, 3))
    for i, method in enumerate(["acf", "max", "hps"]):
        for j, voicingThres in enumerate([-40, -20]):
            all_estimates = np.array([])
            all_groundtruths = np.array([])
            for wavfile, txtfile in zip(wav_path, txt_path):
                fs, x = tool_read_audio(wavfile)
                file = np.loadtxt(txtfile)
                groundtruths = file[:, 2]
                estimates, timestamps = track_pitch(
                    x, blockSize, hopSize, fs, method, voicingThres
                )
                all_estimates = np.append(all_estimates, estimates)
                all_groundtruths = np.append(all_groundtruths, groundtruths)
            errCentRms[i, j] = eval_pitchtrack_v2(
                all_estimates, all_groundtruths
            )
    return errCentRms


# bonus
def apply_band_pass_filter(x: T, fs: int):
    bp_filter = scipy.signal.butter(N=20, Wn=(50, 2000), btype="bandpass", output="sos", fs=fs)
    bp_x = scipy.signal.sosfilt(bp_filter, x)
    return bp_x

# bonus
def get_f0_from_acf_mod(corr, fs):
    # takes the output of comp_acf and computes and returns the fundamental
    # frequency f0 of that block in Hz.
    # TODO what if corr is all zero?
    # TODO what if fs % f0 != 0?

    # still use monotonic stack to solve this.
    def calc_view_range(x):
        stack = []  # (index, value)
        view_range_l = [0] * len(corr)  # leftmost index of samples < cur
        view_range_r = [len(corr)] * len(
            corr
        )  # rightmost index of samples < cur

        for i, val in enumerate(x):
            while stack and val > stack[-1][1]:
                j, _ = stack.pop()
                view_range_r[j] = i
            view_range_l[i] = 0 if not stack else stack[-1][0] + 1  # [l, r)
            stack.append(
                (i, val)
            )  # the stack would be monotonically decreasing

        view_range = [
            (i, l, r)
            for i, (l, r) in enumerate(zip(view_range_l, view_range_r))
        ]
        view_range.sort(
            key=lambda x: x[0]
        )  # the lower frequency, the more likely
        view_range.sort(
            key=lambda x: x[2] - x[1], reverse=True
        )  # the bigger view the more likely.
        return view_range

    # get first local minimum, and the first peak after this sample would be
    # viewed as f0.
    local_min = np.logical_and(
        ((-corr)[:-1] > (-corr)[1:])[1:], ((-corr)[:-1] < (-corr)[1:])[:-1]
    )
    first_local_minimum = np.argmax(local_min, axis=0)

    view_range = calc_view_range(corr)
    for i, _, _ in view_range:
        if i > first_local_minimum:
            return i

    return 0

# bonus
def track_pitch_mod(x, blockSize, hopSize, fs):
    """
    de Obaldía, C., & Zölzer, U. (2019). IMPROVING MONOPHONIC PITCH DETECTION USING THE ACF AND SIMPLE HEURISTICS.

    Step1:  Centre clip with threshold 1e-3
    Step2:  Highpass signal at 50Hz
    Step3:  Lowpass signal at 2000Hz
    Step4:  Blocking and Windowing
    Step5:  ACF
    Step6:  Apply AMDF weighting function
    Step7:  Output is normalized to the maximum of the resulting signal
    Step8:
    """
    xb_clipped = centre_clip(x)
    bp_x = apply_band_pass_filter(x, fs)
    xb, timeInSec = block_audio(bp_x, blockSize, hopSize, fs)
    f0 = np.zeros_like(timeInSec)
    for i, row in enumerate(xb):
        corr = amdf_weighted_acf(row)
        f0_index = get_f0_from_acf_mod(corr, fs)
        
        # apply 3.5 parabolic interpolation
        if 1 <= f0_index < blockSize - 1:
            y1, y2, y3 = map(corr.__getitem__, range(f0_index-1, f0_index+2))
            delta_x_max = 0.5 + 0.5 * ((y1 - y2) * (y2 - y3) * (y3 - y1)) / (2 * y2 - y1 - y3)
            f0[i] = fs / (f0_index + delta_x_max)
        else:
            f0[i] = 0
    return f0, timeInSec


# bonus
def eval_pitchtrack(estimateInHz, groundtruthInHz):
    # return RMS of cent
    res = 0
    for a, b in zip(estimateInHz, groundtruthInHz):
        if a and b:
            res += (1200 * np.log2(a / b)) ** 2
    return (res / len(estimateInHz)) ** 0.5


# bonus
def run_evaluation(complete_path_to_data_folder):
    rms_s, n_s = [], []
    for full_filename in glob("%s/*.wav" % complete_path_to_data_folder):
        filepath, filename_ext = os.path.split(full_filename)
        filename, ext = os.path.splitext(filename_ext)
        fs, x = scipy.io.wavfile.read(full_filename)

        # first calculate f0
        for track_pitch_func in (
            track_pitch_acf, track_pitch_fftmax, track_pitch_hps, track_pitch_mod
        ):
            f0, _ = track_pitch_func(x, BLOCK_SIZE, HOP_SIZE, fs)
            xb, _ = block_audio(x, BLOCK_SIZE, HOP_SIZE, fs)
            rms = extract_rms(xb)
            voicing_mask = create_voicing_mask(rms, -30)
            masked_f0 = f0 * voicing_mask

            # then read file
            ground_truth_f0, ground_truth_time_in_sec = [], []
            with open("%s/%s.f0.Corrected.txt" % (filepath, filename)) as f:
                for line in f.readlines():
                    onset_seconds, _, pitch_frequency, _ = map(float, line.split())
                    ground_truth_f0.append(pitch_frequency)
                    ground_truth_time_in_sec.append(onset_seconds)
            ground_truth_f0 = np.array(ground_truth_f0)
            ground_truth_time_in_sec = np.array(ground_truth_time_in_sec)

            rms_s.append(eval_pitchtrack(masked_f0[:-1], ground_truth_f0[1:]))
            n_s.append(len(f0) - 1)

            print("RMS of %s on %s: %.5f" % (filename, track_pitch_func.__name__, rms_s[-1]))

    overall_err_cent_rms = np.sqrt(
        sum(r ** 2 * n for r, n in zip(rms_s, n_s)) / sum(n_s)
    )
    print("Overall RMS: %.5f" % overall_err_cent_rms)
    return overall_err_cent_rms

if __name__ == "__main__":

    run_evaluation("./trainData/")

    # executeassign3()

    # fs, x = tool_read_audio("trainData/01-D_AMairena.wav")
    # for method in ("max", "hps"):
    #     f0Adj, timeInSec = track_pitch(x, 1024, 512, fs, method, -40)
    #     plt.clf()
    #     plt.plot(f0Adj)
    #     plt.show()

    # print(eval_track_pitch("./trainData/"))

    # fs, x = tool_read_audio("trainData/01-D_AMairena.wav")
    # f0, timeInSec = track_pitch_mod(x, 1024, 512, fs)
    # plt.plot(f0)

    # for full_filename in glob("./trainData/*.wav"):
    #     filepath, filename_ext = os.path.split(full_filename)

    #     fs, x = tool_read_audio(full_filename)
    #     xb, time_in_sec = block_audio(x, BLOCK_SIZE, HOP_SIZE, fs)
    #     Xb, f_in_hz = compute_spectrogram(xb, fs)
    #     # plt.plot(time_in_sec, rms)
    #     # plt.plot(time_in_sec, voicing_mask)
    #     # plt.show()
    #     # plt.clf()
    #     # print(Xb.shape, f0_fftmax.shape, f0_hps.shape)

    #     plt.figure(figsize=(36, 6))
    #     # plt.subplot(2, 1, 1)
    #     plt.imshow(
    #         magnitude_to_db(Xb.T),
    #         cmap="inferno",
    #         origin="lower",
    #         extent=[0, time_in_sec[-1], 0, fs / 2],
    #     )
    #     plt.plot(time_in_sec, f0_fftmax, label="estimated f0 (fft max)")
    #     plt.plot(time_in_sec, f0_hps, label="estimated f0 (hps)")
    #     plt.plot(time_in_sec, f0_acf, label="estimated f0 (acf)")
    #     plt.plot(time_in_sec, f0_mod, label="estimated f0 (acf mod)")
    #     plt.ylim(bottom=10)
    #     plt.yscale("log")
    #     plt.ylabel("Magnitude Response [Hz]")
    #     plt.xlabel("Time [s]")
    #     plt.legend(loc="upper left")
    #     plt.colorbar(format="%+2.0f dB", pad=0.01)
    #     plt.title("block size = %d, hop size = %d" % (BLOCK_SIZE, HOP_SIZE))
    #     plt.show()
    #     break
