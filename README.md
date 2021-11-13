# FreqDomainPitchTracking

## To do

- [x] A. Maximum spectral peak based pitch tracker
  - [x] Implement a function `[X, fInHz] = compute_spectrogram(xb, fs)`
  - [x] Implement a function `[f0, timeInSec] = track_pitch_fftmax(x, blockSize, hopSize, fs)`
  - [x] If the blockSize = 1024 for blocking, what is the exact time resolution of your pitch tracker? Can this be improved without changing the block-size? If yes, how? If no, why? (Use a sampling rate of 44100Hz for all calculations).
    - The time resolution is $\frac{f_s}{\mathrm{fft\_block}} = \frac{44100}{1024} = 43.066\mathrm{Hz}$.
    - The resolution can be improved with smaller `hop_size`.
    - ![A.3_low_res_1024_1024](imgs/CROPPED/A.3_low_res_1024_1024.png)
    - ![A.3_high_res_1024_1024](imgs/CROPPED/A.3_high_res_1024_64.png)

- [x] B. HPS (Harmonic Product Spectrum) based pitch tracker
  - [x] Implement a function `[f0] = get_f0_from_Hps(X, fs, order)`
  - [x] Implement a function `[f0, timeInSec] = track_pitch_hps(x, blockSize, hopSize, fs)`

- [x] C. Voicing Detection
  - [x] Take the function `[rmsDb] = extract_rms(xb)`
  - [x] Implement a function `[mask] = create_voicing_mask(rmsDb, thresholdDb)`
  - [x] Implement a function `[f0Adj] = apply_voicing_mask(f0, mask)`

- [x] D. Different Evaluation Metrics
  - [x] Implement a function `[pfp] = eval_voiced_fp(estimation, annotation)`
  - [x] Implement a function `[pfn] = eval_voiced_fn(estimation, annotation)`
  - [x] Now modify the `eval_pitchtrack()` method that you wrote in Assignment 1 to `[errCentRms, pfp, pfn] = eval_pitchtrack_v2(estimation, annotation)`, to return all the 3 performance metrics

- [x] E. Evaluation
  - [x] `executeassign3()`
  - [ ] Same experiment but with different block & hop size.
  - [ ] Evaluate your `track_pitch_fftmax()` and the `eval_pitchtrack_v2()` 
  - [ ] Evaluate your `track_pitch_hps()` using the development set and the `eval_pitchtrack_v2()` method. Report the average performance metrics across the development set
  - [ ] Implement a MATLAB wrapper function `[f0Adj, timeInSec] = track_pitch(x, blockSize, hopSize, fs, method, voicingThres)` that takes audio signal `x` and related paramters (fs, blockSize, hopSize), calls the appropriate pitch tracker based on the method parameter to compute the fundamental frequency and then applies the voicing mask based on the threshold parameter.
  - [ ] Evaluate your `track_pitch()` using the development set and the `eval_pitchtrack_v2()` method over all 3 pitch trackers (acf, max and hps) and report the results with two values of threshold (threshold = -40, -20)

