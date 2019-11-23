# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
STFT/ISTFT derived from Basj's implementation[1], with minor modifications,
such as the replacement of the hann window by its root square, as specified in
the original paper from Hershey et. al. (2015)[2]

[1] http://stackoverflow.com/a/20409020
[2] https://arxiv.org/abs/1508.04306
"""
import os
import random
import numpy as np
import soundfile as sf
from config import DEEPC_BASE, FRAME_LENGTH, FRAME_SHIFT, FRAME_RATE
from config import TIMESTEPS, DB_THRESHOLD


def sqrt_hann(M):
    """Create the square root Hann window"""
    return np.sqrt(np.hanning(M))


def stft(x, fftsize=int(FRAME_LENGTH*FRAME_RATE),
         overlap=FRAME_LENGTH//FRAME_SHIFT):
    """
    Perform the short-time Fourier transform (STFT).

    Parameters
    ----------
    x: array_like
        input waveform (1D array of samples)

    fftsize:
        in samples, size of the fft window

    overlap:
        should be a divisor of fftsize, represents the rate of
        window superposition (window displacement=fftsize/overlap)

    Returns
    -------
    y: ndarray
        Linear domain spectrum (2D complex array)
    """
    hop = int(np.round(fftsize / overlap))
    w = sqrt_hann(fftsize)
    out = np.array([np.fft.rfft(w*x[i:i+fftsize])
                    for i in range(0, len(x)-fftsize, hop)])
    return out


def istft(X, overlap=FRAME_LENGTH//FRAME_SHIFT):
    """
    Perform the inverse short-time Fourier transform (iSTFT).

    Parameters
    ----------
    X : array_like
        input spectrum (2D complex array)
    overlap: int, optional
        The rate of window superposition. Should be a divisor of
        ``(X.shape[1] - 1) * 2``, (window displacement=fftsize/overlap)

    Returns
    -------
    y : ndarray of real
        Floating-point waveform samples (1D array)
    """
    fftsize = (X.shape[1] - 1) * 2
    hop = int(np.round(fftsize / overlap))
    w = sqrt_hann(fftsize)
    x = np.zeros(X.shape[0]*hop)
    wsum = np.zeros(X.shape[0]*hop)
    for n, i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += np.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x


def get_egs(wavlist, min_mix=2, max_mix=3, batch_size=1):
    """
    Generate examples for the neural network.

    Parameters
    ----------
    wavlist : string
        Path to a text file containing a list of wave files with
        speaker ids. Each line is of type "path speaker", as follows::
        path/to/1st.wav spk1
        path/to/2nd.wav spk2
        path/to/3rd.wav spk1
        ...
    min_mix : int
        Minimum number of speakers to mix
    max_max : int
        Maximum number of speakers to mix
    batch_size : int
        Number of examples to generate
    """
    speaker_wavs = {}
    batch_x = []
    batch_y = []
    batch_count = 0

    while True:  # Generate examples indefinitely

        # Select number of files to mix
        k = np.random.randint(min_mix, max_mix + 1)
        if k > len(speaker_wavs):

            # Reading wav files list and separating per speaker
            speaker_wavs = {}
            f = open(wavlist)
            for line in f:
                line = line.strip().split()
                if len(line) != 2:
                    continue
                p, spk = line
                if spk not in speaker_wavs:
                    speaker_wavs[spk] = []
                speaker_wavs[spk].append(p)
            f.close()
            # Randomizing wav lists
            for spk in speaker_wavs:
                random.shuffle(speaker_wavs[spk])
        wavsum = None
        sigs = []

        # Pop wav files from random speakers, store them individually for
        # dominant spectra decision and generate the mixed input
        for spk in random.sample(speaker_wavs.keys(), k):
            p = speaker_wavs[spk].pop()
            if not speaker_wavs[spk]:
                del(speaker_wavs[spk])  # Remove empty speakers from dictionary
            sig, rate = sf.read(p)
            if rate != FRAME_RATE:
                raise Exception("Config specifies " + str(FRAME_RATE) +
                                "Hz as sample rate, but file " + str(p) +
                                "is in " + str(rate) + "Hz.")
            sig = sig - np.mean(sig)
            sig = sig/np.max(np.abs(sig))
            sig *= (np.random.random()*1/4 + 3/4)
            if wavsum is None:
                wavsum = sig
            else:
                wavsum = wavsum[:len(sig)] + sig[:len(wavsum)]
            sigs.append(sig)

        # STFT for mixed signal
        def get_logspec(sig):
            return np.log10(np.absolute(stft(sig)) + 1e-7)

        X = get_logspec(wavsum)
        if len(X) <= TIMESTEPS:
            continue

        # STFTs for individual signals
        specs = []
        for sig in sigs:
            specs.append(get_logspec(sig[:len(wavsum)]))
        specs = np.array(specs)

        nc = max_mix

        # Get dominant spectra indexes, create one-hot outputs
        Y = np.zeros(X.shape + (nc,))
        vals = np.argmax(specs, axis=0)
        for i in range(k):
            t = np.zeros(nc)
            t[i] = 1
            Y[vals == i] = t

        # Create mask for zeroing out gradients from silence components
        m = np.max(X) - DB_THRESHOLD/20.  # From dB to log10 power
        z = np.zeros(nc)
        Y[X < m] = z

        # Generating sequences
        i = 0
        while i + TIMESTEPS < len(X):
            batch_x.append(X[i:i+TIMESTEPS])
            batch_y.append(Y[i:i+TIMESTEPS])
            i += TIMESTEPS//2

            batch_count = batch_count+1

            if batch_count == batch_size:
                inp = np.array(batch_x).reshape((batch_size,
                                                 TIMESTEPS, -1))
                print(inp.shape)
                out = np.array(batch_y).reshape((batch_size,
                                                 TIMESTEPS, -1))
                yield({'input': inp},
                      {'kmeans_o': out})
                batch_x = []
                batch_y = []
                batch_count = 0


if __name__ == "__main__":
    path_trn = os.path.join(DEEPC_BASE, 'train')
    x, y = next(get_egs(path_trn, batch_size=50))
    print(x['input'].shape)
    print(y['kmeans_o'].shape)
