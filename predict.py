# -*- coding: utf-8 -*-
# pylint: disable=C0103,R0912,R0913,R0914,R0915
"""
Created on Mon Oct 17 10:20:31 2016

@author: valterf
"""
import soundfile as sf
import numpy as np
from feats import stft, istft
from config import FRAME_RATE
import resampy
import os

def prepare_features(wavpath, nnet, pred_index=0):
    """Prepare features"""
    freq = int(nnet.input.get_shape()[2])
    if isinstance(nnet.output, list):
        K = int(nnet.output[pred_index].get_shape()[2]) // freq
    else:
        K = int(nnet.output.get_shape()[2]) // freq
    sig, rate = sf.read(wavpath)
    
    if rate != FRAME_RATE:
        #raise Exception("Config specifies " + str(FRAME_RATE) +
        #                "Hz as sample rate, but file " + str(wavpath) +
        #                "is in " + str(rate) + "Hz.")
        if rate != FRAME_RATE:
                sig = resampy.resample(sig * 0.5, sr_orig=rate, sr_new=FRAME_RATE, axis=0)
                sig *= 2.0
                print("----Resample from " + str(rate) + "->" + str(FRAME_RATE) + "----")
                rate = FRAME_RATE

    try:
        (_,ch_num) = sig.shape
        if ch_num >= 2:
            sig_mc = sig
            sig = sig[:,0]
    except:
        sig = sig

    sig = sig - np.mean(sig)
    sig = sig/np.max(np.abs(sig))
    spec = stft(sig)
    mag = np.real(np.log10(spec))
    X = mag.reshape((1,) + mag.shape)
    if isinstance(nnet.output, list):
        V = nnet.predict(X)[pred_index]
    else:
        V = nnet.predict(X)

    x = X.reshape((-1, freq))
    v = V.reshape((-1, K))

    if ch_num >= 2:
        spec_mc = np.zeros((*spec.shape,ch_num),dtype='complex64')
        x_mc = np.zeros((*x.shape,ch_num))
        for ch in range(ch_num):
            sig = sig_mc[:,ch]
            sig = sig - np.mean(sig)
            sig = sig/np.max(np.abs(sig))
            spec = stft(sig)
            spec_mc[:,:,ch] = spec
            mag = np.real(np.log10(spec))
            X = mag.reshape((1,) + mag.shape)
            x = X.reshape((-1, freq))
            x_mc[:,:,ch] = x
        spec = spec_mc
        x = x_mc

    return spec, rate, x, v


def separate_sources(wavpath, nnet, num_sources, out_prefix):
    """
    Separates sources from a single-channel multiple-source input.

    wavpath is the path for the mixed input

    nnet is a loaded pre-trained model using the "load_model" function from
    predict.py

    num_sources is the expected number of sources from the input, and defines
    the number of output files

    out_prefix is the prefix of each output file, which will be writtin on the
    form {prefix}-N.wav, N in [0..num_sources-1]
    """
    k = num_sources
    freq = int(nnet.input.get_shape()[2])
    spec, rate, _, v = prepare_features(wavpath, nnet, 1)

    from sklearn.cluster import KMeans
    km = KMeans(k)
    eg = km.fit_predict(v)

    imgs = np.zeros((k, eg.size))
    for i in range(k):
        imgs[i, eg == i] = 1

    spec = np.log(spec)
    mag = np.real(spec)
    phase = np.imag(spec)
    i = 1
    for img in imgs:
        mask = img.reshape(-1, freq)
        sig_out = istft(np.exp(mag + 1j * phase) * mask)
        sig_out -= np.mean(sig_out)
        sig_out /= np.max(sig_out)
        sf.write(out_prefix + '_{}.wav'.format(i), sig_out, rate)
        i += 1


def separate_sources_dcase2019_task3(wavpath, nnet, num_sources, aug_iter):
    """
    Separates sources from a single-channel multiple-source input.

    wavpath is the path for the mixed input

    nnet is a loaded pre-trained model using the "load_model" function from
    predict.py

    num_sources is the expected number of sources from the input, and defines
    the number of output files

    Modified for preprocessing purpose of DCASE2019 task3
    1) Take first channel among four of the input
    2) cluster into two source from the first channel
    3) share estimated IBMs to separate other three channels
    4) doubled sources of 4 channels are concatenated into 8 channel wav file then save
    """
    from sklearn.cluster import KMeans
    k = num_sources
    freq = int(nnet.input.get_shape()[2])
    spec, rate, _, v = prepare_features(wavpath, nnet, 1)
    ch_num = spec.shape[-1]
    spec = np.log(spec)
    mag = np.real(spec)
    phase = np.imag(spec)
    
    for it in range(aug_iter):
        km = KMeans(k,'random',random_state=it)
        eg = km.fit_predict(v)

        imgs = np.zeros((k, eg.size))
        for i in range(k):
            imgs[i, eg == i] = 1

        i = 1
        for img in imgs:
            mask = img.reshape(-1, freq)
            for ch in range(ch_num):
                sig_out = istft(np.exp(mag[:,:,ch] + 1j * phase[:,:,ch]) * mask)
                sig_out -= np.mean(sig_out)
                sig_out /= np.max(sig_out)
                sig_out = np.reshape(sig_out, (*sig_out.shape,1))
                if i == 1:
                    sig_out_mc = sig_out
                else:
                    sig_out_mc = np.concatenate([sig_out_mc,sig_out], axis=-1)
                
                i += 1
        if it < 10:
            it = '0'+str(it)
        else:
            it = str(it)

        sf.write(wavpath[:-4] + '_i'+it+'.wav', sig_out_mc, rate)
    
    if os.path.isfile(wavpath):    
        os.remove(wavpath)
