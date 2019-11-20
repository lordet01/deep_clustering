import numpy as np
import os
import soundfile as sf
import resampy
import webrtcvad
from vadwav import frame_generator, vad_collector, write_wave

def main(spk_idx):

    if spk_idx == 'spk1':
        DB_PATH = './dataset/2mix_tr/male'
        OUT_PATH = './dataset/2mix_tr/male_deepc'
        LIST_PATH = 'train/list_spk1.txt'
        FS = 16000
    elif spk_idx == 'spk2':
        DB_PATH = './dataset/2mix_tr/female'
        OUT_PATH = './dataset/2mix_tr/female_deepc'
        LIST_PATH = 'train/list_spk2.txt'
        FS = 16000
    
    #Initialize
    if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)
    if os.path.isfile(LIST_PATH):    
        os.remove(LIST_PATH)
    if os.path.isfile('train/list.txt'):    
        os.remove('train/list.txt')

    #Load audio files from DB path
    flist = []
    for root, dirs, files in os.walk(DB_PATH, topdown=False):
        for fname in files:
            flist = flist + [str(fname)]

    file_cnt = 0
    for fname in flist:
        ext = os.path.splitext(fname)[-1]
        
        #Check whether the file has ov1 tag
        if ext == '.wav': #and fname.find('_ov1_') > 0:
            file_path = os.path.join(DB_PATH, fname)

            #Resample file to 16k and channel to 1
            (x, fs_x) = sf.read(file_path, dtype='int16') 
            try:
                (_,ch_num) = x.shape
                if ch_num >= 2:
                    x = x[:,0]
            except:
                x = x
            print("----Loading " + file_path + ", fs: " + str(fs_x) + "----")
            #---- Resample wav files ----
            if fs_x != FS:
                x = resampy.resample(x * 0.5, sr_orig=fs_x, sr_new=FS)
                x *= 2.0
                print("----Resample from " + str(fs_x) + "->" + str(FS) + "----")

            #Cut off silences using VAD
            vad = webrtcvad.Vad(1)
            frames = frame_generator(30, x.astype('int16'), FS)
            frames = list(frames)
            fout_path = os.path.join(OUT_PATH, fname)
            fout_path = fout_path.replace("//","/")
            f = sf.SoundFile(fout_path, 'w', FS, 1, format='WAV')
            for frame in frames:
                if vad.is_speech(frame.bytes,FS): #when VAD is True
                    frame_buff = np.frombuffer(frame.bytes,dtype='int16')
                    f.write(frame_buff)
            f.close


            #Open write file for list.txt
            ftxt = open(LIST_PATH,'a')
            #Write otuput waves to the txt list
            list_line = fout_path+' '+spk_idx+'\n'
            ftxt.writelines(list_line)
    
    ftxt.close

if __name__ == "__main__":
    main(spk_idx='spk1') #from foa
    main(spk_idx='spk2') #from mic
    #Merge two speaker's list into one text
    filenames = ['train/list_spk1.txt', 'train/list_spk2.txt']
    with open('train/list.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    