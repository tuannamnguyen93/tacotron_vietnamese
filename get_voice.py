# -*- coding:utf8 -*-
# !/usr/bin/env python
from __future__ import print_function
from tacotron.synthesizer import Synthesizer
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from hparams import hparams

# import keras.backend as K
# from keras.backend.tensorflow_backend import set_session
# config = K.tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.45
# set_session(K.tf.Session(config=config))

wav = 'wav'
txt = 'txt'

with open('FINAL.txt') as f:
    files = f.readlines()

with open('log.txt') as f:
    index = int(f.read())

checkpoint = 'logs-Tacotron/taco_pretrained/tacotron_model.ckpt-304000'
synth = Synthesizer()
synth.load(checkpoint,hparams)

for i in range(index, len(files)):
    path = 'wav/' + str(i) + '.wav'
    print(i)
    with open(path, 'wb') as f:
        f.write(synth.synthesize_singer(files[i]))

    print(os.path.getsize(path))
    if os.path.getsize(path) <= 100:
        os.remove(path)
        continue
    with open('txt/' + str(i) + '.txt', 'w') as f:
        f.write(files[i])
    with open('log.txt', 'w') as f:
        f.write(str(i))
