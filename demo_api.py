from __future__ import print_function
import os
import json
from future.standard_library import install_aliases
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from tacotron.synthesizer import Synthesizer
from datasets  import audio
import string
import random
import time

path = 'logs-Tacotron/taco_pretrained/tacotron_model.ckpt-333000'
synth = Synthesizer()
from hparams import hparams
synth.load(path, hparams=hparams)

use_gpu = True
if use_gpu:
    import keras.backend as K
    from keras.backend.tensorflow_backend import set_session
    config = K.tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(K.tf.Session(config=config))
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

install_aliases()
app = Flask(__name__)
cors = CORS(app)
@app.route('/synthesis', methods=['POST'])
@cross_origin()

def train():
    req = request.get_json(silent=True, force=True)
    res = processSynthesis(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


def processSynthesis(req):
    sentence = req["sentence"]
    wav = synth.synthesize_ssml(sentence)
    path = 'tmp/audio_'+str(time.time()) + '.wav'
    print(path)
    audio.save_wav(wav, path=path, sr=16000)
    return {'response': 'Synthesis done!', 'audio': path}

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port,host = '0.0.0.0')





