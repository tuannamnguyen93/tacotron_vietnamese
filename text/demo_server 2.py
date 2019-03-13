import falcon
import tensorflow as tf
import os
from hparams import hparams
from infolog import log
from tacotron.synthesizer import Synthesizer
from wsgiref import simple_server
import argparse
from flask import Flask, request, send_file
from flask.views import MethodView
import argparse
import os
from tacotron.synthesizer import Synthesizer
from flask_cors import CORS
import io
app = Flask(__name__)
CORS(app)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

html_body = '''<html><title>Demo</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="Enter Text">
  <button id="button" name="synthesize">Speak</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Synthesizing...'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}


</script>

<select id="speedlist">
  <option value="1">Bình Thường</option>
  <option value="0.3">Rất Chậm</option>
  <option value=".5">Chậm</option>
  <option value="1.5">Nhanh</option>
  <option value="2">Rất Nhanh</option>
</select>

<script>
var x = document.getElementById("audio");

function getPlaySpeed() { 
    alert(x.playbackRate);
} 


var speedlist = document.getElementById("speedlist");
speedlist.addEventListener("change",changeSpeed);
function changeSpeed(event){
    x.playbackRate = event.target.value;
}

</script> 
</body></html>
'''

synth = Synthesizer()
class Mimic2(MethodView):
    def get(self):
        text = request.args.get('text')

        wav = synth.synthesize_helper(text)
        audio = io.BytesIO(wav)
        return send_file(audio, mimetype="audio/wav")



class UI(MethodView):
    def get(self):
        return html_body


ui_view = UI.as_view('ui_view')
app.add_url_rule('/', view_func=ui_view, methods=['GET'])

mimic2_api = Mimic2.as_view('mimic2_api')
app.add_url_rule('/synthesize', view_func=mimic2_api, methods=['GET'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                          default='logs-Tacotron/taco_pretrained/tacotron_model.ckpt-500000', help='Full path to model checkpoint')
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--ip', type=str, default='0.0.0.0')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument(
        '--gpu_assignment', default='0',
        help='Set the gpu the model should run on')

    args = parser.parse_args()



    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_assignment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    synth = Synthesizer()
    modified_hp = hparams.parse(args.hparams)
    synth.load(args.checkpoint, modified_hp)
    app.run(host=args.ip, port=args.port)

# class Syn:
# 	def on_get(self,req,res):
# 		if not req.params.get('text'):
# 			raise falcon.HTTPBadRequest()
# 		res.data = synth.synthesize([req.params.get('text')], None, None, None, None)
# 		res.content_type = "audio/wav"		
		






