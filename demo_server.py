import falcon
import tensorflow as tf
import os
import time
from hparams import hparams
from infolog import log
from tacotron.synthesizer import Synthesizer
from wsgiref import simple_server
import argparse

use_gpu = False

if use_gpu:
    import keras.backend as K
    from keras.backend.tensorflow_backend import set_session

    config = K.tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    set_session(K.tf.Session(config=config))
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

html_body = '''<html><title>Demo</title>
<style>
*, *:before, *:after {
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  box-sizing: border-box;
}

html {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 100%;
  background: #333;
}

#page-wrapper {
  width: 640px;
  background: #FFFFFF;
  padding: 1em;
  margin: 1em auto;
  border-top: 5px solid #69c773;
  box-shadow: 0 2px 10px rgba(0,0,0,0.8);
}

h1 {
  margin-top: 0;
}

#msg {
  font-size: 0.9em;
  line-height: 1.4em;
}

#msg.not-supported strong {
  color: #CC0000;
}

# input[type="text"] {
#   width: 100%;
#   padding: 0.5em;
#   font-size: 1.2em;
#   border-radius: 3px;
#   border: 1px solid #D9D9D9;
#   box-shadow: 0 2px 3px rgba(0,0,0,0.1) inset;
# }

input[type=text], select, textarea {
    width: 100%;
    padding: 12px;
    border: 1px solid #ccc;
    border-radius: 4px;
    resize: both;
}

input[type="range"] {
  width: 300px;
}

label {
  display: inline-block;
  float: left;
  width: 150px;
}

.option {
  margin: 1em 0;
}

button {
  display: inline-block;
  border-radius: 3px;
  border: none;
  font-size: 0.9rem;
  padding: 0.5rem 0.8em;
  background: #69c773;
  border-bottom: 1px solid #498b50;
  color: white;
  -webkit-font-smoothing: antialiased;
  font-weight: bold;
  margin: 0;
  width: 100%;
  text-align: center;
}

button:hover, button:focus {
  opacity: 0.75;
  cursor: pointer;
}

button:active {
  opacity: 1;
  box-shadow: 0 -3px 10px rgba(0, 0, 0, 0.1) inset;
}

</style>
<body>
<!-- <form>
  <input id="text" type="text" size="40" placeholder="Enter Text">
  <button id="button" name="synthesize">Speak</button>
</form> -->

  <meta charset="UTF-8">
  <title>Demo</title>


  <div id="page-wrapper">
    <h1>Text to Speech Synthesis Demo</h1>
    
    <p id="msg"></p>

    <textarea id="text" name="subject" placeholder="Write something.." style="height:200px"></textarea>

    <div class="option">
      <label for="volume">Volume</label>
      <input type="range" min="0" max="1" step="0.1" name="volume" id="volume" value="0.5">
    </div>
    <div class="option">
      <label for="rate">Rate</label>
      <input type="range" min="0.3" max="3" step="0.1" name="rate" id="rate" value="1">
    </div>
    <div class="option">
      <label for="pitch">Pitch</label>
      <input type="range" min="0" max="2" step="0.1" name="pitch" id="pitch" value="1">
    </div>

  <button id="button" name="synthesize">Speak</button>

  </div>
<!--   
  

    <script  src="js/index.js"></script> -->

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
    synthesize(text);
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
            


<script>
var x = document.getElementById("audio");

function getPlaySpeed() { 
    alert(x.playbackRate);
} 


var speedlist = document.getElementById("rate");
speedlist.addEventListener("change",changeSpeed);
function changeSpeed(event){
    x.playbackRate = event.target.value;
}

var vid = document.getElementById("volume");
function getVolume() { 
    alert(x.volume);
} 


vid.addEventListener("change",changevolume);
function changevolume(event){
    x.volume = event.target.value;
    }
    
</script> 
        
</script>
</body>
</html>
'''

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', \
                    default='logs-Tacotron/taco_pretrained/tacotron_model.ckpt-365000', \
                    help='Full path to model checkpoint')
parser.add_argument('--hparams', default='',
                    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
parser.add_argument('--port', default=9001, help='Port of Http service')
parser.add_argument('--host', default="0.0.0.0", help='Host of Http service')
parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
args = parser.parse_args()
synth = Synthesizer()
modified_hp = hparams.parse(args.hparams)
synth.load(args.checkpoint, modified_hp)


class Res:
    def on_get(self, req, res):
        res.body = html_body
        res.content_type = "text/html"


class Syn:
    def on_get(self, req, res):
        text = req.params.get('text')
        if not text:
            raise falcon.HTTPBadRequest()
        start = time.time()
        print("Start timer.")
        print(text)
        res.data = synth.synthesize_ssml(text)
        end = time.time()
        print('Finish timer. %f seconds.'%(end-start))
        res.content_type = "audio/wav"


api = falcon.API()
api.add_route("/", Res())
api.add_route("/synthesize", Syn())
print("host:{},port:{}".format(args.host, int(args.port)))
simple_server.make_server(args.host, int(args.port), api).serve_forever()
