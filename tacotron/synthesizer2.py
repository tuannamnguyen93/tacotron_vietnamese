# from datetime import datetime
import io
import os
# import wave
import re

import ffmpy
import numpy as np
# import pyaudio
# import sounddevice as sd
import tensorflow as tf
from .ssml import MySSMLParser
from datasets import audio
from infolog import log
# from librosa import effects
from tacotron.models import create_model
# from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence


class Synthesizer:

    def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
        log('Constructing model: %s' % model_name)
        # Force the batch size to be known in order to use attention masking in batch synthesis
        inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
        input_lengths = tf.placeholder(tf.int32, (None), name='input_lengths')
        targets = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
        split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')
        with tf.variable_scope('Tacotron_model') as scope:
            self.model = create_model(model_name, hparams)
            if gta:
                self.model.initialize(inputs, input_lengths, targets, gta=gta, split_infos=split_infos)
            else:
                self.model.initialize(inputs, input_lengths, split_infos=split_infos)

            self.mel_outputs = self.model.tower_mel_outputs
            self.alignments = self.model.tower_alignments
            self.stop_token_prediction = self.model.tower_stop_token_prediction
            self.targets = targets

            if hparams.predict_linear and not gta:
                self.linear_outputs = self.model.tower_linear_outputs
                self.linear_wav_outputs = audio.inv_spectrogram_tensorflow(self.model.tower_linear_outputs[0], hparams)

        self.gta = gta
        self._hparams = hparams
        # pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        # explicitely setting the padding to a value that doesn't originally exist in the spectogram
        # to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.

        self.inputs = inputs
        self.input_lengths = input_lengths
        self.targets = targets
        self.split_infos = split_infos

        log('Loading checkpoint: %s' % checkpoint_path)
        # Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

    def _get_output_lengths(self, stop_tokens):
        # Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
        output_lengths = [row.index(1) for row in np.round(stop_tokens).tolist()]
        return output_lengths

    def synthesize(self, texts, basenames, out_dir, log_dir, mel_filenames):
        hparams = self._hparams
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

        # Repeat last sample until number of samples is dividable by the number of GPUs (last run scenario)
        while len(texts) % hparams.tacotron_synthesis_batch_size != 0:
            texts.append(texts[-1])
            basenames.append(basenames[-1])
            if mel_filenames is not None:
                mel_filenames.append(mel_filenames[-1])

        assert 0 == len(texts) % self._hparams.tacotron_num_gpus
        seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
        input_lengths = [len(seq) for seq in seqs]

        size_per_device = len(seqs) // self._hparams.tacotron_num_gpus

        # Pad inputs according to each GPU max length
        input_seqs = None
        split_infos = []
        for i in range(self._hparams.tacotron_num_gpus):
            device_input = seqs[size_per_device * i: size_per_device * (i + 1)]
            device_input, max_seq_len = self._prepare_inputs(device_input)
            input_seqs = np.concatenate((input_seqs, device_input), axis=1) if input_seqs is not None else device_input
            split_infos.append([max_seq_len, 0, 0, 0])

        feed_dict = {
            self.inputs: input_seqs,
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
        }

        if self.gta:
            np_targets = [np.load(mel_filename) for mel_filename in mel_filenames]
            target_lengths = [len(np_target) for np_target in np_targets]

            # pad targets according to each GPU max length
            target_seqs = None
            for i in range(self._hparams.tacotron_num_gpus):
                device_target = np_targets[size_per_device * i: size_per_device * (i + 1)]
                device_target, max_target_len = self._prepare_targets(device_target, self._hparams.outputs_per_step)
                target_seqs = np.concatenate((target_seqs, device_target),
                                             axis=1) if target_seqs is not None else device_target
                split_infos[i][
                    1] = max_target_len  # Not really used but setting it in case for future development maybe?

            feed_dict[self.targets] = target_seqs
            assert len(np_targets) == len(texts)

        feed_dict[self.split_infos] = np.asarray(split_infos, dtype=np.int32)

        if self.gta or not hparams.predict_linear:
            mels, alignments, stop_tokens = self.session.run(
                [self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)
            # Linearize outputs (1D arrays)
            mels = [mel for gpu_mels in mels for mel in gpu_mels]
            alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
            stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]

            if not self.gta:
                # Natural batch synthesis
                # Get Mel lengths for the entire batch from stop_tokens predictions
                target_lengths = self._get_output_lengths(stop_tokens)

            # Take off the batch wise padding
            mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
            assert len(mels) == len(texts)

        else:
            linear_wavs, linears, mels, alignments, stop_tokens = self.session.run(
                [self.linear_wav_outputs, self.linear_outputs, self.mel_outputs, self.alignments,
                 self.stop_token_prediction], feed_dict=feed_dict)
            # Linearize outputs (1D arrays)
            linear_wavs = [linear_wav for gpu_linear_wav in linear_wavs for linear_wav in gpu_linear_wav]
            linears = [linear for gpu_linear in linears for linear in gpu_linear]
            mels = [mel for gpu_mels in mels for mel in gpu_mels]
            alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
            stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]

            # Natural batch synthesis
            # Get Mel/Linear lengths for the entire batch from stop_tokens predictions
            # target_lengths = self._get_output_lengths(stop_tokens)
            target_lengths = [9999]

            # Take off the batch wise padding
            mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
            linears = [linear[:target_length, :] for linear, target_length in zip(linears, target_lengths)]
            assert len(mels) == len(linears) == len(texts)

            wav = audio.inv_preemphasis(linear_wavs, hparams.preemphasis)

        return wav

    def synthesize_singer(self, text):
        hparams = self._hparams
        wav = self.synthesize([text], None, None, None, None)
        out = io.BytesIO()
        audio.save_wav(wav, out, sr=hparams.sample_rate)
        return out.getvalue()

    def clearn_text(self, text):
        import string
        translator = str.maketrans(' ', ' ', string.punctuation)
        text = text.translate(translator)

        return text

    def change_speed(self, input_path, output_path, speed):
        ff = ffmpy.FFmpeg(inputs={input_path: None}, outputs={output_path: ["-filter:a", "atempo=" + str(speed)]})
        ff.run()

    def change_volume(self, input_path, output_path, vol):
        ff = ffmpy.FFmpeg(inputs={input_path: None}, outputs={output_path: ["-filter:a", "volume=" + vol]})
        ff.run()

    def change_pitch(self, input_path, output_path, pitch):
        ff = ffmpy.FFmpeg(inputs={input_path: None}, outputs={output_path: ["-filter:a", "asetrate=" + str(pitch)]})
        ff.run()

    def synthesize_helper(self, text_k):
        # Dang rat lom, phai sua nhieu :))
        print(text_k)
        puntuation = '\.|\-|\,|\;|\:|\<|\>|\?|\!|\*|\n'
        silen_array = []
        for tex in text_k:
            if tex in puntuation:
                silen_array.append(tex)
        silen_array.append('.')
        hparams = self._hparams
        texts = re.split('\.|\-|\,|\;|\:|\<|\>|\?|\!|\*|\n', text_k)
        print('text', texts)
        # print(len(silen_array), silen_array)
        combined_wav = np.zeros(0)
        for idx, text_ in enumerate(texts):
            text_ = self.clearn_text(text_)
            if silen_array[idx] in '.?!':
                silen = np.zeros(12000)
            elif silen_array[idx] in ',-:*':
                silen = np.zeros(5000)
            elif silen_array[idx] in '\n':
                silen = np.zeros(12000)
            else:
                silen = np.zeros(0)
            tokens = text_.split()
            lentokens = len(tokens)

            print(lentokens)
            if lentokens > 0 and lentokens < 30:
                if 'strong' and 'moderate' and 'reduce' not in text_:
                    wav = self.synthesize([text_], None, None, None, None)
                if 'strong' in text_:
                    text_ = text_.replace('strong', '')
                    wav = self.synthesize([text_], None, None, None, None)
                    wav_file = audio.save_wav(wav, 'strong.wav', 16000)
                    wav_out = self.emphasis_strong('strong.wav', 'out_strong.wav')
                    wav = audio.load_wav('out_strong.wav', 16000)
                    wav = np.concatenate((np.zeros(1000), wav))
                    os.remove('strong.wav')
                    os.remove('out_strong.wav')

                if 'moderate' in text_:
                    text_ = text_.replace('moderate', '')
                    wav = self.synthesize([text_], None, None, None, None)
                    wav_file = audio.save_wav(wav, 'vstrong.wav', 16000)
                    wav_out = self.emphasis_moderate('vstrong.wav', 'vvout_strong.wav')
                    wav = audio.load_wav('vvout_strong.wav', 16000)
                    wav = np.concatenate((np.zeros(1000), wav))
                    os.remove('vstrong.wav')
                    os.remove('vvout_strong.wav')

                if 'reduce' in text_:
                    text_ = text_.replace('reduce', '')
                    wav = self.synthesize([text_], None, None, None, None)
                    wav_file = audio.save_wav(wav, 'pstrong.wav', 16000)
                    wav_out = self.emphasis_reduce('pstrong.wav', 'pout_strong.wav')
                    wav = audio.load_wav('pout_strong.wav', 16000)
                    wav = np.concatenate((np.zeros(1000), wav))
                    os.remove('pstrong.wav')
                    os.remove('pout_strong.wav')

                print(wav.shape)
                # print(idx)

                combined_wav = np.concatenate((combined_wav, wav))
            else:
                chunks = [tokens[x:x + 20] for x in range(0, len(tokens), 20)]
                for index, chunk in enumerate(chunks):
                    sub_text = ' '.join(chunk)
                    wav_sub_text = self.synthesize([sub_text], None, None, None, None)

                    combined_wav = np.concatenate((combined_wav, wav_sub_text))

            if len(texts) > 1:
                combined_wav = np.concatenate((combined_wav, silen))

        out = io.BytesIO()
        audio.save_wav(combined_wav, out, sr=hparams.sample_rate)
        return out.getvalue()

    def change_voice(self, input_file_path, output_file_path, pitch=16000, speed=1.0, volume=1.0):
        import subprocess
        command = ['ffmpeg',
                   '-i', '"%s"' % input_file_path,
                   '-y',
                   "-filter:a",
                   '"asetrate= % s, atempo = %s, volume = %s"' % (str(pitch), str(speed), str(volume)),
                   '"%s"' % output_file_path]
        subprocess.check_call(' '.join(command), shell=True)

    def emphasis_strong(self, input, output):
        self.change_voice(input, output, speed=0.75, volume=2.5, pitch=15700)

    def emphasis_moderate(self, input, output):
        self.change_voice(input, output, speed=0.85, volume=1.3)

    def emphasis_reduce(self, input, output):
        self.change_voice(input, output, speed=1.5, volume=0.95, pitch=16700)


def sysnthesizer_normal(text_k):
    puntuation = '\.|\-|\,|\;|\:|\<|\>|\?|\!|\*|\n'
    silen_array = []
    for tex in text_k:
        if tex in puntuation:
            silen_array.append(tex)
    silen_array.append('.')
    texts = re.split('\.|\-|\,|\;|\:|\<|\>|\?|\!|\*|\n', text_k)
    print('text', texts)
    # print(len(silen_array), silen_array)
    combined_wav = np.zeros(0)
    for idx, text_ in enumerate(texts):
        text_ = Synthesizer.clearn_text(text_)
        # if silen_array[idx] in '.?!':
        #     silen = np.zeros(12000)
        # elif silen_array[idx] in ',-:*':
        #     silen = np.zeros(5000)
        # elif silen_array[idx] in '\n':
        #     silen = np.zeros(12000)
        # else:
        #     silen = np.zeros(0)
        tokens = text_.split()
        lentokens = len(tokens)

        if lentokens > 0 and lentokens < 30:
            # print('lentokens > 0 and lentokens <30')
            wav = Synthesizer.synthesize([text_], None, None, None, None)
            if idx == 0:
                combined_wav = wav
            else:
                combined_wav = np.concatenate((combined_wav, wav))
        else:
            chunks = [tokens[x:x + 20] for x in range(0, len(tokens), 20)]
            for index, chunk in enumerate(chunks):
                sub_text = ' '.join(chunk)
                wav = Synthesizer.synthesize([sub_text], None, None, None, None)
                if index == 0:
                    combined_wav = wav
                else:
                    combined_wav = np.concatenate((combined_wav, wav))


def sysnthesizer_say_as(text, interpret_as='spell-out'):
    if interpret_as == 'characters':
        pass
    elif interpret_as == 'number':
        pass
    elif interpret_as == 'fraction':
        pass
    elif interpret_as == 'unit':
        pass
    elif interpret_as == 'date':
        pass
    elif interpret_as == 'time':
        pass
    elif interpret_as == 'telephone':
        pass
    elif interpret_as == 'address':
        pass

def sysnthesizer_break(text, strength='none', time='0s'):
    # strength is one of ['none', 'x-weak', 'weak', 'medium', 'strong', 'x-strong']
    pass

def sysnthesizer_emphasis(text, level='strong'):
    # level is one of ['strong', 'moderate', 'reduced']
    parser = MySSMLParser()
    ssml_str = text.lower()
    parser.feed(ssml_str)
    results = parser.get_data()
    combined_wav = np.zeros(0)
    for res in results:
        text_ = res[0]
        if "emphasis" == res[1]:
            if res[2]['level'] == 'strong':
                wav = sysnthesizer_normal(text_)
                audio.save_wav(wav, path="emphasis.wav", sr=16000)
                wav_out = Synthesizer.emphasis_strong('emphasis.wav', 'out_emphasis.wav')
                wav = audio.load_wav('out_emphasis.wav', 16000)
                wav = np.concatenate((np.zeros(1000), wav))
                os.remove('emphasis.wav.wav')
                os.remove('out_emphasis.wav')

            elif res[2]['level'] == 'moderate':
                wav = sysnthesizer_normal(text_)
                audio.save_wav(wav, path="emphasis.wav", sr=16000)
                wav_out = Synthesizer.emphasis_moderate('emphasis.wav', 'out_emphasis.wav')
                wav = audio.load_wav('out_emphasis.wav', 16000)
                wav = np.concatenate((np.zeros(1000), wav))
                os.remove('emphasis.wav.wav')
                os.remove('out_emphasis.wav')

            elif res[2]['level'] == 'reduced':
                wav = sysnthesizer_normal(text_)
                audio.save_wav(wav, path="emphasis.wav", sr=16000)
                wav_out = Synthesizer.emphasis_moderate('emphasis.wav', 'out_emphasis.wav')
                wav = audio.load_wav('out_emphasis.wav', 16000)
                wav = np.concatenate((np.zeros(1000), wav))
                os.remove('emphasis.wav.wav')
                os.remove('out_emphasis.wav')

            combined_wav = np.concatenate((combined_wav, wav))
            out = io.BytesIO()
            audio.save_wav(combined_wav, out, sr=hparams.sample_rate)
    return out.getvalue()

    # if "prosody" = res[1]:
    #         if res[2]['pitch']=='16000' and res[2]['rate'] == 'moderate':

    #         wav = sysnthesizer_normal(text_)
    #         audio.save_wav(wav, path="emphasis.wav", sr=16000)
    #         wav_out = Synthesizer.emphasis_strong('emphasis.wav', 'out_emphasis.wav')
    #         wav = audio.load_wav('out_strong.wav', 16000)
    #         wav = np.concatenate((np.zeros(1000),wav))
    #         os.remove('emphasis.wav.wav')
    #         os.remove('out_emphasis.wav')

def sysnthesizer_prosody(text, rate='medium', pitch='medium', volume='medium'):
    # rate : [x-slow, slow, medium, fast, x-fast]
    # pitch : [x-low, low, medium, high, x-high]
    # volume : [silent, x-soft, soft, medium, loud, x-loud]
    pass

MY_SYNTHESIZER_MAPER = {
    'text': sysnthesizer_normal,
    'say-as': sysnthesizer_say_as,
    'break': sysnthesizer_break,
    'emphasis': sysnthesizer_emphasis,
    'prosody': sysnthesizer_prosody
}
