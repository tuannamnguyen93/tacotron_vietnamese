# from datetime import datetime
import io
import os
# import wave
import re

import numpy as np
# import pyaudio
# import sounddevice as sd
import tensorflow as tf
from datasets import audio
from infolog import log
# from librosa import effects
from tacotron.models import create_model
# from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence

from .ssml import MySSMLParser
from .utils.cleaner_vietnamese import date, charaters, time, telephone, fraction

sample_rate = 16000
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

    def synthesize(self, texts, basenames=None, out_dir=None, log_dir=None, mel_filenames=None):
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
        self.change_voice(input, output, speed=0.88, volume=2.2)

    def emphasis_moderate(self, input, output):
        self.change_voice(input, output, speed=0.94, volume=1.8)

    def emphasis_reduce(self, input, output):
        self.change_voice(input, output, speed=1.2, volume=0.9)

    def change_speed_slow(self, input, output):
        self.change_voice(input, output, speed=0.6)

    def change_speed_medium(self, input, output):
        self.change_voice(input, output, speed=1.2)

    def change_speed_fast(self, input, output):
        self.change_voice(input, output, speed=1.8)

    def change_volume_soft(self, input, output):
        self.change_voice(input, output, volume=0.6)

    def change_volume_medium(self, input, output):
        self.change_voice(input, output, volume=1.2)

    def change_volume_loud(self, input, output):
        self.change_voice(input, output, volume=2.5)

    def change_pitch_low(self, input, output):
        self.change_voice(input, output, pitch=13000,volume=1.5)

    def change_pitch_medium(self, input, output):
        self.change_voice(input, output, pitch=16000,volume=1.5)

    def change_pitch_high(self, input, output):
        self.change_voice(input, output, pitch=17000,volume=2.0)

    def synthesize_normal(self, text_k):
        texts = re.split('\.|\-|\,|\;|\:|\<|\>|\?|\!|\*|\n', text_k)
        combined_wav = np.zeros(0)
        for idx, text_ in enumerate(texts):
            tokens = text_.split()
            lentokens = len(tokens)

            if lentokens > 0:
                chunks = [tokens[x:x + 20] for x in range(0, len(tokens), 20)]
                for chunk in chunks:
                    sub_text = ' '.join(chunk)
                    wav = self.synthesize([sub_text], None, None, None, None)
                    combined_wav = np.concatenate((combined_wav, wav))
        return combined_wav

    def synthesize_ssml(self, text):
        texts = re.split('\.|\-|\,|\;|\:|\?|\!|\*|\n', text)
        parser = MySSMLParser()
        ssml_str = text.lower()
        parser.feed(ssml_str)
        results = parser.get_data()
        print(results)

        combined_wav = np.zeros(0)
        for res in results:
            text_ = res[0]
            # print('text value:', text_)
            if "text" == res[1]:
                print("ssml type:", res[1])
                wav = self.synthesize_normal(text_)*0.75
                print('\n')
            if "emphasis" == res[1]:
                # print("ssml type:", res[1])

                if res[2]['level'] == 'strong':
                    wav = self.synthesize_normal(text_)
                    print(text_)
                    audio.save_wav(wav, "emphasis_strong.wav", sr=sample_rate)
                    self.emphasis_strong('emphasis_strong.wav', 'strongout_emphasis.wav')
                    wav = audio.load_wav('strongout_emphasis.wav', sample_rate)
                    wav = np.concatenate((wav,np.zeros(2800)))
                    os.remove('emphasis_strong.wav')
                    os.remove('strongout_emphasis.wav')
                    print('-'*100)

                elif res[2]['level'] == 'moderate':
                    wav = self.synthesize_normal(text_)
                    audio.save_wav(wav,"emphasis_moderate.wav", sr=sample_rate)
                    self.emphasis_moderate('emphasis_moderate.wav', 'moderateout_emphasis.wav')
                    wav = audio.load_wav('moderateout_emphasis.wav', sample_rate)
                    wav = np.concatenate((wav,np.zeros(2500)))
                    os.remove('emphasis_moderate.wav')
                    os.remove('moderateout_emphasis.wav')

                elif res[2]['level'] == 'reduced':
                    wav = self.synthesize_normal(text_)
                    audio.save_wav(wav,"emphasis_reduce.wav", sr=sample_rate)
                    self.emphasis_reduce('emphasis_reduce.wav', 'reducedout_emphasis.wav')
                    wav = audio.load_wav('reducedout_emphasis.wav', sample_rate)
                    wav = np.concatenate((wav,np.zeros(2000)))
                    os.remove('emphasis_reduce.wav')
                    os.remove('reducedout_emphasis.wav')

            elif "prosody" == res[1]:
                print("ssml type:", res[1])
                for key in res[2].keys():
                    if key == 'rate':
                        if res[2]['rate'] == "medium":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_speed_medium('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')

                        elif res[2]['rate'] == "slow":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_speed_medium('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')

                        elif res[2]['rate'] == "fast":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_speed_medium('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')
                    elif key == 'pitch':
                        if res[2]['pitch'] == "medium":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_pitch_medium('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')

                        elif res[2]['pitch'] == "low":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_pitch_low('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')

                        elif res[2]['pitch'] == "high":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_pitch_high('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')
                    elif key == 'volume':
                        if res[2]['volume'] == "medium":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_volume_medium('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')

                        elif res[2]['volume'] == "soft":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_volume_soft('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')

                        elif res[2]['volume'] == "loud":
                            wav = self.synthesize_normal(text_)
                            audio.save_wav(wav, "change_speed_medium.wav", sr=sample_rate)
                            self.change_volume_loud('change_speed_medium.wav', 'speed_medium.wav')
                            wav = audio.load_wav('speed_medium.wav', sample_rate)
                            # wav = np.concatenate((np.zeros(1000), wav))
                            os.remove('change_speed_medium.wav')
                            os.remove('speed_medium.wav')

            elif "break" == res[1]:
                for key in res[2].keys():
                    if key == 'strength':
                        if res[2]['strength'] == 'medium':
                            wav = np.zeros(12000)

                        if res[2]['strength'] == 'strong':
                            wav = np.zeros(20000)

                        if res[2]['strength'] == 'weak':
                            wav = np.zeros(8000)

                        if res[2]['strength'] == 'x-strong':
                            wav = np.zeros(40000)

                    if key == 'time':
                        if res[2]['time'][-2:] == 'ms':
                            time_to_break = int(res[2]['time'][:-2]) * sample_rate/1000
                            wav = np.zeros(time_to_break)

                        elif res[2]['time'][-1:] == 's':
                            time_to_break = int(res[2]['time'][:-1]) * sample_rate
                            wav = np.zeros(time_to_break)

            elif "say-as" == res[1]:
                if res[2]['interpret-as'] == 'date':
                    text = date(text_)
                    wav = self.synthesize_normal(text)

                elif res[2]['interpret-as'] == 'time':
                    text = time(text_)
                    wav = self.synthesize_normal(text)

                elif res[2]['interpret-as'] == 'telephone':
                    text = telephone(text_)
                    wav = self.synthesize_normal(text)

                elif res[2]['interpret-as'] == 'characters':
                    text = charaters(text_)
                    wav = self.synthesize_normal(text)

                elif res[2]['interpret-as'] == 'fraction':
                    text = fraction(text_)
                    wav = self.synthesize_normal(text)

            combined_wav = np.concatenate((combined_wav, wav))
        return combined_wav
        # out = io.BytesIO()
        # audio.save_wav(combined_wav, out, sr=sample_rate)
        # return out.getvalue()

