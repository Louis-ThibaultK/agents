# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
import requests 
from livekit import rtc
from livekit.agents import tts, utils
from .log import logger

from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile
from io import BytesIO
import numpy as np

class TTS(tts.TTS):
    def __init__(
        self,
        *,
        seed: int = 42,
        style_type: str = "中文女",
        base_url: str = "http://10.218.127.100:3000/instruct/synthesize",
        prompt: str = "A girl speaker with a brisk pitch, an enthusiastic speaking pace, and a upbeat emotional demeanor.",
        sample_rate: int = 48000,
    ) -> None:
        """
        Create a new instance of Google TTS.

        Credentials must be provided, either by using the ``credentials_info`` dict, or reading
        from the file specified in ``credentials_file`` or the ``GOOGLE_APPLICATION_CREDENTIALS``
        environmental variable.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._seed, self._style_type, self._base_url, self._prompt = seed, style_type, base_url, prompt


    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(text, self._seed, self._style_type, self.sample_rate, self._prompt, self._base_url)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, seed: int, style_type: str, rate: int, prompt:str, url: str
    ) -> None:
        super().__init__()
        self._text, self._seed, self._style_type, self._sample_rate, self._prompt, self._url = text, seed, style_type, rate, prompt, url

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        
        payload = {
            'text': self._text,
            'seed': self._seed,
            'style_type': self._style_type,
            'prompt':self._prompt,
            'format': 48000
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = requests.post(self._url, headers=headers, data=payload)
        data = response.content
        if response.headers['Content-Type'] == "audio/mp3":
            decoder = utils.codecs.Mp3StreamDecoder()
            for frame in decoder.decode_chunk(data):
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id, segment_id=segment_id, frame=frame
                    )
                )
        else:
            with open('/Users/qihoo/livekit/agents/examples/voice-assistant/response.wav', 'wb') as file:
                file.write(response.content)
            data = self.resample_audio(data, 20500, self._sample_rate)
            with open('/Users/qihoo/livekit/agents/examples/voice-assistant/resample.wav', 'wb') as file:
                file.write(data)
            data = data[44:]  # skip WAV header

            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id,
                    segment_id=segment_id,
                    frame=rtc.AudioFrame(
                        data=data,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                        samples_per_channel=len(data) // 2,  # 16-bit
                    ),
                )
            )
    def resample_audio(self, audio_bytes, original_sample_rate, target_sample_rate):
        # audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        # resampled_audio = audio.set_frame_rate(target_sample_rate)
        # resampled_audio = AudioSegment.from_raw(
        #     io.BytesIO(audio_bytes),
        #     sample_width=2,
        #     frame_rate=original_sample_rate,
        #     channels=1,
        # ).set_frame_rate(target_sample_rate)
        # return resampled_audio.raw_data
        with BytesIO(audio_bytes) as wav_file:
            _, audio_data = wavfile.read(wav_file)
        num_samples = int(len(audio_data) * target_sample_rate / original_sample_rate)
        audio_resampled = signal.resample(audio_data, num_samples)
        # 将重采样后的音频数据转换为 16-bit PCM
        # 将 float32 数据缩放到 int16 的范围 [-32768, 32767]
        audio_resampled_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
        # 保存重采样后的音频数据
        with BytesIO() as output:
            wavfile.write(output, target_sample_rate, audio_resampled_int16)
            resampled_data = output.getvalue()
        return resampled_data
