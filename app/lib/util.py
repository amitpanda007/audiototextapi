import io
import os
import asyncio
import wave
import audioop
from queue import Queue
from os.path import dirname, abspath, join
from threading import Thread

import numpy
import pyaudio
import whisper

from app.lib import constants

'''
Get status as an event generator
'''
status_stream_delay = 5  # second
status_stream_retry_timeout = 30000  # milisecond


async def compute_status(param1):
    return "DONE"


async def status_event_generator(request, param1):
    previous_status = None
    while True:
        if await request.is_disconnected():
            print('Request disconnected')
            break

        if previous_status and previous_status == "DONE":
            print('Request completed. Disconnecting now')
            yield {
                "event": "end",
                "data": ''
            }
            break

        current_status = await compute_status(param1)
        if previous_status != current_status:
            yield {
                "event": "update",
                "retry": status_stream_retry_timeout,
                "data": current_status
            }
            previous_status = current_status
            print('Current status :', current_status)
        else:
            print('No change in status...')

        await asyncio.sleep(status_stream_delay)


async def transcribe_status(param, request):
    folder_name = param
    previous_status = None
    current_status = None
    transcribe_data = ""
    while True:
        if await request.is_disconnected():
            print('Request disconnected')
            break

        if previous_status and previous_status == "DONE":
            print('Request completed. Disconnecting now')
            yield {
                "event": "end",
                "data": previous_status
            }
            break

        # Read folder from provided folder_name to check if transcribed data exist
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if os.path.isdir(f'{dir_path}/../../upload/{folder_name}'):
            dirs = os.listdir(f'{dir_path}/../../upload/{folder_name}')
            if len(dirs) == 0:
                print("Directory is empty")
                current_status = None
            else:
                print("Directory is not empty")
                file = dirs[0]
                with open(f'{dir_path}/../../upload/{folder_name}/{file}') as f:
                    file_content = f.read()

                yield {
                    "event": "update",
                    "retry": status_stream_retry_timeout,
                    "data": file_content
                }
                current_status = "DONE"
        else:
            print(f"Directory /upload/{folder_name} doesn't exist")

        if previous_status != current_status:
            previous_status = current_status
            print('Current status :', current_status)
        else:
            print('No change in status...')

        await asyncio.sleep(status_stream_delay)


async def language_detect_status(param, request):
    folder_name = param
    previous_status = None
    current_status = None
    transcribe_data = ""
    while True:
        if await request.is_disconnected():
            print('Request disconnected')
            break

        if previous_status and previous_status == "DONE":
            print('Request completed. Disconnecting now')
            yield {
                "event": "end",
                "data": previous_status
            }
            break

        # Read folder from provided folder_name to check if transcribed data exist
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if os.path.isdir(f'{dir_path}/../../upload-recording/{folder_name}'):
            dirs = os.listdir(f'{dir_path}/../../upload-recording/{folder_name}')
            if len(dirs) == 0:
                print("Directory is empty")
                current_status = None
            else:
                print("Directory is not empty")
                file = dirs[0]
                with open(f'{dir_path}/../../upload-recording/{folder_name}/{file}') as f:
                    file_content = f.read()

                yield {
                    "event": "update",
                    "retry": status_stream_retry_timeout,
                    "data": file_content
                }
                current_status = "DONE"
        else:
            print(f"Directory /upload-recording/{folder_name} doesn't exist")

        if previous_status != current_status:
            previous_status = current_status
            print('Current status :', current_status)
        else:
            print('No change in status...')

        await asyncio.sleep(status_stream_delay)


def get_lang(code: str):
    langs = constants.LANGUAGES
    try:
        return langs[code]
    except KeyError:
        return ""


transcription_file = "transcription.txt"
max_energy = 5000
sample_rate = 16000
chunk_size = 1024
max_int16 = 2 ** 15
max_record_time = 30
last_sample = bytes()
record_thread: Thread = None
currently_transcribing = False
data_queue = Queue()
audio_model:whisper.Whisper = None
pa = pyaudio.PyAudio()
run_record_thread = True


def recording_thread(stream: pyaudio.Stream):
    while run_record_thread:
        # We record as fast as possible so that we can update the volume bar at a fast rate.
        data = stream.read(chunk_size)
        # energy = audioop.rms(data, pa.get_sample_size(pyaudio.paInt16))
        data_queue.put(data)


def live_transcribe():
    global currently_transcribing, audio_model, record_thread, run_record_thread
    microphones = {}
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0 and device_info['hostApi'] == 0:
            microphones[device_info['index']] = device_info['name']

    default_mic = pa.get_default_input_device_info()['index']
    print(f"MIC: {default_mic}")
    selected_mic = int(default_mic)
    audio_model = whisper.load_model('medium')
    if not currently_transcribing:
        if not record_thread:
            stream = pa.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=sample_rate,
                             input=True,
                             frames_per_buffer=chunk_size,
                             input_device_index=selected_mic)
            record_thread = Thread(target=recording_thread, args=[stream])
            run_record_thread = True
            record_thread.start()

        last_sample = bytes()
        currently_transcribing = True
    else:
        # Stop the record thread.
        if record_thread:
            run_record_thread = False
            record_thread.join()
            record_thread = None

        # Drain all the remaining data but save the last sample.
        # This is to pump the main loop one more time, otherwise we'll end up editing
        # the last line when we start transcribing again, rather than creating a new line.
        data = None
        while not data_queue.empty():
            data = data_queue.get()
        if data:
            data_queue.put(data)

        # Save transcription.

        # with open(transcription_file, 'w+') as f:
        #     f.writelines('\n'.join([item.value for item in transcription_list.controls]))
        currently_transcribing = False

    while True:
        if audio_model and not data_queue.empty():

            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data

            # Write out raw frames as a wave file.
            wav_file = io.BytesIO()
            wav_writer: wave.Wave_write = wave.open(wav_file, "wb")
            wav_writer.setframerate(sample_rate)
            wav_writer.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wav_writer.setnchannels(1)
            wav_writer.writeframes(last_sample)
            wav_writer.close()

            # Read the audio data, now with wave headers.
            wav_file.seek(0)
            wav_reader: wave.Wave_read = wave.open(wav_file)
            samples = wav_reader.getnframes()
            audio = wav_reader.readframes(samples)
            wav_reader.close()

            # Convert the wave data straight to a numpy array for the model.
            # https://stackoverflow.com/a/62298670
            audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
            audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)
            audio_normalised = audio_as_np_float32 / max_int16

            print(audio_normalised)

            language = 'en'
            task = 'transcribe'

            result = audio_model.transcribe(audio_normalised, language=language, task=task)
            text = result['text'].strip()
            print(f"Transcribed Text: {text}")

            # If we've reached our max recording time, it's time to break up the buffer, add an empty line after we
            # edited the last line.
            audio_length_in_seconds = samples / float(sample_rate)
            if audio_length_in_seconds > max_record_time:
                last_sample = bytes()


if __name__ == '__main__':
    live_transcribe()