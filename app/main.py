import os
import time
from typing import Union

import asyncio
import aiofiles
import uvicorn
import whisper

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sse_starlette.sse import EventSourceResponse

from app.lib.util import transcribe_status, language_detect_status, get_lang

app = FastAPI()

CHUNK_SIZE = 4096 * 4096

origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


def transcribe_data(filename: str, result_folder: str):
    # Start Transcribe with Whisper
    model = whisper.load_model('medium')

    # To run only on CPU set fp16=False. eg. model.transcribe(f'./input/{filename}', fp16=False)
    out = model.transcribe(f'./upload/{filename}', fp16=False)
    transcribed_text = out["text"]
    if not os.path.exists(f'upload/{result_folder}'):
        os.makedirs(f'upload/{result_folder}')
        name = filename.rsplit(".", 1)[0]
        with open(f'upload/{result_folder}/{name}.txt', 'w') as out_file:
            out_file.write(transcribed_text)
    return transcribed_text


def detect_language(filename: str, result_folder: str):
    model = whisper.load_model("medium")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(f'./upload-recording/{filename}')
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    probable_lang = max(probs, key=probs.get)

    probable_lang_long = get_lang(probable_lang)
    print(f"Detected language: {probable_lang_long}")

    if not os.path.exists(f'upload-recording/{result_folder}'):
        os.makedirs(f'upload-recording/{result_folder}')
        name = filename.rsplit(".", 1)[0]
        with open(f'upload-recording/{result_folder}/{name}.txt', 'w') as out_file:
            out_file.write(probable_lang_long)
    return probable_lang_long


async def transcribe_wrapper(filename: str, result_folder: str, operation_type: str):
    loop = asyncio.get_event_loop()
    # loop.create_task(transcribe_data(filename))
    if operation_type == "transcribe":
        text = await loop.run_in_executor(None, lambda: transcribe_data(filename, result_folder))
        print(text)
    elif operation_type == "language":
        text = await loop.run_in_executor(None, lambda: detect_language(filename, result_folder))
        print(text)


@app.post("/transcribe-upload/")
async def transcribe_upload_files(request: Request, file: UploadFile, background_tasks: BackgroundTasks):
    form = await request.form()
    folder = form["folder"]
    file_mask = ["mp3", "mp4", "mkv", "m4a", "wav", "flac"]
    filename = file.filename
    file_ext = filename.rsplit(".", 1)[1]
    if file_ext.lower() in file_mask:
        try:
            if not os.path.exists('./upload'):
                os.makedirs('upload')
            async with aiofiles.open(f'upload/{file.filename}', 'wb') as out_file:
                print("Starting file upload")
                content = await file.read(CHUNK_SIZE)
                while content:  # async read chunk
                    await out_file.write(content)  # async write chunk
                    content = await file.read(CHUNK_SIZE)

            print("File upload done. starting transcription.")
            background_tasks.add_task(transcribe_wrapper, filename, folder, "transcribe")

            transcribed_text = ""
            return JSONResponse(
                status_code=200,
                content={"message": f"Transcription Started.", "transcription": transcribed_text},
            )
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"message": f"Oops! Something went wrong..."},
            )
    else:
        return JSONResponse(
            status_code=403,
            content={"message": f"Unsupported filetype uploaded. Please upload {file_mask} files."},
        )


@app.post("/transcribe-audio-blob/")
async def transcribe_audio_blob(request: Request, file: UploadFile, background_tasks: BackgroundTasks):
    cur_epoch_time = int(time.time())
    filename = file.filename + f'_{cur_epoch_time}' + ".wav"
    form = await request.form()
    folder = form["folder"]
    try:
        if not os.path.exists('./upload-recording'):
            os.makedirs('upload-recording')
        async with aiofiles.open(f'upload-recording/{filename}', 'wb') as out_file:
            print("Starting file upload")
            content = await file.read(CHUNK_SIZE)
            while content:  # async read chunk
                await out_file.write(content)  # async write chunk
                content = await file.read(CHUNK_SIZE)

        print("File upload done. starting transcription.")
        background_tasks.add_task(transcribe_wrapper, filename, folder, "language")

        return JSONResponse(
            status_code=200,
            content={"message": f"Audio recording received."},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"message": f"Oops! Something went wrong..."},
        )


@app.get('/transcribe/result')
async def get_transcribe_result(param: str, request: Request):
    event_generator = transcribe_status(param, request)
    return EventSourceResponse(event_generator)


@app.get('/language-detect/result')
async def get_transcribe_result(param: str, request: Request):
    event_generator = language_detect_status(param, request)
    return EventSourceResponse(event_generator)