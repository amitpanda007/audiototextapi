import os
from typing import Union

import asyncio
import aiofiles
import whisper

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sse_starlette.sse import EventSourceResponse

from app.lib.util import transcribe_status

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


async def transcribe_wrapper(filename: str, result_folder: str):
    loop = asyncio.get_event_loop()
    # loop.create_task(transcribe_data(filename))
    text = await loop.run_in_executor(None, lambda: transcribe_data(filename, result_folder))
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
            background_tasks.add_task(transcribe_wrapper, filename, folder)

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


@app.get('/transcribe/result')
async def get_transcribe_result(param: str, request: Request):
    event_generator = transcribe_status(param, request)
    return EventSourceResponse(event_generator)
