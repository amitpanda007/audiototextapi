import os
import asyncio
from os.path import dirname, abspath, join


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
