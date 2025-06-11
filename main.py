from ATLAS.assistant import ATLAS
import asyncio

async def main():
    atlas = ATLAS()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(atlas.stt())
        #input_message = tg.create_task(atlas.input_message())
        tg.create_task(atlas.prompt_response())
        tg.create_task(atlas.tts())

        #await input_message

if __name__ == '__main__':
    asyncio.run(main())