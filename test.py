import asyncio
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
async def main():
    async with aiosqlite.connect("checkpoints.db") as conn:
         saver = AsyncSqliteSaver(conn)
         config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
         checkpoint = {"ts": "2023-05-03T10:00:00Z", "data": {"key": "value"}, "id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}
         saved_config = await saver.aput(config, checkpoint, {}, {})
         print(saved_config)
asyncio.run(main())