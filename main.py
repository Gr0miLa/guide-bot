import asyncio
import logging
import sys

from dotenv import load_dotenv
from src.handlers.callbacks import callback_router
from src.settings.loader import load_dispatcher, load_bot
from src.handlers.commands import commands_router
from src.handlers.handlers import handlers_router
from src.ai.rag_logic import create_index


async def main() -> None:
    logging.info("Bot started")
    load_dotenv()
    create_index()
    dp = load_dispatcher()
    bot = load_bot()
    dp.include_router(commands_router)
    dp.include_router(handlers_router)
    dp.include_router(callback_router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
