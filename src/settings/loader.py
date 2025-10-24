from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

import os


def load_dispatcher() -> Dispatcher:
    return Dispatcher(storage=MemoryStorage())


def load_bot() -> Bot:
    load_dotenv()
    return Bot(token=os.getenv("BOT_TOKEN"))
