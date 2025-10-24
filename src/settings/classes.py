from aiogram.fsm.state import StatesGroup, State
from enum import Enum


class UserState(StatesGroup):
    Interests = State()
    Time = State()
    Location = State()
    ManualLocation = State()


class FileType(Enum):
    DOCUMENT = "document"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    STICKER = "sticker"
