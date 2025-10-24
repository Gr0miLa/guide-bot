from aiogram.types import Message, BufferedInputFile
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


async def send_photo(message: Message, file: BufferedInputFile):
    await message.answer_photo(
        photo=file
    )


async def send_sticker(message: Message, file: BufferedInputFile):
    await message.answer_document(
        document=file
    )


async def send_document(message: Message, file: BufferedInputFile):
    await message.answer_document(
        document=file
    )


def extract_username(text: str) -> str:
    text = text.strip()

    prefixes = [
        'https://t.me/',
        'http://t.me/',
        't.me/',
        '@'
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    username = text.split('/')[0].split('?')[0].split('#')[0]

    if username and not username.startswith('@'):
        username = f'@{username}'

    return username if username else '@'
