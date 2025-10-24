from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

location_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Отправить геолокацию", request_location=True)],
        [KeyboardButton(text="Ввести адрес вручную")]
    ],
    resize_keyboard=True,
    one_time_keyboard=True
)

manual_location_button = KeyboardButton(text="Ввести адрес вручную")