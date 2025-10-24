from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, InlineKeyboardMarkup, \
    InlineKeyboardButton


async def show_keyboard(message, text_message, keyboard, order=None, parse_mode='Markdown'):
    keyboard_layout = []
    index = 0
    for row_size in order:
        row = []
        for _ in range(row_size):
            if index < len(keyboard):
                row.append(KeyboardButton(text=keyboard[index]))
                index += 1
        keyboard_layout.append(row)
    reply_keyboard = ReplyKeyboardMarkup(keyboard=keyboard_layout, resize_keyboard=True)
    await message.answer(text_message, reply_markup=reply_keyboard, parse_mode=parse_mode)


def get_inline_keyboard(buttons: list[list[dict[str, str]]]) -> InlineKeyboardMarkup:
    keyboard_buttons = []

    for row in buttons:
        row_buttons = []
        for button in row:
            row_buttons.append(
                InlineKeyboardButton(
                    text=button["text"],
                    callback_data=button["callback_data"]
                )
            )
        keyboard_buttons.append(row_buttons)

    return InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)


async def hide_keyboard(message, text_message):
    await message.answer(text_message, reply_markup=ReplyKeyboardRemove())
