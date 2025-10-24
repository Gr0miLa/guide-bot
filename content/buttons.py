from aiogram.types import InlineKeyboardButton

compose_route_button = InlineKeyboardButton(text="Составить маршрут", callback_data="compose_route")
remake_route_button = InlineKeyboardButton(text="Составить новый маршрут", callback_data="remake_route")