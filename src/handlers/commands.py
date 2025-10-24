import logging

from aiogram import types, Router
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from content import messages, buttons

commands_router = Router(name=__name__)


@commands_router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    builder = InlineKeyboardBuilder()
    builder.add(buttons.compose_route_button)
    
    await message.answer(messages.GREETING_MESSAGE, reply_markup=builder.as_markup())
