from aiogram import Router, F
from aiogram.types import CallbackQuery
from aiogram.fsm.context import FSMContext

from content import messages
from src.handlers.handlers import UserState

callback_router = Router()


@callback_router.callback_query(F.data == "compose_route")
async def compose_route_callback(callback_query: CallbackQuery, state: FSMContext):
    await callback_query.answer()
    await callback_query.message.answer(messages.INTERESTS_MESSAGE)
    await state.set_state(UserState.Interests)


@callback_router.callback_query(F.data == "remake_route")
async def remake_route_callback(callback_query: CallbackQuery, state: FSMContext):
    await callback_query.answer()
    await callback_query.message.edit_reply_markup(reply_markup=None)
    await callback_query.message.answer(messages.REMAKE_ROUTE_MESSAGE)
    await state.set_state(UserState.Interests)
