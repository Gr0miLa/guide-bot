import logging
import folium
import asyncio
import os

from aiogram import F, Router, Bot
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, ReplyKeyboardRemove, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from geopy.geocoders import Nominatim

from src.ai.rag_logic import RAGSystem
from content import messages, keyboards, buttons
from src.settings.classes import UserState


handlers_router = Router(name=__name__)


def split_message(text: str, chunk_size: int = 4096) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    while len(text) > 0:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        
        split_pos = text.rfind('\n', 0, chunk_size)
        if split_pos == -1:
            split_pos = chunk_size
            
        chunks.append(text[:split_pos])
        text = text[split_pos:]
        
    return chunks


async def _generate_and_send_route(message: Message, state: FSMContext, location, bot: Bot):
    await message.answer(messages.ROUTE_GENERATION_MESSAGE, reply_markup=ReplyKeyboardRemove())
    gif_message = await message.answer_animation(
        animation=FSInputFile("media/processing.gif"),
    )
    rag_system = RAGSystem()

    user_data = await state.get_data()
    interests = user_data.get('interests')
    time = user_data.get('time')

    # Generate the route using the RAG system
    generated_route, retrieved_docs = await rag_system.generate_route_rag(interests, time, location)
    logging.info(f"Сгенерированный маршрут: \n {generated_route}")

    if retrieved_docs is not None:
        # Create a map
        m = folium.Map(location=[location.latitude, location.longitude], zoom_start=13)

        # Add markers
        for _, row in retrieved_docs.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"{row['title']}\n{row['address']}",
            ).add_to(m)

        # Save map to file
        reports_dir = "D:\\Denchek\\Work\\bots\\guide-bot\\reports"
        user_dir = os.path.join(reports_dir, str(message.from_user.username))
        os.makedirs(user_dir, exist_ok=True)
        map_path = os.path.join(user_dir, f"map_{message.from_user.id}.html")
        m.save(map_path)

        # Send map
        await message.answer("Я не могу отправить карту как геолокацию, но вы можете скачать HTML-файл и открыть его, чтобы увидеть маршрут на карте.")
        await message.answer_document(FSInputFile(map_path))

    await bot.delete_message(chat_id=message.chat.id, message_id=gif_message.message_id)
    message_chunks = split_message(generated_route)
    for i, chunk in enumerate(message_chunks):
        if i == len(message_chunks) - 1:
            builder = InlineKeyboardBuilder()
            builder.add(buttons.remake_route_button)
            await message.answer(
                chunk,
                reply_markup=builder.as_markup(),
                parse_mode=ParseMode.MARKDOWN  # или 'MarkdownV2' если нужно экранировать спецсимволы
            )
        else:
            await message.answer(
                chunk,
                parse_mode=ParseMode.MARKDOWN  # без экранирования, Telegram сам обработает
            )

    await state.clear()


@handlers_router.message(UserState.Interests)
async def process_interests(message: Message, state: FSMContext):
    await state.update_data(interests=message.text)
    await message.answer(messages.TIME_MESSAGE)
    await state.set_state(UserState.Time)


@handlers_router.message(UserState.Time)
async def process_time(message: Message, state: FSMContext):
    await state.update_data(time=message.text)
    await message.answer(messages.LOCATION_MESSAGE,
                         reply_markup=keyboards.location_keyboard)
    await state.set_state(UserState.Location)


@handlers_router.message(F.text == keyboards.manual_location_button.text)
async def manual_location_prompt(message: Message, state: FSMContext):
    await message.answer(messages.MANUAL_LOCATION_MESSAGE, reply_markup=ReplyKeyboardRemove())
    await state.set_state(UserState.ManualLocation)


@handlers_router.message(UserState.ManualLocation)
async def process_manual_location(message: Message, state: FSMContext, bot: Bot):
    geolocator = Nominatim(user_agent="guide-bot-2")

    if "нижний новгород" in message.text.lower():
        location = geolocator.geocode(message.text)
    else:
        location = geolocator.geocode("Нижний Новгород, " + message.text)
    
    logging.info(f"Локация: \b{location}")

    if location:
        await _generate_and_send_route(message, state, location, bot)
    else:
        await message.answer(messages.LOCATION_NOT_FOUND_MESSAGE)


@handlers_router.message(UserState.Location, F.location)
async def process_location(message: Message, state: FSMContext, bot: Bot):
    if message.location:
        geolocator = Nominatim(user_agent="guide-bot-2")
        location = geolocator.reverse((message.location.latitude, message.location.longitude), exactly_one=True)
        await _generate_and_send_route(message, state, location, bot)
    else:
        await message.answer(messages.GEOLOCATION_ERROR_MESSAGE)
