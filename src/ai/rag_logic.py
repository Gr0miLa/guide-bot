import os
import re
import faiss
import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import pytz
from typing import Tuple

from mistralai import Mistral
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, 'data.index')
DATAFRAME_PATH = os.path.join(BASE_DIR, 'data.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'cultural_objects_mnn.xlsx')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

CATEGORIES = {
    1: "Памятники и монументы",
    2: "Парки, сады и общественные пространства",
    3: "Тактильные макеты",
    4: "Набережные",
    5: "Архитектурные и исторические объекты",
    6: "Культурно-досуговые центры и кинотеатры",
    7: "Музеи и галереи",
    8: "Театры",
    10: "Советские мозаики и монументальное искусство"
}

# --- INDEXING (Run once) ---
def create_index():
    """Reads data, creates embeddings, builds a FAISS index, and saves everything."""
    if os.path.exists(INDEX_PATH) and os.path.exists(DATAFRAME_PATH):
        logging.info("Индекс и файлы данных уже существуют. Создание пропускается.")
        return

    logging.info("Создание нового индекса и файлов данных...")
    try:
        # Загружаем Excel
        logging.info("Загружаем Excel")
        df = pd.read_excel(DATA_PATH)

        # Фильтруем df, убирая category_id = 9
        logging.info("Фильтруем df, убирая category_id = 9")
        df = df[df['category_id'] != 9].copy()

        # Убираем ненужные столбцы
        if 'url' in df.columns:
            df = df.drop(columns=['url'])
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        # Парсим столбец 'coordinate'
        logging.info("Парсим столбец 'coordinate'")
        coords = df['coordinate'].str.extract(r'POINT \(([^ ]+) ([^ ]+)\)', expand=True)
        df['latitude'] = coords[1]
        df['longitude'] = coords[0]

        # Удаляем старый столбец coordinate
        logging.info("Удаляем старый столбец coordinate")
        df = df.drop(columns=['coordinate'])

        # Объединяем ВСЕ оставшиеся столбцы в одну строку текста
        logging.info("Объединяем ВСЕ оставшиеся столбцы в одну строку текста")
        df['text_for_embedding'] = df.fillna('').astype(str).agg('; '.join, axis=1)

        # Создаём SentenceTransformer
        logging.info("Создаём SentenceTransformer")
        model = SentenceTransformer(EMBEDDING_MODEL)

        # Генерируем эмбеддинги
        logging.info("Генерируем эмбеддинги")
        embeddings = model.encode(df['text_for_embedding'].tolist(), show_progress_bar=True)

        # Создаём FAISS-индекс
        logging.info("Создаём FAISS-индекс")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))

        # Сохраняем индекс и DataFrame
        logging.info("Сохраняем индекс и DataFrame")
        faiss.write_index(index, INDEX_PATH)

        if os.path.exists(INDEX_PATH):
            logging.info(f"✅ Файл индекса успешно сохранен в {INDEX_PATH}")
        else:
            logging.error(f"❌ Не удалось сохранить файл индекса в {INDEX_PATH}")
        df.to_pickle(DATAFRAME_PATH)
        if os.path.exists(DATAFRAME_PATH):
            logging.info(f"✅ Файл DataFrame успешно сохранен в {DATAFRAME_PATH}")
        else:
            logging.error(f"❌ Не удалось сохранить файл DataFrame в {DATAFRAME_PATH}")

    except FileNotFoundError as e:
        logging.error(f"❌ Файл данных не найден: {e}. Невозможно создать индекс.")
    except Exception as e:
        logging.error(f"❌ Произошла ошибка при создании индекса: {e}")

from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние в километрах между двумя точками на Земле."""
    R = 6371.0  # Радиус Земли в километрах

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

class RAGSystem:
    def __init__(self):
        self.index = None
        self.df_retrieval = None
        self.model = None
        self.client = None
        self.mistral_semaphore = asyncio.Semaphore(1)  # Limit to 1 concurrent request initially
        self._load_components()

    def _load_components(self):
        try:
            self.index = faiss.read_index(INDEX_PATH)
            self.df_retrieval = pd.read_pickle(DATAFRAME_PATH)
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                logging.error("MISTRAL_API_KEY не найден. Система RAG не сможет генерировать текст.")
            self.client = Mistral(api_key=api_key)
            logging.info("Компоненты RAGSystem успешно загружены.")
        except Exception as e:
            logging.error(f"Не удалось загрузить компоненты RAG: {e}. Система не будет работать.")

    async def _call_mistral_api(self, messages: list, model: str, retries: int = 5, initial_delay: float = 1.0):
        for attempt in range(retries):
            try:
                async with self.mistral_semaphore:
                    response = self.client.chat.complete(
                        model=model,
                        messages=messages,
                    )
                return response
            except Exception as e:
                logging.error(f"Ошибка вызова Mistral API (попытка {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    logging.info(f"Повторная попытка через {delay:.2f} секунд...")
                    await asyncio.sleep(delay)
                else:
                    raise

    async def generate_route_rag(self, interests: str, time: str, location) -> Tuple[str, pd.DataFrame]:
        if not (self.index and self.df_retrieval is not None and not self.df_retrieval.empty and self.model and self.client):
            return "Ошибка: Система поиска по объектам не инициализирована. Пожалуйста, проверьте логи сервера.", None

        # logging.info(f"Система RAG получила запрос с интересами: {interests}")

        # Create a prompt to find the best category
        category_prompt = f"""
На основе интереса пользователя "{interests}", выбери из списка все категории, которые могут быть ему интересны.

Список доступных категорий:
{CATEGORIES}

Верни только ID подходящих категорий из списка (можно несколько, если они все соответствуют интересам пользователя).
Не добавляй ничего лишнего.
"""

        # logging.info(f"Промпт для категорий: \b{category_prompt}")

        try:
            category_response = await self._call_mistral_api(
                messages=[{
                    "role": "user",
                    "content": category_prompt,
                }],
                model="mistral-large-latest",
            )
            response_text = category_response.choices[0].message.content.strip()
            best_categories_ids = [int(cat_id) for cat_id in re.findall(r'\d+', response_text)]
            # logging.info(f"Найдены лучшие категории: {best_categories_ids}")

            # Filter the DataFrame by the best categories
            filtered_df = self.df_retrieval[self.df_retrieval['category_id'].isin(best_categories_ids)]
            if filtered_df.empty:
                logging.warning(f"Объекты не найдены для категорий: {best_categories_ids}. Используются все данные.")
                filtered_df = self.df_retrieval
        except Exception as e:
            logging.error(f"Ошибка при поиске лучшей категории: {e}")
            filtered_df = self.df_retrieval
            
        query_embedding = self.model.encode([interests])
        
        # Создаём копию отфильтрованного датафрейма, чтобы избежать SettingWithCopyWarning
        filtered_df_copy = filtered_df.copy()

        # Удаляем ненужные столбцы
        filtered_df_copy.drop(columns=['category_id'], inplace=True)

        # Объединяем все оставшиеся столбцы в одну строку для эмбеддингов
        # Это делается для того, чтобы эмбеддинги были более точными, не включая уже отфильтрованную категорию
        filtered_df_copy['text_for_embedding'] = filtered_df_copy.fillna('').astype(str).agg('; '.join, axis=1)

        # Создаём FAISS-индекс для отфильтрованных данных
        filtered_embeddings = self.model.encode(filtered_df_copy['text_for_embedding'].tolist(), show_progress_bar=False)
        filtered_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        filtered_index.add(np.array(filtered_embeddings, dtype=np.float32))
        
        k = 20
        distances, indices = filtered_index.search(np.array(query_embedding, dtype=np.float32), k)
        
        retrieved_docs = filtered_df.iloc[indices[0]]

        if retrieved_docs.empty:
            return "К сожалению, я не смог найти ничего подходящего по вашим интересам. Попробуйте изменить запрос.", None

        locations_text = "\n".join([f"- {row.get('title', 'N/A')}: {row.get('address', 'N/A')}\n  Описание: {row.get('description', 'N/A')}" for _, row in retrieved_docs.iterrows()])

        # --- Расчет расстояний ---
        locations = []
        for _, row in retrieved_docs.iterrows():
            locations.append({
                "title": row.get('title', 'N/A'),
                "lat": float(row.get('latitude')),
                "lon": float(row.get('longitude'))
            })

        distance_info = "\nРасстояния:\n"
        # Расстояния от туриста до каждого места
        for i, loc in enumerate(locations):
            dist = haversine(location.latitude, location.longitude, loc["lat"], loc["lon"])
            distance_info += f"- От вашего местоположения до \"{loc['title']}\" - {dist:.2f} км.\n"

        # Расстояния между местами
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                loc1 = locations[i]
                loc2 = locations[j]
                dist = haversine(loc1["lat"], loc1["lon"], loc2["lat"], loc2["lon"])
                distance_info += f"- Между \"{loc1['title']}\" и \"{loc2['title']}\" - {dist:.2f} км.\n"

        msk_tz = pytz.timezone('Europe/Moscow')
        current_time_msk = datetime.now(msk_tz).strftime('%Y-%m-%d %H:%M:%S')

        prompt = f"""
Ты — эксперт-гид по Нижнему Новгороду. Твоя задача — создать персонализированный пеший маршрут для туриста.

Вот данные от туриста:
- Интересы: {interests}
- Доступное время: {time} часа(ов)
- Текущие координаты: {location.latitude}, {location.longitude}
- Текущий адрес: {location.address}
- Текущее время в Нижнем Новгороде: {current_time_msk}, считай его началом для прогулки

Я нашел несколько мест, которые могут быть интересны, основываясь на предпочтениях туриста:
{locations_text}

Вот информация о расстояниях между объектами и от вашего начального местоположения:
{distance_info}

Выполни следующие шаги:
1.  Выбери несколько самых подходящих мест из предложенного списка, которые можно посетить за указанное время.
2.  Составь логичный и последовательный пеший маршрут, начиная от текущего местоположения туриста, **учитывая реальные расстояния между точками** для оптимизации пути.
3.  Для каждого места в маршруте напиши краткое и увлекательное объяснение, почему его стоит посетить.
4.  Предложи примерный таймлайн прогулки, **основываясь на времени ходьбы между точками (считай среднюю скорость пешехода 4-5 км/ч) и времени на осмотр достопримечательностей**.
5.  Твой ответ должен быть дружелюбным, структурированным и легким для чтения через телеграм (поэтому не надо сложных таблиц в ответе).
"""

        # logging.info(f"Промпт: {prompt}\n")

        try:
            chat_response = await self._call_mistral_api(
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                model="mistral-large-latest",
            )
            
            final_route_text = chat_response.choices[0].message.content
            
            # Extract titles from the generated route
            final_titles = []
            for line in final_route_text.split('\n'):
                if ":" in line:
                    title = line.split(":")[0].strip()
                    # Check if the title exists in the retrieved documents
                    if title in retrieved_docs['title'].values:
                        final_titles.append(title)
            
            # Filter retrieved_docs to only include locations in the final route
            final_docs = retrieved_docs[retrieved_docs['title'].isin(final_titles)]

            return final_route_text, final_docs
        except Exception as e:
            logging.error(f"Ошибка вызова Mistral API: {e}")
            return "Произошла ошибка при обращении к AI. Пожалуйста, попробуйте позже.", None
