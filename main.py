import logging
import time

from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.utils import executor
from sklearn.preprocessing import StandardScaler

import re
from pymorphy3 import MorphAnalyzer
import pandas as pd
import joblib
import numpy as np
from Levenshtein import distance as lev

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Токен бота
API_TOKEN = '7347680636:AAFIZk4hX1uO49VUVykPOvB_b12voV90ZtI'

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Список регионов
subjects_of_russia = [
    "Республика Адыгея", "Республика Алтай", "Республика Башкортостан", "Республика Бурятия",
    "Республика Дагестан", "Республика Ингушетия", "Кабардино-Балкарская Республика", "Республика Калмыкия",
    "Карачаево-Черкесская Республика", "Республика Карелия", "Республика Коми", "Республика Крым",
    "Республика Марий Эл",
    "Республика Мордовия", "Республика Саха (Якутия)", "Республика Северная Осетия — Алания", "Республика Татарстан",
    "Республика Тыва", "Удмуртская Республика", "Республика Хакасия", "Чеченская Республика", "Чувашская Республика",
    "Алтайский край", "Забайкальский край", "Камчатский край", "Краснодарский край", "Красноярский край",
    "Пермский край",
    "Приморский край", "Ставропольский край", "Хабаровский край", "Амурская область", "Архангельская область",
    "Астраханская область",
    "Белгородская область", "Брянская область", "Владимирская область", "Волгоградская область", "Вологодская область",
    "Воронежская область",
    "Ивановская область", "Иркутская область", "Калининградская область", "Калужская область", "Кемеровская область",
    "Кировская область",
    "Костромская область", "Курганская область", "Курская область", "Ленинградская область", "Липецкая область",
    "Магаданская область",
    "Московская область", "Мурманская область", "Нижегородская область",
    "Новгородская область", "Новосибирская область", "Омская область", "Оренбургская область", "Орловская область",
    "Пензенская область", "Псковская область", "Ростовская область", "Рязанская область", "Самарская область",
    "Саратовская область",
    "Сахалинская область", "Свердловская область", "Смоленская область", "Тамбовская область", "Тверская область",
    "Томская область", "Тульская область", "Тюменская область", "Ульяновская область", "Челябинская область",
    "Забайкальский край", "Ярославская область", "Москва", "Санкт-Петербург", "Севастополь",
    "Еврейская автономная область",
    "Ненецкий автономный округ", "Ханты-Мансийский автономный округ — Югра", "Чукотский автономный округ",
    "Ямало-Ненецкий автономный округ"
]
# Инициализация лемматизаторов
m = MorphAnalyzer()
regex = re.compile("[А-я]+")
mystopwords = stopwords.words('russian')

# Загрузка датасета
df = pd.read_csv('RSHB_data_1k.csv')

# Загрузка предобученных моделей
vectorizer = joblib.load('vectorizer.pickle')  # TfidfVectorizer(max_features=250,ngram_range = (1, 3))


# Определение состояний
class DeliveryState(StatesGroup):
    waiting_for_region = State()
    waiting_for_choice = State()
    waiting_for_square = State()
    waiting_for_floors = State()
    waiting_for_bathrooms = State()
    waiting_for_bedrooms = State()
    waiting_for_price = State()
    waiting_for_description = State()
    waiting_for_recommendation = State()


# Команда /start
@dp.message_handler(commands='start', state='*')
async def start_command(message: types.Message):
    await message.answer(f"Здравствуйте, {message.from_user.full_name}! Введите желаемый регион постройки дома.")
    await DeliveryState.waiting_for_region.set()


# Обработчик команды /help
@dp.message_handler(commands='help', state='*')
async def help_command(message: types.Message):
    help_text = (
        "Это бот для подбора проекта дома сернвиса Свое Село.\n\n"
        "Команды:\n"
        "/start - начать выбор дома\n"
        "/help - получить справку по использованию бота\n"
        "Связь с разработчиком - @Daniil_Berezin"
    )
    await message.answer(help_text)


# Обработка ввода региона
@dp.message_handler(state=DeliveryState.waiting_for_region)
async def process_region(message: types.Message, state: FSMContext):
    region = message.text
    # Предполагаемая проверка региона
    variants, flag = chek_region_input(region)
    print(variants)
    if flag == False:
        # Создание инлайн-клавиатуры с вариантами
        keyboard = InlineKeyboardMarkup()
        for variant in variants:
            keyboard.add(InlineKeyboardButton(text=variant, callback_data=f"region_{variant}"))
        keyboard.add(InlineKeyboardButton(text="Моего варианта нет", callback_data="region_none"))

        # Отправка сообщения с инлайн-клавиатурой
        await message.answer("Выберите ваш регион из предложенных вариантов:", reply_markup=keyboard)
        await DeliveryState.waiting_for_choice.set()
    else:
        selected_region = variants
        await state.update_data(region=selected_region)
        await message.answer(f"Вы выбрали: {selected_region}. Теперь введите площадь вашего участка (в м²).")
        await DeliveryState.waiting_for_square.set()


# Обработка выбора региона через инлайн-кнопки
@dp.callback_query_handler(state=DeliveryState.waiting_for_choice)
async def process_choice(callback_query: types.CallbackQuery, state: FSMContext):
    choice = callback_query.data

    if choice == "region_none":
        await bot.send_message(callback_query.from_user.id, "Попробуйте еще раз. Введите ваш регион.\nПриемер: Курск -> Курская область ")

        await DeliveryState.waiting_for_region.set()
    else:
        selected_region = choice.replace("region_", "")
        await state.update_data(region=selected_region)
        await bot.send_message(callback_query.from_user.id,
                               f"Вы выбрали: {selected_region}. Теперь введите желаемую площадь дома (в м²).")
        await DeliveryState.waiting_for_square.set()


# Обработка ввода площади дома
@dp.message_handler(state=DeliveryState.waiting_for_square)
async def process_square(message: types.Message, state: FSMContext):
    square = message.text
    # Валидация и обработка площади дома
    try:
        square_value = float(square)
        await state.update_data(square=square_value)
        await message.answer(f"Вы указали площадь: {square_value} м². Спасибо!")
        await message.answer("Пожалуйста, введите желаемое число этажей в доме")
        await DeliveryState.waiting_for_floors.set()
    except ValueError:
        await message.answer("Пожалуйста, введите корректное значение площади (число).")


# Обработка ввода числа этажей
@dp.message_handler(state=DeliveryState.waiting_for_floors)
async def process_floors(message: types.Message, state: FSMContext):
    floors = message.text
    try:
        floors_value = int(floors)
        await state.update_data(floors=floors_value)
        await message.answer(f"Вы указали количество этажей: {floors_value}. Сколько ванных комнат вы хотите?")
        await DeliveryState.waiting_for_bathrooms.set()
    except ValueError:
        await message.answer("Пожалуйста, введите корректное число этажей.")


# Обработка ввода числа ванных комнат
@dp.message_handler(state=DeliveryState.waiting_for_bathrooms)
async def process_bathrooms(message: types.Message, state: FSMContext):
    bathrooms = message.text
    try:
        bathrooms_value = int(bathrooms)
        await state.update_data(bathrooms=bathrooms_value)
        await message.answer(f"Вы указали количество ванных комнат: {bathrooms_value}. Сколько спален вы хотите?")
        await DeliveryState.waiting_for_bedrooms.set()
    except ValueError:
        await message.answer("Пожалуйста, введите корректное число ванных комнат.")


# Обработка ввода числа спален
@dp.message_handler(state=DeliveryState.waiting_for_bedrooms)
async def process_bedrooms(message: types.Message, state: FSMContext):
    bedrooms = message.text
    try:
        bedrooms_value = int(bedrooms)
        await state.update_data(bedrooms=bedrooms_value)
        await message.answer(f"Вы указали количество спален: {bedrooms_value}. Какова ваша ожидаемая цена?")
        await DeliveryState.waiting_for_price.set()
    except ValueError:
        await message.answer("Пожалуйста, введите корректное число спален.")


# Обработка ввода ожидаемой цены
@dp.message_handler(state=DeliveryState.waiting_for_price)
async def process_price(message: types.Message, state: FSMContext):
    price = message.text
    try:
        price_value = float(price)
        await state.update_data(price=price_value)
        await message.answer(
            f"Вы указали ожидаемую цену: {price_value}. Пожалуйста, опишите желаемый проект вашего дома.")
        await DeliveryState.waiting_for_description.set()
    except ValueError:
        await message.answer("Пожалуйста, введите корректное значение цены.")


# Обработка ввода описания проекта дома
@dp.message_handler(state=DeliveryState.waiting_for_description)
async def process_description(message: types.Message, state: FSMContext):
    description = message.text
    await state.update_data(description=description)
    await message.answer(
        f"Спасибо! Ваш проект дома описан следующим образом: {description}.")
    # Переход к состоянию для рекомендаций
    await message.answer("На основании ваших данных мы можем порекомендовать следующие варианты...")
    # Здесь можно добавить логику для предоставления конкретных рекомендаций

    user_data = await state.get_data()

    new_arr = np.array([
        user_data.get('floors'),  # "floors"
        user_data.get('square'),  # "square"
        user_data.get('bedrooms'),  # "bedrooms"
        user_data.get('bathrooms'),  # "bathrooms"
        user_data.get('price')  # "prices	"
    ])
    knn_df = df[df[user_data.get('region')] == 1]
    knn_df.reset_index(inplace=True)
    drop_col = ['service_titles', 'partners', 'regions', 'text_lemmas', 'index', 'descriptions',
                'links', 'image_link'] + subjects_of_russia
    num_df = knn_df.drop(columns=drop_col)

    num_df_5 = np.array(num_df[['floors', 'spaces', 'bedrooms', 'bathrooms', 'prices']])
    num_df = np.array(num_df.drop(columns=['floors', 'spaces', 'bedrooms', 'bathrooms', 'prices']))

    scaler = StandardScaler()
    num_df_5 = scaler.fit_transform(num_df_5)
    num_df = np.hstack([num_df_5, num_df])

    lemmas = clean_text(user_data.get('description'))
    embed = vectorizer.transform([lemmas]).toarray()
    new_arr = scaler.transform(new_arr.reshape(1, -1))
    new_arr = np.concatenate((new_arr[0], embed[0]), axis=0)

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='cosine').fit(num_df)

    distances, indices = nbrs.kneighbors(new_arr.reshape(1, -1))

    nearest_ids = knn_df.iloc[indices[0]]
    print(nearest_ids)
    for i in range(2):
        region = user_data.get('region')
        square = int(list(nearest_ids['spaces'])[i])
        floors = int(list(nearest_ids['floors'])[i])
        bathrooms = int(list(nearest_ids['bathrooms'])[i])
        bedrooms = int(list(nearest_ids['bedrooms'])[i])
        price = int(list(nearest_ids['prices'])[i])
        description = list(nearest_ids['descriptions'])[i]
        link = list(nearest_ids['links'])[i]
        recommendation_text = f'''
🔎 *Вариант №:*  {i + 1}\n\n   
🌍 *Регион:*  {region}\n 
📐 *Площадь дома:*  {square} м²\n
🏢 *Количество этажей:*  {floors}\n
🚿 *Количество ванных комнат:*  {bathrooms}\n
🛌 *Количество спален:*  {bedrooms}\n
💰 *Ожидаемая цена:*  {price} ₽\n
📝 *Описание проекта:*  {description}\n
🔗 *Ссылка на проект:*\n
🌐 [Перейти по ссылке]({link})
        '''
        print(1)
        image_link = list(nearest_ids['image_link'])[i]
        print(2)
        print(image_link)
        time.sleep(15)
        await message.answer_photo(photo=image_link, caption=recommendation_text, parse_mode="Markdown")
        # await message.answer(recommendation_text, parse_mode="Markdown")

    region = user_data.get('region')
    square = int(list(nearest_ids['spaces'])[2])
    floors = int(list(nearest_ids['floors'])[2])
    bathrooms = int(list(nearest_ids['bathrooms'])[2])
    bedrooms = int(list(nearest_ids['bedrooms'])[2])
    price = int(list(nearest_ids['prices'])[2])
    description = list(nearest_ids['descriptions'])[2]
    link = list(nearest_ids['links'])[2]
    recommendation_text = f'''
🔎 *Вариант №:*  {3}\n\n 
🌍 *Регион:*  {region}\n 
📐 *Площадь дома:*  {square} м²\n
🏢 *Количество этажей:*  {floors}\n
🚿 *Количество ванных комнат:*  {bathrooms}\n
🛌 *Количество спален:*  {bedrooms}\n
💰 *Ожидаемая цена:*  {price} ₽\n
📝 *Описание проекта:*  {description}\n
🔗 *Ссылка на проект:*\n
🌐 [Перейти по ссылке]({link})
    '''
    # Клавиатура с кнопкой "Заново"
    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton(text="Заново", callback_data="restart"))
    image_link = list(nearest_ids['image_link'])[2]
    await message.answer_photo(photo=image_link, caption=recommendation_text, reply_markup=keyboard,
                               parse_mode="Markdown")
    # await message.answer(recommendation_text, reply_markup=keyboard, parse_mode="Markdown")
    await DeliveryState.waiting_for_recommendation.set()


# Обработка кнопки "Заново"
@dp.callback_query_handler(lambda c: c.data == 'restart', state=DeliveryState.waiting_for_recommendation)
async def process_restart(callback_query: types.CallbackQuery, state: FSMContext):
    await state.finish()  # Завершаем текущий контекст
    await bot.send_message(callback_query.from_user.id, "Давайте начнем заново. Введите ваш регион доставки.")
    await DeliveryState.waiting_for_region.set()


# Функция проверки региона
def chek_region_input(text):
    if text in subjects_of_russia:
        return text, True
    else:
        ans = {}
        for subjects in subjects_of_russia:
            ans[subjects] = lev(text.lower(), subjects.lower())
        sort = dict(sorted(ans.items(), key=lambda item: item[1]))
        return list(sort.keys())[:3], False


def words_only(text, regex=regex):
    try:
        return regex.findall(text.lower())
    except:
        return []


def lemmatize_word(token, pymorphy=m):
    return pymorphy.parse(token)[0].normal_form


def lemmatize_text(text):
    return [lemmatize_word(w) for w in text]


def remove_stopwords(lemmas, stopwords=mystopwords):
    return [w for w in lemmas if not w in stopwords and len(w) > 3]


def clean_text(text):
    tokens = words_only(text)
    lemmas = lemmatize_text(tokens)
    return ' '.join(remove_stopwords(lemmas))


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
