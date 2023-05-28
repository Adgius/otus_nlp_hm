import os
import aiogram 
import datetime as dt
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from wic import Wic
from qa import QA
#from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


BOT_COMMANDS = """
/****BOT_COMMANDS****/
/start - показать кнопки
"""

bot = Bot(os.getenv('API_ID') + ':' + os.getenv('API_HASH'))
dp = Dispatcher(bot, storage=MemoryStorage())

wic = Wic()
qa = QA()
def logged(func):
	async def wrapper(message: types.Message, state: FSMContext):
		with open('log.txt', 'a', encoding='utf-8') as f:
		    f.write(f'{message.from_user.id} | {message.text} | {dt.datetime.now()}\n')
		await func(message, state)
	return wrapper


async def send_message_and_log(**kwargs):
	with open('log.txt', 'a', encoding='utf-8') as f:
		f.write(f'BOT | {kwargs["text"]} | {dt.datetime.now()}\n')
	await bot.send_message(**kwargs)

### START KEYBOARD ###

btn_wic = KeyboardButton(text='Word in context')
btn_qa = KeyboardButton(text='Yes/no Question Answering')
start_keyboard = ReplyKeyboardMarkup([[btn_wic, btn_qa]])

######################
	

@dp.message_handler(commands=['start'])
@logged
async def start_command(message: types.Message, *arg):
	await send_message_and_log(chat_id=message.from_user.id, 
					   text='Выберите действие', 
					   reply_markup=start_keyboard)
	await message.delete()

##### WIC #####

class WicState(StatesGroup):
    first_sent = State()
    second_sent = State()
    target_word = State()

@dp.message_handler(regexp='Word in context')
@logged
async def get_first_sent(message: types.Message, *arg):
	await send_message_and_log(chat_id=message.from_user.id, 
					text='Введите первое предложение')
	await WicState.first_sent.set()

@dp.message_handler(state=WicState.first_sent)
@logged
async def get_second_sent(message: types.Message, state: FSMContext):
    await state.update_data(first_sent=message.text)
    await send_message_and_log(chat_id=message.from_user.id, text="Теперь введите второе предложение")
    await WicState.second_sent.set()
    
@dp.message_handler(state=WicState.second_sent)
@logged
async def get_target_word(message: types.Message, state: FSMContext):
    await state.update_data(second_sent=message.text)
    await send_message_and_log(chat_id=message.from_user.id, text="Теперь введите слово")
    await WicState.next()
    
@dp.message_handler(state=WicState.target_word)
@logged
async def wic_inference(message: types.Message, state: FSMContext):
    await state.update_data(target_word=message.text)
    await send_message_and_log(chat_id=message.from_user.id,
			       text=wic(**await state.get_data()))
    await state.finish()


#### QA #####

class QAState(StatesGroup):
    text = State()
    q = State()
    
@dp.message_handler(regexp='Yes/no Question Answering')
@logged
async def start_command(message: types.Message, *arg):
	await send_message_and_log(chat_id=message.from_user.id, 
					text='Введите текст')
	await QAState.text.set()

@dp.message_handler(state=QAState.text)
@logged
async def start_command(message: types.Message, state: FSMContext):
	await state.update_data(text=message.text)
	await send_message_and_log(chat_id=message.from_user.id, 
					text='Введите вопрос')
	await QAState.q.set()

@dp.message_handler(state=QAState.q)
@logged
async def start_command(message: types.Message, state: FSMContext):
	await state.update_data(q=message.text)
	await send_message_and_log(chat_id=message.from_user.id, 
			       text=qa(**await state.get_data()))
	await state.finish()


if __name__ == '__main__':
	executor.start_polling(dp, skip_updates=True)