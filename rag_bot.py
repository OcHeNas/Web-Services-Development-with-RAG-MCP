import os
from pathlib import Path
import logging
import nest_asyncio
from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ========== Настройка ==========

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

# Проверка токена
if TELEGRAM_TOKEN is None:
    raise ValueError("Ошибка: TELEGRAM_TOKEN не установлен в .env")

# Логирование
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
nest_asyncio.apply()

# Клиент LM Studio
client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")

# ========== Загрузка документов ==========

DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)

def load_documents():
    documents = []
    for file_path in DOCS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({"filename": file_path.name, "content": content})
            print(f"✓ Загружен: {file_path.name} ({len(content)} символов)")
    print(f"\nВсего загружено документов: {len(documents)}")
    return documents

documents = load_documents()

def build_context(documents):
    context_parts = [f"=== Документ: {doc['filename']} ===\n{doc['content']}" for doc in documents]
    return "\n\n".join(context_parts)

context = build_context(documents)

SYSTEM_PROMPT = f"""Ты - полезный ассистент, который отвечает на вопросы на основе предоставленных документов.

У тебя есть доступ к следующим документам:

{context}

Правила:
1. Отвечай только на основе информации из документов выше
2. Если информации нет в документах, честно скажи об этом
3. Указывай, из какого документа взята информация
4. Отвечай на том же языке, на котором задан вопрос
"""

print(f"Системный промпт создан ({len(SYSTEM_PROMPT)} символов)")

# ========== Функция запроса к LM Studio ==========

def ask_question(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при обработке запроса: {e}"

# ========== Обработчики Telegram ==========

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я RAG-бот с LM Studio.\n\n"
        "Команды:\n"
        "/start - Начало работы\n"
        "/status - Статус системы\n\n"
        "Просто напишите вопрос, и я отвечу на основе загруженных документов."
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs_count = len(documents)
    try:
        client.models.list()
        lm_status = "+ подключен"
    except:
        lm_status = "- недоступен"
    await update.message.reply_text(
        f"Статус системы:\n\n"
        f"LM Studio: {lm_status}\n"
        f"Документов загружено: {docs_count}\n"
        f"URL: {LM_STUDIO_BASE_URL}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not documents:
        await update.message.reply_text(
            "⚠ Документы не загружены.\nДобавьте .txt файлы в папку 'documents/' и перезапустите скрипт."
        )
        return
    
    question = update.message.text
    await update.message.reply_text("🤔 Думаю...")
    answer = ask_question(question)
    await update.message.reply_text(answer)

# ========== Запуск бота ==========

def run_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    run_bot()
