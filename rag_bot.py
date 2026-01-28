import os
from pathlib import Path
import logging
import nest_asyncio
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ================== НАСТРОЙКА ==================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Ошибка: TELEGRAM_TOKEN не установлен в .env")

LM_STUDIO_CHAT_URL = "http://127.0.0.1:1234/api/v1/chat"
MODEL_NAME = "mistralai/mistral-7b-instruct-v0.3"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

nest_asyncio.apply()

# ================== ЗАГРУЗКА ДОКУМЕНТОВ ==================

DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)


def load_documents():
    docs = []
    for file_path in DOCS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(
                {
                    "filename": file_path.name,
                    "content": content,
                }
            )
            print(f"✓ Загружен: {file_path.name} ({len(content)} символов)")
    print(f"\nВсего загружено документов: {len(docs)}")
    return docs


documents = load_documents()


def build_context(docs):
    return "\n\n".join(
        f"=== Документ: {doc['filename']} ===\n{doc['content']}"
        for doc in docs
    )


DOCUMENTS_CONTEXT = build_context(documents)

SYSTEM_PROMPT = f"""Ты — полезный ассистент, который отвечает на вопросы строго на основе предоставленных документов.

У тебя есть доступ к следующим документам:

{DOCUMENTS_CONTEXT}

Правила:
1. Отвечай только на основе информации из документов выше
2. Если информации нет в документах, честно скажи об этом
3. Указывай, из какого документа взята информация
4. Отвечай на том же языке, на котором задан вопрос
"""

print(f"Системный промпт создан ({len(SYSTEM_PROMPT)} символов)")

# ================== ЗАПРОС К LM STUDIO (/api/v1/chat) ==================

def ask_question(question: str) -> str:
    """
    Отправка запроса к LM Studio natively через /api/v1/chat
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "temperature": 0.7,
            "max_output_tokens": 1000,
        }

        response = requests.post(LM_STUDIO_CHAT_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # LM Studio chat API возвращает ответ в ['response']
        if "response" in data:
            return data["response"]
        else:
            return "⚠ Не удалось получить ответ от модели."

    except Exception as e:
        logger.exception("Ошибка при обращении к LM Studio")
        return f"❌ Ошибка при обработке запроса: {e}"


# ================== TELEGRAM HANDLERS ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! 👋 Я RAG-бот с локальной LLM (LM Studio).\n\n"
        "Команды:\n"
        "/start — начало работы\n"
        "/status — статус системы\n\n"
        "Просто напишите вопрос — я отвечу на основе загруженных документов."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs_count = len(documents)
    try:
        requests.post(
            LM_STUDIO_CHAT_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "ping"}],
                "max_output_tokens": 1,
            },
            timeout=10,
        )
        lm_status = "✅ подключен"
    except Exception as e:
        lm_status = f"❌ недоступен ({e})"

    await update.message.reply_text(
        "📊 Статус системы:\n\n"
        f"LM Studio: {lm_status}\n"
        f"Документов загружено: {docs_count}\n"
        f"Модель: {MODEL_NAME}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not documents:
        await update.message.reply_text(
            "⚠ Документы не загружены.\n"
            "Добавьте .txt файлы в папку 'documents/' и перезапустите бота."
        )
        return

    question = update.message.text
    await update.message.reply_text("🤔 Думаю...")
    answer = ask_question(question)
    await update.message.reply_text(answer)


# ================== ЗАПУСК БОТА ==================

def run_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()







