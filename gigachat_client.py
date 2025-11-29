# gigachat_client.py

from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat

# Загружаем переменные окружения
load_dotenv()

GIGACHAT_API_KEY = os.getenv("GIGACHAT_CREDENTIALS")
if not GIGACHAT_API_KEY:
    raise ValueError("❌ GIGACHAT_CREDENTIALS не найден в .env")

# Создаём один общий LLM-объект, который будем переиспользовать
langchain_llm = GigaChat(
    credentials=GIGACHAT_API_KEY,
    verify_ssl_certs=False,
    model="GigaChat",      # можешь поменять на Pro/Plus, если есть
    temperature=0.4,
    max_tokens=512,        # можно немного увеличить
)


def gigachat_chat(messages: List[Dict[str, str]]) -> str:
    """
    Простейший адаптер под формат сообщений AutoGen:
    messages: [{"role": "user" | "assistant" | "system", "content": "..."}, ...]

    Возвращает только текст ответа (str).
    """

    # Простой способ: склеиваем историю в один prompt.
    # Можно сделать умнее (учитывать роли, system отдельно и т.п.)
    prompt_lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt_lines.append(f"{role}: {content}")
    prompt = "\n".join(prompt_lines)

    # Вызываем LangChain-модель
    response = langchain_llm.predict(prompt)
    return response
