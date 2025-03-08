FROM python:3.11-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Создание рабочей директории
WORKDIR /app

# Копируем файлы проекта
COPY pyproject.toml requirements.txt ./

# Установка зависимостей через uv
RUN uv venv .venv && uv pip install -r requirements.txt

# Копируем код приложения
COPY . .

# Открываем порты для ChromaDB и Streamlit
EXPOSE 8501 8000

# Запуск приложения
CMD ["uv", "pip", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
