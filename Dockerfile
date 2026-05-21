FROM python:3.12-slim

WORKDIR /app

# Создаём директорию для логов
RUN mkdir -p /app/logs && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект внутрь контейнера
COPY . .

# Создаём input/output директории внутри контейнера
RUN mkdir -p /app/input /app/output

# Директории, которые будут монтироваться снаружи
VOLUME /app/input
VOLUME /app/output

# Запуск сервиса
CMD ["python", "./app/app.py"]