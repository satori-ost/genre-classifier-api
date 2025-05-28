# Dockerfile (покладіть його в сам корінь репозиторію)
FROM python:3.10-slim

# Якщо потрібні build-залежності, додавайте сюди, наприклад:
# RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

# Копіюємо тільки requirements.txt і ставимо пакети
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо решту коду
COPY . .

# Виставляємо порт та команду запуску
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
