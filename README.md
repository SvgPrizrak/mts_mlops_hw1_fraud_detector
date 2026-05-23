# ML Fraud Detection Service

## Описание проекта

Сервис подготовлен в рамках домашнего задания №1 по MLOps.

Проект реализует batch inference service для ML-модели из соревнования блока ML 2025 года:

```text
https://www.kaggle.com/t/c9670cbd31c645839a4fca29e68c2bc2
```

Сервис принимает файл `test.csv` через монтированную директорию `./input`, выполняет preprocessing, применяет предобученную CatBoost-модель и сохраняет результаты в монтированную директорию `./output`.

Контейнер выполняет только inference на CPU. Обучение модели внутри контейнера не выполняется.

---

## Что делает сервис

После запуска контейнера сервис:

1. Загружает входной файл `test.csv` из директории `./input`
2. Выполняет preprocessing данных
3. Применяет предобученную ML-модель
4. Сохраняет результат в формате `sample_submission.csv` в директорию `./output`
5. Дополнительно сохраняет:
   - `feature_importances_top5.json`
   - `prediction_score_density.png`

---

## Архитектура проекта

```text
.
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── app/
│   └── app.py
├── models/
│   └── my_catboost.cbm
├── src/
│   ├── loader.py
│   ├── preprocessing.py
│   ├── scorer.py
│   ├── exporter.py
│   └── reports.py
├── train_data/
│   └── .gitkeep
├── input/
│   └── .gitkeep
└── output/
    └── .gitkeep
```

---

## Описание основных файлов

### `app/app.py`

Основной файл сервиса.

Отвечает за:

- инициализацию сервиса;
- загрузку reference train dataset;
- загрузку модели;
- отслеживание директории `/app/input`;
- запуск полного inference pipeline;
- сохранение результатов в `/app/output`.

---

### `src/loader.py`

Отдельный скрипт для загрузки входного файла.

Выполняет этап:

```text
Загрузка входного файла
```

---

### `src/preprocessing.py`

Скрипт с preprocessing pipeline.

Выполняет этап:

```text
Препроцессинг данных
```

В preprocessing используются преобразования, реализованные для модели из соревнования:

- обработка временных признаков;
- обработка географических признаков;
- обработка категориальных признаков;
- mean encoding;
- обработка числовых признаков;
- подготовка итогового датасета для модели.

---

### `src/scorer.py`

Скрипт для скоринга обработанного датасета.

Выполняет этап:

```text
Скоринг обработанного датасета
```

Файл отвечает за:

- загрузку CatBoost-модели;
- получение predicted scores через `predict_proba`;
- получение бинарных предсказаний;
- получение top-5 feature importances.

---

### `src/exporter.py`

Скрипт для сохранения результатов.

Выполняет этап:

```text
Выгрузка результирующего файла
```

Сохраняет:

```text
output/sample_submission.csv
output/feature_importances_top5.json
```

---

### `src/reports.py`

Скрипт для сохранения дополнительных отчётов.

Сохраняет график распределения predicted scores:

```text
output/prediction_score_density.png
```

---

## Модель

В проекте используется предобученная CatBoost-модель:

```text
models/my_catboost.cbm
```

Модель загружается внутри контейнера и используется только для inference.

Обучение модели внутри контейнера не выполняется.

---

## Требования

Для запуска проекта нужен Docker.

Проверить наличие Docker можно командой:

```bash
docker --version
```

Также можно проверить, что Docker daemon запущен:

```bash
docker ps
```

Если Docker не запущен, необходимо открыть Docker Desktop и дождаться запуска Docker Engine.

---

## Подготовка данных

Скачайте данные соревнования:

```text
https://www.kaggle.com/t/c9670cbd31c645839a4fca29e68c2bc2
```

Перед сборкой Docker image положите файл `train.csv` в директорию:

```text
train_data/train.csv
```

Файл `train.csv` используется как reference dataset для preprocessing. Модель на нём внутри контейнера не обучается.

Файл для скоринга положите в директорию:

```text
input/test.csv
```

Итоговая локальная структура перед сборкой должна быть такой:

```text
.
├── train_data/
│   └── train.csv
├── input/
│   └── test.csv
```

Важно: CSV-файлы не хранятся в GitHub-репозитории. Они игнорируются через `.gitignore`.

---

## Сборка Docker image

Из корня проекта выполните:

```bash
docker build -t mts25-mlops-hw1 .
```

Первичная сборка может занять несколько минут, так как Docker установит Python-зависимости из `requirements.txt`.

---

## Запуск контейнера

### Windows PowerShell

```powershell
docker run --rm -it `
  -v "${PWD}/input:/app/input" `
  -v "${PWD}/output:/app/output" `
  mts25-mlops-hw1
```

### macOS / Linux

```bash
docker run --rm -it \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  mts25-mlops-hw1
```

Флаг `--rm` означает, что контейнер будет автоматически удалён после остановки.

---

## Проверка работы сервиса

Если файл `input/test.csv` уже лежит в директории до запуска контейнера, сервис обработает его автоматически.

Также можно запустить контейнер и после запуска добавить файл `test.csv` в директорию:

```text
input/
```

После успешной обработки в директории `output/` появятся файлы:

```text
sample_submission.csv
feature_importances_top5.json
prediction_score_density.png
```

---

## Описание выходных файлов

### `sample_submission.csv`

Файл с предсказаниями модели в формате Kaggle `sample_submission.csv`.

---

### `feature_importances_top5.json`

JSON-файл с top-5 feature importances используемой модели.

Формат:

```json
{
  "feature_name": 12.34
}
```

Где:

- `key` — название признака;
- `value` — значение feature importance.

---

### `prediction_score_density.png`

PNG-файл с графиком плотности распределения predicted scores по загруженному датасету.

График строится по вероятностям модели, полученным через `predict_proba`.

---

## Логирование

Сервис пишет логи:

1. В консоль контейнера
2. В файл внутри контейнера:

```text
/app/logs/service.log
```

В логах отображаются основные этапы работы:

- запуск сервиса;
- загрузка train dataset;
- загрузка модели;
- обнаружение входного файла;
- preprocessing;
- scoring;
- сохранение результатов;
- ошибки при обработке.

---

## Возможные проблемы

### Не найден `train.csv`

Проверьте, что файл лежит здесь:

```text
train_data/train.csv
```

Важно: файл `train.csv` должен находиться в папке `train_data` до сборки Docker image.

После добавления файла пересоберите image:

```bash
docker build -t mts25-mlops-hw1 .
```

---

### Не найдена модель

Проверьте, что файл модели лежит здесь:

```text
models/my_catboost.cbm
```

---

### Docker не видит `input` или `output`

Проверьте, что контейнер запущен с volume mount:

```powershell
-v "${PWD}/input:/app/input"
-v "${PWD}/output:/app/output"
```

Для macOS / Linux:

```bash
-v "$(pwd)/input:/app/input"
-v "$(pwd)/output:/app/output"
```

---

### В папке `output` ничего не появилось

Проверьте:

1. Файл `test.csv` лежит в папке `input`
2. Файл имеет расширение `.csv`
3. Контейнер запущен из корня проекта
4. В логах нет ошибок
5. В команде запуска указаны volume mount для `input` и `output`

---

### Не совпадает формат `sample_submission.csv`

Откройте файл `sample_submission.csv` из Kaggle и проверьте названия колонок.

Если формат отличается, измените функцию `save_submission` в файле:

```text
src/exporter.py
```

Итоговый файл должен иметь тот же набор колонок, что и Kaggle `sample_submission.csv`.

---

### Docker daemon не запущен

Если появляется ошибка вида:

```text
failed to connect to the docker API
```

или:

```text
Cannot connect to the Docker daemon
```

проверьте, что Docker Desktop запущен.

После запуска Docker Desktop проверьте:

```bash
docker ps
```

---

## Очистка после тестирования

Если контейнер запускался с флагом `--rm`, отдельное удаление контейнера не требуется: Docker удалит его автоматически после остановки.

Остановить запущенный контейнер можно сочетанием клавиш:

```text
Ctrl + C
```

Если контейнер был запущен без `--rm`, посмотреть список контейнеров можно командой:

```bash
docker ps -a
```

Удалить контейнер:

```bash
docker rm <container_id_or_name>
```

Удалить Docker image проекта:

```bash
docker rmi mts25-mlops-hw1
```

Очистить входные и выходные директории:

### Windows PowerShell

```powershell
Remove-Item .\input\* -Force
Remove-Item .\output\* -Force
```

### macOS / Linux

```bash
rm -rf input/*
rm -rf output/*
```

Если нужно удалить неиспользуемые Docker-объекты:

```bash
docker system prune
```

Если нужно полностью удалить неиспользуемые контейнеры, сети, образы и build cache:

```bash
docker system prune -a
```

Важно: команда `docker system prune -a` может удалить неиспользуемые образы других проектов.

---

## Проверка перед отправкой в GitHub

Перед отправкой проекта проверьте статус Git:

```bash
git status
```

В репозиторий не должны попасть:

```text
train.csv
test.csv
*.csv
```

В репозиторий должны попасть:

```text
Dockerfile
README.md
requirements.txt
app/
src/
models/my_catboost.cbm
train_data/.gitkeep
input/.gitkeep
output/.gitkeep
```

Если CSV-файл случайно попал в индекс Git, удалите его из индекса:

```bash
git rm --cached train_data/train.csv
git rm --cached input/test.csv
```

---

## Итоговый результат

После успешного запуска сервис создаёт:

```text
output/sample_submission.csv
output/feature_importances_top5.json
output/prediction_score_density.png
```

Таким образом, проект удовлетворяет требованиям домашнего задания:

- Docker image собирается;
- контейнер стабильно запускается;
- сервис принимает `test.csv`;
- сервис выдаёт `sample_submission.csv`;
- preprocessing и scoring реализованы отдельными скриптами;
- модель действительно применяется для подготовки результата;
- дополнительно сохраняются top-5 feature importances и график распределения predicted scores.

Внимание: сервис по умолчанию не хранит `sample_submission.csv`, `feature_importances_top5.json` и `prediction_score_density.png` в GitHub. Эти файлы получаются только после запуска сервиса. При необходимости результаты можно приложить отдельно в домашнюю работу в личном кабинете МТС.