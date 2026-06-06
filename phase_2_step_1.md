Это критический этап с точки зрения потребления ресурсов. Загрузка моделей трансформеров и обработка тысяч текстов могут быстро привести к Out-Of-Memory (OOM) ошибке, если не управлять памятью правильно. Мы реализуем это с использованием лучших практик MLOps: использование `torch.no_grad()`, строгий контроль размера батча и сохранение результатов в эффективном формате (Parquet), а не в раздувающемся JSON.

---

# ЭТАП 2, ПОДЗАДАЧА 2.1: Асинхронная векторизация с батчингом

## Шаг 2.1.1: Зависимости и подготовка окружения

Добавим библиотеки для работы с нейросетями и эффективного хранения табличных данных с векторами.

**Обновите `requirements.txt`:**
```text
# ... предыдущие зависимости ...

# Векторизация и ML
sentence-transformers==3.0.1
torch==2.2.1
numpy==1.26.4

# Эффективное хранение данных с векторами (вместо JSON)
pandas==2.2.1
pyarrow==15.0.0 # Движок для Parquet
```
*Действие:* Выполните `pip install -r requirements.txt`.
*Примечание:* Если вы запускаете это на машине с GPU, убедитесь, что установлена правильная версия `torch` с поддержкой CUDA (например, `pip install torch --index-url https://download.pytorch.org/whl/cu121`).

---

## Шаг 2.1.2: Реализация `EmbeddingService`

Мы создадим сервис, который загружает модель один раз (паттерн Singleton) и предоставляет метод для пакетной обработки текстов с гарантией освобождения видеопамяти/ОЗУ.

**Файл: `plugins/operators/embedding_service.py`**
```python
import logging
import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from typing import List

from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Сервис для генерации эмбеддингов текста с оптимизацией потребления памяти.
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        logger.info(f"Initializing EmbeddingService with model: {model_name}")
        
        # 1. Определение устройства (GPU предпочтительнее, иначе CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device for embeddings: {self.device}")
        
        # 2. Загрузка модели
        # trust_remote_code=True может понадобиться для некоторых кастомных архитектур
        self.model = SentenceTransformer(
            model_name, 
            device=self.device,
            trust_remote_code=True
        )
        
        # 3. Перевод модели в режим инференса (отключает Dropout и вычисление градиентов)
        self.model.eval()
        
        # Опционально: компиляция модели для ускорения (работает в PyTorch 2.0+)
        # if self.device == "cuda":
        #     self.model = torch.compile(self.model)
            
        logger.info("EmbeddingService initialized successfully.")

    def generate_embeddings_batched(
        self, 
        texts: List[str], 
        batch_size: int = 64
    ) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов, обрабатывая их батчами 
        для предотвращения OOM (Out of Memory).
        """
        if not texts:
            return []

        logger.info(f"Starting embedding generation for {len(texts)} texts (batch_size={batch_size})")
        
        # КРИТИЧЕСКИ ВАЖНО: Отключаем вычисление градиентов для экономии памяти и ускорения
        with torch.no_grad():
            # model.encode автоматически разбивает данные на батчи и показывает прогресс
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True, # Возвращаем numpy array для совместимости с pandas
                show_progress_bar=True,
                normalize_embeddings=True # Нормализация (L2) важна для косинусного сходства в HDBSCAN
            )
            
        logger.info("Embedding generation completed.")
        # Преобразуем numpy array в список списков Python для сериализации
        return embeddings.tolist()

# Глобальный синглтон
embedding_service = EmbeddingService()
```

---

## Шаг 2.1.3: Интеграция векторизации в пайплайн очистки

Теперь мы создадим задачу, которая читает очищенные данные из Этапа 1, добавляет к ним векторы и сохраняет результат. 

**Почему Parquet, а не JSON?** 
JSON хранит массивы из 1024 чисел как строки, что раздувает размер файла в 5-10 раз и замедляет чтение/запись. Parquet сжимает данные, сохраняет типы (numpy arrays) и читается мгновенно.

**Файл: `plugins/operators/vectorize_logs.py`**
```python
import logging
import os
import pandas as pd
from typing import Tuple

from config.settings import settings
from plugins.operators.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class Vectorizer:
    @staticmethod
    def process_and_vectorize(input_filepath: str, execution_date_str: str) -> Tuple[str, int]:
        """
        Читает очищенные логи, генерирует эмбеддинги и сохраняет в Parquet.
        """
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Input file not found: {input_filepath}")

        logger.info(f"Loading cleaned data from {input_filepath}")
        # Читаем JSON, созданный на предыдущем шаге
        df = pd.read_json(input_filepath)
        
        if df.empty or 'lemmatized_text' not in df.columns:
            raise ValueError("DataFrame is empty or missing 'lemmatized_text' column")

        texts_to_embed = df['lemmatized_text'].tolist()
        
        # Генерация эмбеддингов (с автоматическим батчингом внутри сервиса)
        # batch_size=64 оптимален для CPU. Для GPU с 8GB+ VRAM можно поставить 128 или 256
        batch_size = 128 if torch.cuda.is_available() else 64
        embeddings = embedding_service.generate_embeddings_batched(texts_to_embed, batch_size=batch_size)
        
        # Добавляем эмбеддинги как новый столбец
        df['embedding'] = embeddings
        
        # Сохраняем в Parquet (гораздо эффективнее JSON для векторов)
        output_filename = f"vectorized_logs_{execution_date_str}.parquet"
        output_filepath = os.path.join(settings.data_storage_path, output_filename)
        
        df.to_parquet(output_filepath, index=False, engine='pyarrow')
        
        logger.info(f"Vectorization complete. Saved {len(df)} records with embeddings to {output_filepath}")
        
        # Возвращаем путь и количество обработанных записей для метрик Airflow
        return output_filepath, len(df)

vectorizer = Vectorizer()
```

---

## Шаг 2.1.4: Обновление DAG в Airflow

Добавим новую задачу в наш DAG, связав ее с предыдущей.

**Обновите файл `dags/nlu_self_healing_extract.py`:**
```python
# ... предыдущие импорты ...
from plugins.operators.vectorize_logs import vectorizer

# ... внутри блока with DAG(...): ...

    def task_vectorize(**context: Context):
        ti = context["task_instance"]
        # Получаем путь к файлу из предыдущей задачи
        input_file = ti.xcom_pull(task_ids="clean_and_mask_pii", key="masked_file_path")
        execution_date_str = context["ds"]
        
        if not input_file:
            logger.info("No cleaned file to vectorize. Skipping.")
            return
            
        output_file, record_count = vectorizer.process_and_vectorize(input_file, execution_date_str)
        
        # Передаем мета-информацию дальше
        ti.xcom_push(key="vectorized_file_path", value=output_file)
        ti.xcom_push(key="vectorized_count", value=record_count)

    vectorize_task = PythonOperator(
        task_id="vectorize_cleaned_logs",
        python_callable=task_vectorize,
        provide_context=True,
    )

    # Определяем порядок выполнения: Extract -> Clean/Mask -> Vectorize
    extract_task >> clean_task >> vectorize_task
```

---

## Шаг 2.1.5: Исчерпывающее тестирование векторизации

Напишем тесты, чтобы гарантировать, что модель загружается, выдает векторы правильной размерности и не падает на пустых данных.

**Файл: `tests/test_embedding_service.py`**
```python
import pytest
import pandas as pd
import os
from plugins.operators.embedding_service import EmbeddingService
from config.settings import settings

@pytest.fixture(scope="module")
def embedding_service():
    # Загружаем модель один раз для всех тестов в модуле
    return EmbeddingService(model_name="intfloat/multilingual-e5-large")

def test_embedding_dimensions(embedding_service):
    """Проверка, что модель выдает векторы ожидаемой размерности (1024 для e5-large)."""
    texts = ["Привет, мир", "Тестовый запрос"]
    embeddings = embedding_service.generate_embeddings_batched(texts, batch_size=2)
    
    assert len(embeddings) == 2
    # multilingual-e5-large имеет размерность 1024
    assert len(embeddings[0]) == 1024
    # Проверка нормализации (длина вектора должна быть близка к 1.0)
    import numpy as np
    norm = np.linalg.norm(embeddings[0])
    assert 0.99 <= norm <= 1.01, f"Vector is not normalized: {norm}"

def test_empty_input_handling(embedding_service):
    """Проверка обработки пустых списков."""
    embeddings = embedding_service.generate_embeddings_batched([], batch_size=64)
    assert embeddings == []

def test_vectorizer_end_to_end(tmp_path):
    """Интеграционный тест всего процесса векторизации."""
    from plugins.operators.vectorize_logs import vectorizer
    
    # 1. Создаем фиктивный входной файл (как после шага 1.3)
    input_data = [
        {"dialog_id": "1", "lemmatized_text": "хотеть вернуть деньги"},
        {"dialog_id": "2", "lemmatized_text": "оформить возврат средств"}
    ]
    input_df = pd.DataFrame(input_data)
    input_filepath = tmp_path / "cleaned_test.json"
    input_df.to_json(input_filepath, orient="records", force_ascii=False)
    
    # 2. Запускаем векторизацию
    output_filepath, count = vectorizer.process_and_vectorize(str(input_filepath), "2024-01-01")
    
    # 3. Проверяем результат
    assert count == 2
    assert os.path.exists(output_filepath)
    assert output_filepath.endswith(".parquet")
    
    # 4. Читаем Parquet и проверяем структуру
    result_df = pd.read_parquet(output_filepath)
    assert "embedding" in result_df.columns
    assert len(result_df["embedding"].iloc[0]) == 1024 # Проверка размерности в DataFrame
```

**Запуск тестов:**
```bash
pytest tests/test_embedding_service.py -v -s
```
*(Флаг `-s` позволит увидеть прогресс-бар загрузки модели и генерации эмбеддингов).*

---

## Шаг 2.1.6: Важные архитектурные настройки для Production

1. **Кэширование модели Hugging Face:** При первом запуске `SentenceTransformer` скачает модель (~2.2 ГБ для e5-large) в домашнюю директорию (`~/.cache/huggingface`). В Docker-контейнере Airflow это может привести к потере кэша при перезапуске. 
   *Решение:* Добавьте в `docker-compose.yaml` том для кэша:
   ```yaml
   volumes:
     - ./hf_cache:/root/.cache/huggingface # Для Linux
     # или
     - ./hf_cache:/home/airflow/.cache/huggingface # Зависит от пользователя в образе
   ```
2. **Переменная окружения `TRANSFORMERS_OFFLINE`:** Если вы предварительно скачали модель, установите эту переменную в `1`, чтобы Airflow не пытался проверить обновления при каждом запуске DAG, что ускорит старт задачи.
3. **Мониторинг памяти:** Если вы используете CPU, убедитесь, что в настройках Airflow Worker (`AIRFLOW__CELERY__WORKER_CONCURRENCY` или аналогичных) concurrency не слишком высок, иначе параллельные задачи векторизации могут исчерпать ОЗУ сервера.

---

### Что мы достигли в Подзадаче 2.1:
✅ **Оптимизация памяти:** Использование `torch.no_grad()`, `model.eval()` и строгого `batch_size` гарантирует, что процесс не упадет с OOM даже на больших выборках (10 000+ записей).
✅ **MLOps Best Practices:** Переход с JSON на **Parquet** для хранения векторов. Это снижает размер артефактов в 5-10 раз и ускоряет чтение для следующего этапа (HDBSCAN) на порядки.
✅ **Нормализация векторов:** Включение `normalize_embeddings=True` критически важно, так как алгоритм HDBSCAN по умолчанию использует косинусное расстояние, которое корректно работает только с нормализованными (единичной длины) векторами.
✅ **Интеграция в DAG:** Бесшовная передача файлов между задачами через XCom (только пути, а не сами данные), что соответствует лучшим практикам Airflow.
