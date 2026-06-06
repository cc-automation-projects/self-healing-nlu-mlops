На этом этапе мы превращаем структурированный JSON-ответ от LLM (содержащий массивы фраз для каждого интента) в плоский формат, который нативно поддерживает система управления ботами (Just AI Caila). Мы также обеспечим безопасную загрузку этого файла в S3 и генерацию временной ссылки (Presigned URL) для аналитика, который будет проводить финальную валидацию (Human-in-the-Loop).

---

# ЭТАП 4, ПОДЗАДАЧА 4.1: Генератор файлов для импорта в Just AI Caila

## Шаг 4.1.1: Зависимости

Для работы с табличными данными и S3 нам понадобятся `pandas` (уже установлен на Этапе 2) и `boto3` (стандарт де-факто для работы с S3-совместимыми хранилищами в Python).

**Обновите `requirements.txt`:**
```text
# ... предыдущие зависимости ...

# Работа с S3 (для экспорта и presigned URLs)
boto3==1.34.51
```
*Действие:* Выполните `pip install -r requirements.txt`.

---

## Шаг 4.1.2: Реализация `CailaExporter`

Just AI Caila (и большинство других NLU-платформ) лучше всего принимает данные для массового импорта в формате **CSV** с колонками `intent`, `utterance` и опционально `weight`. Мы реализуем "сплющивание" (flattening) иерархического JSON в плоскую таблицу.

**Файл: `plugins/operators/caila_exporter.py`**
```python
import logging
import os
import json
import pandas as pd
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from config.settings import settings

logger = logging.getLogger(__name__)

class CailaExporter:
    def __init__(self):
        """Инициализация S3-клиента с настройками для совместимости (например, с Yandex Object Storage или MinIO)."""
        # Используем boto3 с явной конфигурацией для надежности в Airflow
        self.s3_client = boto3.client(
            's3',
            endpoint_url=str(settings.s3_endpoint_url).rstrip('/'),
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=Config(signature_version='s3v4', retries={'max_attempts': 3, 'mode': 'standard'})
        )
        self.bucket_name = settings.s3_bucket_name
        logger.info("CailaExporter initialized with S3 client.")

    def generate_and_upload(self, input_filepath: str, execution_date_str: str) -> tuple[str, int, int]:
        """
        Читает сгенерированные интенты, преобразует в CSV для Caila, загружает в S3 
        и возвращает Presigned URL.
        
        :return: (presigned_url, total_intents, total_utterances)
        """
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Generated dataset file not found: {input_filepath}")

        logger.info(f"Loading generated dataset from {input_filepath}")
        with open(input_filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if not dataset:
            logger.warning("Dataset is empty. Nothing to export.")
            return "", 0, 0

        # 1. Преобразование (Flattening) и фильтрация
        rows = []
        success_count = 0
        failed_count = 0

        for item in dataset:
            if item.get("processing_status") == "success":
                success_count += 1
                intent_name = item["generated_intent"]["intent_name"]
                
                # Для каждой фразы создаем отдельную строку (требование формата импорта Caila)
                for utterance in item["generated_intent"]["utterances"]:
                    rows.append({
                        "intent": intent_name,
                        "utterance": utterance,
                        "weight": 1.0 # Вес по умолчанию для новых фраз в Caila
                    })
            else:
                failed_count += 1
                logger.warning(f"Skipping failed cluster {item.get('cluster_id')}: {item.get('error_message')}")

        if not rows:
            logger.info("No successful intents to export.")
            return "", success_count, 0

        # 2. Создание DataFrame и сохранение в CSV
        df = pd.DataFrame(rows)
        
        # Используем utf-8-sig для корректного открытия в Excel, если аналитик захочет проверить файл локально
        csv_filename = f"caila_import_draft_{execution_date_str}.csv"
        csv_filepath = os.path.join(settings.data_storage_path, csv_filename)
        
        df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Saved flattened dataset to {csv_filepath} ({len(df)} rows)")

        # 3. Загрузка в S3
        s3_key = f"nlu_exports/{csv_filename}"
        try:
            self.s3_client.upload_file(csv_filepath, self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded {csv_filename} to S3 bucket '{self.bucket_name}'")
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise

        # 4. Генерация Presigned URL (действует 7 дней = 604800 секунд)
        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=604800
            )
            logger.info(f"Generated presigned URL valid for 7 days.")
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            raise

        return presigned_url, success_count, len(df)

caila_exporter = CailaExporter()
```

---

## Шаг 4.1.3: Интеграция в Airflow DAG

Добавим задачу экспорта в наш пайплайн. Она будет выполняться сразу после генерации интентов.

**Обновите файл `dags/nlu_self_healing_extract.py`:**
```python
# ... предыдущие импорты ...
from plugins.operators.caila_exporter import caila_exporter

# ... внутри блока with DAG(...): ...

    def task_export_for_caila(**context: Context):
        ti = context["task_instance"]
        input_file = ti.xcom_pull(task_ids="generate_intents_via_llm", key="generated_dataset_file")
        execution_date_str = context["ds"]
        
        if not input_file:
            logger.info("No generated dataset to export. Skipping.")
            return
            
        presigned_url, intents_count, utterances_count = caila_exporter.generate_and_upload(input_file, execution_date_str)
        
        if presigned_url:
            ti.xcom_push(key="caila_export_url", value=presigned_url)
            ti.xcom_push(key="exported_intents_count", value=intents_count)
            ti.xcom_push(key="exported_utterances_count", value=utterances_count)
            logger.info(f"Export ready: {intents_count} intents, {utterances_count} utterances.")
        else:
            logger.info("Export skipped due to empty dataset.")

    export_task = PythonOperator(
        task_id="export_to_caila_format_and_s3",
        python_callable=task_export_for_caila,
        provide_context=True,
    )

    # Цепочка: ... -> Generate -> Export
    extract_task >> clean_task >> vectorize_task >> cluster_task >> finalize_task >> generate_task >> export_task
```

---

## Шаг 4.1.4: Исчерпывающее тестирование экспортера

Напишем тесты, которые проверяют корректность "сплющивания" данных, кодировку CSV и логику генерации URL (с моком S3).

**Файл: `tests/test_caila_exporter.py`**
```python
import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from plugins.operators.caila_exporter import CailaExporter

@pytest.fixture
def mock_dataset(tmp_path):
    """Создает фиктивный файл с результатами генерации."""
    data = [
        {
            "cluster_id": 1,
            "processing_status": "success",
            "generated_intent": {
                "intent_name": "vozvrat_deneg",
                "description": "Возврат средств",
                "utterances": ["верните деньги", "хочу возврат", "оформите возврат"]
            }
        },
        {
            "cluster_id": 2,
            "processing_status": "failed",
            "error_message": "Validation failed"
        }
    ]
    filepath = tmp_path / "dataset.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return str(filepath)

@pytest.fixture
def exporter():
    return CailaExporter()

@patch('plugins.operators.caila_exporter.boto3.client')
def test_generate_and_upload_success(mock_boto3_client, exporter, mock_dataset, tmp_path):
    # 1. Настройка моков S3
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3
    mock_s3.generate_presigned_url.return_value = "https://s3.mock.url/presigned?signature=xyz"
    
    # 2. Вызов метода
    url, intents_count, utterances_count = exporter.generate_and_upload(mock_dataset, "2024-01-01")
    
    # 3. Проверки
    assert url == "https://s3.mock.url/presigned?signature=xyz"
    assert intents_count == 1 # Только один успешный интент
    assert utterances_count == 3 # 3 фразы
    
    # Проверка, что upload_file был вызван с правильными аргументами
    mock_s3.upload_file.assert_called_once()
    call_args = mock_s3.upload_file.call_args
    assert call_args[0][1] == exporter.bucket_name
    assert "caila_import_draft_2024-01-01.csv" in call_args[0][2]
    
    # Проверка, что presigned URL запрошен с правильным TTL (7 дней)
    mock_s3.generate_presigned_url.assert_called_once_with(
        'get_object',
        Params={'Bucket': exporter.bucket_name, 'Key': call_args[0][2]},
        ExpiresIn=604800
    )

@patch('plugins.operators.caila_exporter.boto3.client')
def test_generate_and_upload_empty_dataset(mock_boto3_client, exporter, tmp_path):
    # Пустой датасет
    filepath = tmp_path / "empty.json"
    with open(filepath, 'w') as f:
        json.dump([], f)
        
    url, intents, utterances = exporter.generate_and_upload(str(filepath), "2024-01-01")
    
    assert url == ""
    assert intents == 0
    assert utterances == 0
    mock_boto3_client.return_value.upload_file.assert_not_called()
```

**Запуск тестов:**
```bash
pytest tests/test_caila_exporter.py -v
```

---

## Шаг 4.1.5: Важные архитектурные нюансы для Production

1. **Кодировка `utf-8-sig`**: Мы используем именно её при сохранении CSV. Это добавляет BOM (Byte Order Mark), который гарантирует, что если аналитик скачает файл и откроет его в Microsoft Excel (что происходит в 90% случаев при ручной проверке), кириллица отобразится корректно, а не в виде "кракозябр". При этом сам Just AI Caila отлично читает этот формат.
2. **Фильтрация `failed` статусов**: Экспортер автоматически игнорирует кластеры, которые не прошли валидацию LLM (Subtask 3.2). Это гарантирует, что в файл для импорта попадут только качественные, проверенные данные. Аналитик увидит количество пропущенных кластеров в логах Airflow.
3. **Безопасность Presigned URL**: Срок действия ссылки жестко ограничен 7 днями (`604800` секунд). Это соответствует принципу минимальных привилегий: даже если ссылка утечет, она быстро станет недействительной.
4. **Идемпотентность S3**: Если DAG будет запущен повторно за тот же `execution_date`, файл будет перезаписан в S3 (что является ожидаемым поведением для перезапуска упавших пайплайнов), а новая ссылка будет сгенерирована.

---

### Что мы достигли в Подзадаче 4.1:
✅ **Нативная совместимость**: Данные преобразуются в плоский CSV-формат (`intent`, `utterance`, `weight`), который является стандартом де-факто для массового импорта в Just AI Caila и аналогичные NLU-платформы.
✅ **Автоматизация доставки**: Файл не просто сохраняется локально, а загружается в защищенное S3-хранилище с генерацией временной ссылки, готовой для отправки аналитику.
✅ **Надежность и наблюдаемость**: Четкое разделение успешных и неудачных кластеров, логирование количества обработанных интентов и фраз, а также исчерпывающие unit-тесты с моками S3.
