Это критически важный шаг для реализации паттерна **Human-in-the-Loop (HITL)**. Мы не просто отправляем сухие данные, мы создаем для лингвиста/аналитика удобную, визуально структурированную задачу в Битрикс24, содержащую всю необходимую информацию для принятия решения, и делаем это с гарантией того, что сбои CRM не "повесят" весь Airflow DAG.

---

# ЭТАП 4, ПОДЗАДАЧА 4.2: Интеграция с Битрикс24 и Circuit Breaker

## Шаг 4.2.1: Зависимости и Конфигурация

Добавим библиотеку для реализации паттерна Circuit Breaker и обновим настройки.

**Обновите `requirements.txt`:**
```text
# ... предыдущие зависимости ...

# Circuit Breaker для защиты от каскадных сбоев внешних API
pybreaker==1.0.2
```
*Действие:* Выполните `pip install -r requirements.txt`.

**Обновите `config/settings.py`:**
```python
    # --- Bitrix24 Integration ---
    bitrix24_webhook_url: str = Field(default="https://your-domain.bitrix24.ru/rest/1/your_webhook_token/")
    bitrix24_timeout_sec: float = Field(default=5.0, description="Таймаут запросов к Битрикс24")
    bitrix24_cb_fail_max: int = Field(default=3, description="Кол-во ошибок подряд для размыкания цепи")
    bitrix24_cb_reset_timeout: int = Field(default=60, description="Секунд до попытки восстановления цепи")
    bitrix24_responsible_id: int = Field(default=1, description="ID пользователя в Битрикс24, на которого ставится задача")
```

---

## Шаг 4.2.2: Реализация защищенного клиента Битрикс24

**Ключевой нюанс Битрикс24:** При возникновении ошибки (например, неверный формат поля) Битрикс часто возвращает HTTP-статус `200 OK`, а саму ошибку кладет в тело ответа: `{"error": "ERROR_CODE", "error_description": "..."}`. Стандартный `response.raise_for_status()` это пропустит. Мы должны обрабатывать это явно.

Кроме того, `pybreaker` является синхронной библиотекой. Чтобы не блокировать `asyncio` event loop в Airflow, мы обернем его вызов в `asyncio.to_thread`.

**Файл: `plugins/operators/bitrix24_client.py`**
```python
import logging
import asyncio
import pybreaker
import httpx
from typing import Dict, Any, List

from config.settings import settings

logger = logging.getLogger(__name__)

# 1. Глобальная настройка Circuit Breaker
bitrixCircuitBreaker = pybreaker.CircuitBreaker(
    fail_max=settings.bitrix24_cb_fail_max,
    reset_timeout=settings.bitrix24_cb_reset_timeout
)

class Bitrix24Client:
    def __init__(self):
        # Создаем один клиент на приложение для переиспользования соединений (connection pooling)
        self.client = httpx.AsyncClient(
            timeout=settings.bitrix24_timeout_sec,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers={"Content-Type": "application/json"}
        )
        self.base_url = settings.bitrix24_webhook_url.rstrip('/')
        logger.info("Bitrix24Client initialized with Circuit Breaker protection.")

    def _check_bitrix_error(self, response: httpx.Response) -> None:
        """Специфичная проверка: ловим ошибки, спрятанные внутри HTTP 200 OK."""
        response.raise_for_status() # Ловим реальные HTTP ошибки (404, 502, 504)
        
        data = response.json()
        if "error" in data:
            raise ValueError(f"Bitrix24 API Error: {data['error']} - {data.get('error_description', 'Unknown error')}")

    async def create_approval_task(
        self, 
        intent_name: str,
        description: str,
        original_examples: List[str],
        generated_utterances: List[str],
        s3_export_url: str,
        cluster_id: int
    ) -> Dict[str, Any]:
        """
        Создает задачу в Битрикс24 с богатым HTML-описанием для апрува лингвистом.
        """
        # Формируем красивое HTML-описание
        original_list = "".join([f"<li>{ex}</li>" for ex in original_examples[:3]]) # Показываем топ-3 оригинала
        generated_list = "".join([f"<li>{utt}</li>" for utt in generated_utterances[:5]]) # Показываем топ-5 сгенерированных
        
        html_description = f"""
        <b>🤖 AI предложил новый интент для добавления в NLU-модель</b><br><br>
        <b>📌 Название интента:</b> <code>{intent_name}</code><br>
        <b>📝 Описание:</b> {description}<br>
        <b>🔗 ID исходного кластера:</b> {cluster_id}<br><br>
        
        <b>🔍 Топ-3 реальных фразы из логов (после очистки):</b>
        <ul>{original_list}</ul>
        
        <b>✨ Топ-5 сгенерированных фраз для обучения:</b>
        <ul>{generated_list}</ul>
        
        <hr>
        <b>📥 Полный список фраз для импорта:</b><br>
        <a href="{s3_export_url}" target="_blank">Скачать CSV-файл для Just AI Caila</a><br><br>
        
        <i>Пожалуйста, проверьте корректность интента. Если все верно, подтвердите задачу, и данные будут автоматически импортированы.</i>
        """

        # Поля для создания задачи в Битрикс24
        task_fields = {
            "TITLE": f"[NLU Auto] Аппрув интента: {intent_name}",
            "DESCRIPTION": html_description,
            "RESPONSIBLE_ID": settings.bitrix24_responsible_id,
            "PRIORITY": "2", # Средняя приоритетность
            "TAGS": "nlu,ai-generated,approval-needed"
        }

        # 2. Оборачиваем синхронный Circuit Breaker в asyncio.to_thread
        def _make_request() -> Dict[str, Any]:
            @bitrixCircuitBreaker
            def _protected_request():
                response = self.client.post(
                    f"{self.base_url}/task.item.add.json",
                    json={"fields": task_fields}
                )
                self._check_bitrix_error(response)
                return response.json().get("result", {})
            
            return _protected_request()

        try:
            logger.info(f"Creating Bitrix24 task for intent: {intent_name}")
            result = await asyncio.to_thread(_make_request)
            logger.info(f"Bitrix24 task created successfully. Task ID: {result.get('task', {}).get('id')}")
            return result
            
        except pybreaker.CircuitBreakerError:
            logger.critical(
                "bitrix24_circuit_breaker_open", 
                msg="Битрикс24 недоступен или возвращает ошибки. Запрос отклонен для защиты системы."
            )
            raise # Пробрасываем исключение, чтобы Airflow перевел задачу в состояние 'up_for_retry' или 'failed'
            
        except Exception as e:
            logger.error(f"Failed to create Bitrix24 task: {str(e)}", exc_info=True)
            raise

    async def close(self):
        await self.client.aclose()

# Глобальный синглтон
bitrix_client = Bitrix24Client()
```

---

## Шаг 4.2.3: Оператор Airflow для создания задач

Теперь создадим задачу, которая читает сгенерированный датасет и для *каждого* успешного интента создает отдельную задачу в Битрикс24 (или одну сводную, в зависимости от бизнес-логики. Для лучшего контроля качества лингвистом лучше создавать **одну задачу на каждый новый интент**).

**Файл: `plugins/operators/bitrix_approval.py`**
```python
import logging
import json
import os
import asyncio
from typing import List, Dict, Any

from config.settings import settings
from plugins.operators.bitrix24_client import bitrix_client

logger = logging.getLogger(__name__)

class BitrixApprovalManager:
    @staticmethod
    async def _create_single_task(
        intent_data: Dict[str, Any], 
        original_examples: List[str], 
        s3_url: str
    ) -> Dict[str, Any]:
        """Асинхронная обертка для создания одной задачи."""
        gen_intent = intent_data["generated_intent"]
        return await bitrix_client.create_approval_task(
            intent_name=gen_intent["intent_name"],
            description=gen_intent["description"],
            original_examples=original_examples,
            generated_utterances=gen_intent["utterances"],
            s3_export_url=s3_url,
            cluster_id=intent_data["cluster_id"]
        )

    @staticmethod
    def process_and_notify(input_filepath: str, s3_export_url: str, execution_date_str: str) -> int:
        """
        Читает датасет и создает задачи в Битрикс24 параллельно.
        """
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Generated dataset file not found: {input_filepath}")

        with open(input_filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Фильтруем только успешные интенты
        successful_intents = [item for item in dataset if item.get("processing_status") == "success"]
        
        if not successful_intents:
            logger.info("No successful intents to create approval tasks for.")
            return 0

        logger.info(f"Starting creation of {len(successful_intents)} Bitrix24 approval tasks...")

        # Ограничиваем параллелизм, чтобы не получить бан от Битрикс24 за спам запросами
        semaphore = asyncio.Semaphore(3) # Максимум 3 одновременных запроса к API Битрикс24
        
        async def bounded_create(item):
            async with semaphore:
                # Для демонстрации берем первые 3 фразы как "оригинальные примеры" из кластера
                # В реальном сценарии их можно сохранить на этапе кластеризации в metadata
                original_examples = item["generated_intent"]["utterances"][:3] 
                return await BitrixApprovalManager._create_single_task(item, original_examples, s3_export_url)

        async def run_all():
            tasks = [bounded_create(item) for item in successful_intents]
            # return_exceptions=True позволяет продолжить выполнение, если одна задача упала, 
            # но мы хотим знать о сбоях, поэтому обрабатываем их ниже
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Запуск асинхронного кода в синхронном контексте Airflow
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_all())
        finally:
            loop.close()

        # Анализ результатов
        success_count = 0
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Failed to create Bitrix task: {str(res)}")
                # В продакшене здесь можно добавить логику повторной попытки или алертинга
            else:
                success_count += 1

        logger.info(f"Bitrix24 integration complete. Successfully created {success_count}/{len(successful_intents)} tasks.")
        return success_count

approval_manager = BitrixApprovalManager()
```

---

## Шаг 4.2.4: Интеграция в Airflow DAG

Добавим финальную задачу в наш пайплайн.

**Обновите файл `dags/nlu_self_healing_extract.py`:**
```python
# ... предыдущие импорты ...
from plugins.operators.bitrix_approval import approval_manager

# ... внутри блока with DAG(...): ...

    def task_create_bitrix_approvals(**context: Context):
        ti = context["task_instance"]
        input_file = ti.xcom_pull(task_ids="generate_intents_via_llm", key="generated_dataset_file")
        s3_url = ti.xcom_pull(task_ids="export_to_caila_format_and_s3", key="caila_export_url")
        execution_date_str = context["ds"]
        
        if not input_file or not s3_url:
            logger.info("Missing input file or S3 URL. Skipping Bitrix integration.")
            return
            
        created_tasks_count = approval_manager.process_and_notify(input_file, s3_url, execution_date_str)
        ti.xcom_push(key="bitrix_tasks_created", value=created_tasks_count)

    bitrix_task = PythonOperator(
        task_id="create_bitrix24_approval_tasks",
        python_callable=task_create_bitrix_approvals,
        provide_context=True,
        # Увеличиваем таймаут, так как создание множества задач может занять время
        execution_timeout=timedelta(minutes=15)
    )

    # Финальная цепочка: ... -> Export -> Bitrix Approval
    extract_task >> clean_task >> vectorize_task >> cluster_task >> finalize_task >> generate_task >> export_task >> bitrix_task
```

---

## Шаг 4.2.5: Исчерпывающее тестирование Circuit Breaker и интеграции

Напишем тест, который эмулирует сбой Битрикс24 и проверяет, что Circuit Breaker корректно размыкается и защищает систему.

**Файл: `tests/test_bitrix24_client.py`**
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pybreaker
from plugins.operators.bitrix24_client import Bitrix24Client, bitrixCircuitBreaker

@pytest.fixture
def bx_client():
    # Сбрасываем состояние breaker перед каждым тестом
    bitrixCircuitBreaker.reset()
    return Bitrix24Client()

@pytest.mark.asyncio
async def test_successful_task_creation(bx_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"task": {"id": 12345}}}
    mock_response.raise_for_status = MagicMock()
    
    with patch.object(bx_client.client, 'post', new_callable=AsyncMock, return_value=mock_response) as mock_post:
        result = await bx_client.create_approval_task(
            intent_name="test_intent", description="desc", 
            original_examples=["a"], generated_utterances=["b"], 
            s3_export_url="http://s3.url", cluster_id=1
        )
        assert result["task"]["id"] == 12345
        assert mock_post.call_count == 1

@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_repeated_failures(bx_client):
    mock_response = MagicMock()
    # Эмулируем ошибку Битрикс24: HTTP 200, но с error в теле
    mock_response.json.return_value = {"error": "ERROR_METHOD_NOT_FOUND", "error_description": "..."}
    mock_response.raise_for_status = MagicMock()
    
    with patch.object(bx_client.client, 'post', new_callable=AsyncMock, return_value=mock_response):
        # Делаем 3 запроса, чтобы исчерпать fail_max (по умолчанию 3)
        for i in range(3):
            with pytest.raises(ValueError, match="Bitrix24 API Error"):
                await bx_client.create_approval_task("test", "desc", ["a"], ["b"], "url", 1)
        
        # 4-й запрос должен быть мгновенно отклонен Circuit Breaker-ом
        with pytest.raises(pybreaker.CircuitBreakerError):
            await bx_client.create_approval_task("test", "desc", ["a"], ["b"], "url", 1)
```

**Запуск тестов:**
```bash
pytest tests/test_bitrix24_client.py -v
```

---

## Шаг 4.2.6: Важные архитектурные нюансы для Production

1. **Семафор параллелизма (`asyncio.Semaphore(3)`)**: Битрикс24 (особенно на тарифах с ограничениями) может вернуть ошибку `429 Too Many Requests` или временно заблокировать IP при массовом спаме запросами. Семафор гарантирует, что мы не превысим безопасный лимит одновременных соединений, жертвуя минимальной скоростью ради 100% надежности доставки.
2. **Обработка `error` внутри `200 OK`**: Метод `_check_bitrix_error` является критическим. Без него Airflow считал бы запрос успешным, задача создавалась бы "в никуда", а лингвист никогда не получил бы уведомление, что сломало весь бизнес-процесс HITL.
3. **Читабельность для человека**: HTML-шаблон намеренно ограничен (топ-3 оригинала, топ-5 генераций). Если сгенерировать 30 фраз прямо в описании задачи, Битрикс24 может обрезать текст или сделать задачу нечитаемой. Ссылка на полный CSV в S3 решает эту проблему элегантно.
4. **Изоляция сбоев**: Благодаря `return_exceptions=True` в `asyncio.gather`, сбой при создании *одной* задачи не прерывает создание остальных. Ошибка логируется, и пайплайн продолжает работу.

---

### Что мы достигли в Подзадаче 4.2:
✅ **Надежная интеграция (Circuit Breaker)**: Система защищена от каскадных сбоев. Если Битрикс24 "упал", Airflow не зависнет на 30 минут, а быстро получит ошибку и либо завершится, либо уйдет в retry, не создавая очередь из зависших процессов.
✅ **Human-Centric Design**: Задача в Битрикс24 содержит ровно ту информацию, которая нужна лингвисту для принятия решения за 10-15 секунд, плюс прямую ссылку на полные данные.
✅ **Асинхронная эффективность**: Использование `httpx` и `asyncio` позволяет создавать десятки задач за секунды, не блокируя воркер Airflow.
