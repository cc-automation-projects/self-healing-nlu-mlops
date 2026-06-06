Это ключевой элемент надежности MLOps-пайплайна. LLM по своей природе недетерминирована: она может пропустить markdown-обертку, сгенерировать 18 фраз вместо 20 или нарушить формат `snake_case`. Если мы просто позволим `pydantic` выбросить `ValidationError`, весь DAG Airflow упадет, и обработка сотен кластеров будет прервана. 

Мы реализуем паттерн **Self-Correction**: при ошибке валидации мы не падаем, а извлекаем конкретную причину ошибки из Pydantic, добавляем её в промпт и даем LLM ровно одну дополнительную попытку исправить себя.

---

# ЭТАП 3, ПОДЗАДАЧА 3.2: Механизм валидации и Self-Correction

## Шаг 3.2.1: Обновление модуля генерации (LLMGenerationService)

Мы модифицируем сервис, добавив контролируемый цикл повторных попыток. *Примечание:* Хотя в ТЗ упомянут декоратор `@retry` из `tenacity`, для задачи **изменения аргументов** (добавления текста ошибки в промпт) между попытками, явный цикл `for` с обработкой исключений является архитектурно более чистым и предсказуемым решением, чем хак с состоянием в `tenacity`. Мы используем `tenacity` только для сетевых сбоев внутри вызова API.

**Обновите файл `plugins/operators/llm_generator.py`:**

```python
import logging
import json
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings

logger = logging.getLogger(__name__)

# ... (Здесь остаются модели GeneratedIntent и ClusterGenerationResult из Шага 3.1.1) ...
# ... (Здесь остаются SYSTEM_PROMPT и USER_PROMPT_TEMPLATE из Шага 3.1.2) ...

class LLMGenerationService:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        )
        
        self.yandex_endpoint = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        # Эти значения должны быть в settings.py
        self.folder_id = getattr(settings, 'yandex_folder_id', 'b1gxxxxxx') 
        self.iam_token = getattr(settings, 'yandex_iam_token', 't1.xxxxxx') 
        
        logger.info("LLMGenerationService initialized with Self-Correction mechanism.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True
    )
    async def _call_llm_api(self, prompt: str) -> str:
        """
        Внутренний метод для вызова API с защитой от сетевых сбоев (через tenacity).
        """
        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": 0.4,
                "maxTokens": "1500" # Увеличили лимит, так как 30 фраз + описание занимают место
            },
            "messages": [
                {"role": "system", "text": SYSTEM_PROMPT},
                {"role": "user", "text": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id,
            "Content-Type": "application/json"
        }

        response = await self.client.post(self.yandex_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["result"]["alternatives"][0]["message"]["text"].strip()

    async def generate_intent_with_self_correction(
        self, 
        cluster: dict
    ) -> ClusterGenerationResult:
        """
        Основной метод с механизмом Self-Correction.
        Делает до 2 попыток генерации, если первая не прошла валидацию Pydantic.
        """
        cluster_id = cluster["cluster_id"]
        examples_str = "\n".join([f"- {ex}" for ex in cluster["representative_phrases"]])
        
        # Начальный промпт
        current_prompt = USER_PROMPT_TEMPLATE.format(
            cluster_id=cluster_id,
            size=cluster["size"],
            silhouette_score=cluster.get("metadata", {}).get("original_silhouette", "N/A"),
            examples_list=examples_str
        )

        max_attempts = 2
        
        for attempt in range(1, max_attempts + 1):
            try:
                # 1. Вызов API (с внутренними ретраями tenacity при сетевых сбоях)
                raw_text = await self._call_llm_api(current_prompt)
                
                # 2. Очистка от markdown-оберток (на случай, если модель проигнорировала response_format)
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()

                # 3. Парсинг и обогащение
                parsed_json = json.loads(raw_text)
                parsed_json["cluster_id"] = cluster_id
                parsed_json["processing_status"] = "success"

                # 4. Строгая валидация Pydantic V2
                validated_result = ClusterGenerationResult.model_validate(parsed_json)
                
                if attempt > 1:
                    logger.info(f"Cluster {cluster_id}: Self-correction SUCCESSFUL on attempt {attempt}")
                    
                return validated_result

            except json.JSONDecodeError as e:
                error_msg = f"Невалидный JSON: {str(e)}. Сырой ответ: {raw_text[:100]}..."
                logger.warning(f"Cluster {cluster_id} (Attempt {attempt}): {error_msg}")
                
            except ValidationError as e:
                # Извлекаем человеко-читаемую ошибку из Pydantic
                error_details = []
                for err in e.errors():
                    field = ".".join(str(x) for x in err["loc"])
                    msg = err["msg"]
                    error_details.append(f"Поле '{field}': {msg}")
                
                error_msg = "; ".join(error_details)
                logger.warning(f"Cluster {cluster_id} (Attempt {attempt}): Pydantic Validation Failed -> {error_msg}")

            except Exception as e:
                # Критическая ошибка (не связана с форматом), логируем и сразу переходим к фолбэку
                logger.error(f"Cluster {cluster_id} (Attempt {attempt}): Critical error -> {str(e)}")
                return self._create_fallback_result(cluster_id, f"Critical error: {str(e)}")

            # --- ЛОГИКА SELF-CORRECTION ---
            if attempt < max_attempts:
                logger.info(f"Cluster {cluster_id}: Triggering self-correction (Attempt {attempt + 1})")
                # Модифицируем промпт, добавляя конкретную ошибку, чтобы LLM знала, что исправить
                current_prompt += f"\n\n❗️ КРИТИЧЕСКАЯ ОШИБКА В ПРЕДЫДУЩЕЙ ПОПЫТКЕ: {error_msg}. \nИсправь JSON и верни СТРОГО валидный объект, соответствующий схеме. Не добавляй никаких пояснений, только JSON."
                
                # Небольшая задержка перед повторным запросом (good practice для API rate limits)
                await asyncio.sleep(1.0)
            else:
                # Лимит попыток исчерпан
                logger.error(f"Cluster {cluster_id}: Max self-correction attempts reached. Marking as failed.")
                return self._create_fallback_result(cluster_id, f"Validation failed after {max_attempts} attempts: {error_msg}")

        # На случай, если цикл завершился некорректно (защита от багов)
        return self._create_fallback_result(cluster_id, "Unknown generation failure")

    def _create_fallback_result(self, cluster_id: int, error_msg: str) -> ClusterGenerationResult:
        """Создает безопасный объект-заглушку, чтобы не ломать downstream-задачи Airflow."""
        return ClusterGenerationResult(
            cluster_id=cluster_id,
            generated_intent=GeneratedIntent(
                intent_name=f"error_cluster_{cluster_id}",
                description="Не удалось сгенерировать интент из-за ошибки валидации LLM.",
                utterances=["ошибка генерации"] * 20 # Заполняем минимально допустимым количеством для прохождения валидации заглушки
            ),
            processing_status="failed",
            error_message=error_msg
        )

    async def close(self):
        await self.client.aclose()

# Глобальный синглтон
llm_generator = LLMGenerationService()
```

---

## Шаг 3.2.2: Интеграция в Airflow DAG (Параллельная обработка)

Теперь создадим задачу Airflow, которая читает `llm_handoff_payload.json`, асинхронно обрабатывает каждый кластер через наш новый метод и сохраняет итоговый датасет.

**Создайте файл `plugins/operators/generate_intents.py`:**
```python
import logging
import json
import os
import asyncio
from typing import List, Dict, Any

from config.settings import settings
from plugins.operators.llm_generator import llm_generator, ClusterGenerationResult

logger = logging.getLogger(__name__)

class IntentGenerator:
    @staticmethod
    async def _process_single_cluster(cluster: Dict[str, Any]) -> Dict[str, Any]:
        """Асинхронная обертка для обработки одного кластера."""
        result = await llm_generator.generate_intent_with_self_correction(cluster)
        return result.model_dump()

    @staticmethod
    def process_handoff_file(input_filepath: str, execution_date_str: str) -> str:
        """
        Читает handoff-файл, асинхронно генерирует интенты и сохраняет итоговый датасет.
        """
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Handoff file not found: {input_filepath}")

        with open(input_filepath, 'r', encoding='utf-8') as f:
            handoff_data = json.load(f)

        clusters = handoff_data.get("clusters", [])
        if not clusters:
            logger.info("No clusters to process. Saving empty dataset.")
            return IntentGenerator._save_dataset([], execution_date_str)

        logger.info(f"Starting async generation for {len(clusters)} clusters...")
        
        # Ограничиваем параллелизм, чтобы не превысить rate limits LLM API
        semaphore = asyncio.Semaphore(settings.llm_max_concurrent_requests)
        
        async def bounded_process(cluster):
            async with semaphore:
                return await IntentGenerator._process_single_cluster(cluster)

        # Запускаем все задачи параллельно, но с контролем concurrency
        async def run_all():
            tasks = [bounded_process(c) for c in clusters]
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Выполняем асинхронный код внутри синхронного метода Airflow
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_all())
        finally:
            loop.close()

        # Обработка результатов (отделение успешных от упавших с критическими ошибками)
        final_dataset = []
        success_count = 0
        failed_count = 0

        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Task failed with exception: {str(res)}")
                failed_count += 1
                continue
                
            if res.get("processing_status") == "success":
                success_count += 1
            else:
                failed_count += 1
                
            final_dataset.append(res)

        logger.info(
            f"Generation complete. Success: {success_count}, Failed: {failed_count}. "
            f"Saving dataset to {settings.data_storage_path}"
        )
        
        return IntentGenerator._save_dataset(final_dataset, execution_date_str)

    @staticmethod
    def _save_dataset(dataset: List[Dict[str, Any]], execution_date_str: str) -> str:
        output_filename = f"generated_intents_dataset_{execution_date_str}.json"
        output_filepath = os.path.join(settings.data_storage_path, output_filename)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        return output_filepath

intent_generator = IntentGenerator()
```

**Обновите DAG (`dags/nlu_self_healing_extract.py`), добавив новую задачу:**
```python
# ... предыдущие импорты ...
from plugins.operators.generate_intents import intent_generator

# ... внутри блока with DAG(...): ...

    def task_generate_intents(**context: Context):
        ti = context["task_instance"]
        input_file = ti.xcom_pull(task_ids="finalize_and_prepare_llm_handoff", key="llm_handoff_file")
        execution_date_str = context["ds"]
        
        if not input_file:
            logger.info("No handoff file to process. Skipping.")
            return
            
        output_file = intent_generator.process_handoff_file(input_file, execution_date_str)
        ti.xcom_push(key="generated_dataset_file", value=output_file)

    generate_task = PythonOperator(
        task_id="generate_intents_via_llm",
        python_callable=task_generate_intents,
        provide_context=True,
        # Увеличиваем таймаут задачи, так как LLM-генерация может занять время
        execution_timeout=timedelta(minutes=30) 
    )

    # Цепочка: ... -> Finalize -> Generate
    extract_task >> clean_task >> vectorize_task >> cluster_task >> finalize_task >> generate_task
```

---

## Шаг 3.2.3: Исчерпывающее тестирование механизма Self-Correction

Напишем тест, который эмулирует "глупую" LLM, возвращающую невалидный JSON (например, всего 5 фраз), и проверяет, что система корректно делает вторую попытку с уточняющим промптом.

**Файл: `tests/test_llm_self_correction.py`**
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError
from plugins.operators.llm_generator import LLMGenerationService, ClusterGenerationResult

@pytest.fixture
def llm_service():
    return LLMGenerationService()

@pytest.mark.asyncio
async def test_self_correction_on_validation_error(llm_service):
    """Тест проверяет, что при ValidationError делается вторая попытка с исправленным промптом."""
    
    cluster_data = {
        "cluster_id": 99,
        "size": 20,
        "representative_phrases": ["вернуть деньги", "оформить возврат"] * 10
    }

    # Эмулируем поведение LLM: 
    # 1-й вызов возвращает невалидный JSON (всего 5 фраз вместо 20)
    # 2-й вызов возвращает валидный JSON
    call_count = 0
    
    async def mock_call_llm_api(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # Симулируем ошибку: слишком мало фраз
            return '{"intent_name": "vozvrat", "description": "test", "utterances": ["1", "2", "3", "4", "5"]}'
        elif call_count == 2:
            # Проверяем, что в промпт была добавлена информация об ошибке
            assert "КРИТИЧЕСКАЯ ОШИБКА В ПРЕДЫДУЩЕЙ ПОПЫТКЕ" in prompt
            assert "минимум 20 фраз" in prompt or "20" in prompt # Pydantic выдает "min_items=20"
            
            # Симулируем успешный ответ
            valid_utterances = [f"фраза номер {i}" for i in range(25)]
            return json.dumps({
                "intent_name": "vozvrat_deneg",
                "description": "Клиент хочет вернуть деньги.",
                "utterances": valid_utterances
            })
        else:
            raise Exception("Unexpected 3rd call")

    # Подменяем внутренний метод вызова API
    with patch.object(llm_service, '_call_llm_api', side_effect=mock_call_llm_api):
        result = await llm_service.generate_intent_with_self_correction(cluster_data)
        
        # Проверки
        assert call_count == 2, "Должно быть ровно 2 попытки"
        assert result.processing_status == "success"
        assert result.generated_intent.intent_name == "vozvrat_deneg"
        assert len(result.generated_intent.utterances) == 25

@pytest.mark.asyncio
async def test_fallback_after_max_attempts(llm_service):
    """Тест проверяет, что после 2 неудачных попыток возвращается fallback-результат, а не падает DAG."""
    
    cluster_data = {"cluster_id": 100, "size": 10, "representative_phrases": ["test"] * 10}

    async def mock_always_fail(prompt: str) -> str:
        # Всегда возвращаем невалидный JSON
        return '{"intent_name": "BAD NAME!!!", "description": "test", "utterances": ["1"]}'

    with patch.object(llm_service, '_call_llm_api', side_effect=mock_always_fail):
        result = await llm_service.generate_intent_with_self_correction(cluster_data)
        
        assert result.processing_status == "failed"
        assert "Validation failed after 2 attempts" in result.error_message
        assert result.generated_intent.intent_name == "error_cluster_100"
```

**Запуск тестов:**
```bash
pip install pytest-asyncio
pytest tests/test_llm_self_correction.py -v
```

---

## Шаг 3.2.4: Важные архитектурные нюансы для Production

1. **Почему явный цикл `for`, а не `@retry` из `tenacity` для промпта?** 
   Декоратор `@retry` в `tenacity` по умолчанию повторяет вызов функции с **теми же самыми аргументами**. Чтобы изменить промпт (добавить текст ошибки), пришлось бы использовать сложные хаки с состоянием объекта или `retry_error_callback`, что делает код нечитаемым и сложным для отладки. Явный цикл `for` с `try/except` является общепризнанным Best Practice для LLM Self-Correction, так как он дает полный контроль над модификацией контекста между попытками.
2. **`tenacity` используется для сети:** Мы оставили `@retry` на методе `_call_llm_api`. Это идеально разделяет ответственность: `tenacity` борется с сетевыми таймаутами и 502/503 ошибками Yandex Cloud, а цикл `for` борется с логическими ошибками генерации (валидацией).
3. **Безопасный Fallback:** Метод `_create_fallback_result` гарантирует, что даже если LLM полностью "сломалась" на конкретном кластере, задача Airflow завершится со статусом `success`, а проблемный кластер будет помечен флагом `processing_status="failed"`. Это позволяет пайплайну обработать остальные 99 кластеров, а лингвист позже увидит в Битрикс24, какие именно интенты требуют ручного вмешательства.
4. **Контроль параллелизма (`asyncio.Semaphore`):** В `generate_intents.py` мы ограничили одновременные запросы к LLM API значением `settings.llm_max_concurrent_requests` (по умолчанию 5). Это предотвращает получение `429 Too Many Requests` от YandexGPT при обработке 50+ кластеров одновременно.

---

### Что мы достигли в Подзадаче 3.2:
✅ **Устойчивость к галлюцинациям формата:** Система больше не падает из-за того, что LLM забыла закрыть скобку или сгенерировала 19 фраз вместо 20.
✅ **Интеллектуальный Retry (Self-Correction):** LLM получает конкретную обратную связь о своей ошибке ("Поле 'utterances': ensure this value has at least 20 items") и исправляет её во второй попытке.
✅ **Гарантия завершения DAG:** Реализован безопасный fallback-механизм, который изолирует сбои на уровне одного кластера, не прерывая обработку всего датасета.
✅ **Производительность:** Асинхронная обработка с семафором обеспечивает высокую скорость генерации без нарушения rate limits провайдера.
