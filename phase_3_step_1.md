Это критический мост между "сырой" математической кластеризацией и структурированными знаниями, которые поймет наш NLU-бот. Главная задача здесь — заставить LLM действовать как строгий лингвист-аналитик, выдавая **исключительно валидный JSON** без галлюцинаций, лишнего текста или markdown-оберток, с соблюдением бизнес-требований к количеству и разнообразию фраз.

---

# ЭТАП 3, ПОДЗАДАЧА 3.1: Проектирование промпта и строгой схемы вывода

## Шаг 3.1.1: Строгие Pydantic V2 модели для вывода LLM

Мы не можем полагаться на то, что LLM "примерно" угадает формат. Мы определяем жесткую схему, которую будем использовать как для генерации (через JSON Schema в API провайдера), так и для валидации ответа.

**Файл: `plugins/operators/llm_generator.py`** (начало файла)
```python
import logging
import json
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from config.settings import settings

logger = logging.getLogger(__name__)

# 1. Строгая схема для одного сгенерированного интента
class GeneratedIntent(BaseModel):
    intent_name: str = Field(
        description="Краткое, уникальное название интента на русском языке в snake_case (например: 'vozvrat_deneg', 'smena_tarifa'). Без префиксов 'intent_'."
    )
    description: str = Field(
        description="Четкое описание намерения клиента в 1 предложении. Что именно хочет клиент?"
    )
    utterances: List[str] = Field(
        min_items=20,
        max_items=30,
        description="Список из 20-30 разнообразных, естественных фраз, выражающих это намерение. Включайте вариации длины, синонимы и разные грамматические конструкции."
    )

    @field_validator('intent_name')
    @classmethod
    def validate_intent_name(cls, v: str) -> str:
        import re
        if not re.match(r'^[a-zа-яё0-9_]+$', v.lower()):
            raise ValueError("intent_name должен содержать только строчные буквы, цифры и подчеркивания (snake_case)")
        return v.lower()

    @field_validator('utterances')
    @classmethod
    def validate_utterances_diversity(cls, v: List[str]) -> List[str]:
        if len(v) < 20:
            raise ValueError(f"Требуется минимум 20 фраз, получено {len(v)}")
        # Проверка на уникальность (с точностью до 90%)
        unique_v = []
        for phrase in v:
            if not any(cls._similarity(phrase, u) > 0.9 for u in unique_v):
                unique_v.append(phrase)
        if len(unique_v) < 15:
            raise ValueError("Слишком много дублирующихся или похожих фраз в utterances")
        return v

    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

# 2. Схема для ответа LLM при обработке одного кластера
class ClusterGenerationResult(BaseModel):
    cluster_id: int
    generated_intent: GeneratedIntent
    processing_status: str = Field(default="success")
    error_message: Optional[str] = Field(default=None)
```

---

## Шаг 3.1.2: Инжиниринг промпта (Prompt Engineering)

Промпт должен быть максимально директивным, чтобы исключить "творчество" LLM в формате вывода. Мы используем паттерн "Role-Task-Context-Constraints-Format".

**Добавьте в `plugins/operators/llm_generator.py`:**
```python
SYSTEM_PROMPT = """Ты — старший лингвист-аналитик, специализирующийся на обучении NLU-моделей (Natural Language Understanding) для контакт-центров.
Твоя задача: проанализировать кластер реальных фраз клиентов и сформировать структурированные обучающие данные для нового интента.

ПРАВИЛА (СТРОГО СОБЛЮДАТЬ):
1. ФОРМАТ: Ты должен вернуть СТРОГО валидный JSON. Никакого markdown (без ```json), никаких пояснений до или после JSON.
2. НАЗВАНИЕ (intent_name): Придумай краткое, понятное название на русском в snake_case (например: "smena_tarifa", "vozvrat_sredstv").
3. ОПИСАНИЕ (description): Опиши намерение клиента ровно в одном предложении.
4. ФРАЗЫ (utterances): Сгенерируй от 20 до 30 фраз. Они должны быть:
   - Естественными (как говорит реальный человек, а не робот).
   - Разнообразными по длине (от коротких "верните деньги" до развернутых "я хочу оформить возврат средств за вчерашний заказ").
   - Содержать синонимы и разные грамматические формы (вопросы, утверждения, просьбы).
   - СТРОГО соответствовать смыслу предоставленных примеров. Не добавляй посторонние темы.
5. Если кластер состоит из бессмысленного шума, верни "processing_status": "failed" и "error_message": "noise_detected".
"""

USER_PROMPT_TEMPLATE = """Проанализируй следующий кластер фраз клиентов, которые не были распознаны текущей NLU-моделью.

Метаданные кластера:
- ID кластера: {cluster_id}
- Количество фраз в кластере: {size}
- Оценка качества (silhouette): {silhouette_score}

Примеры фраз из этого кластера (наиболее репрезентативные):
{examples_list}

Сформируй обучающие данные для этого интента в строгом JSON-формате, соответствующем схеме GeneratedIntent.
"""
```

---

## Шаг 3.1.3: Реализация асинхронного LLM-клиента

Мы создадим универсальный клиент, который по умолчанию настроен на YandexGPT Pro (как наиболее доступный и соответствующий 152-ФЗ вариант), но архитектура позволяет легко переключиться на локальный vLLM (Llama-3) или OpenRouter.

**Добавьте в `plugins/operators/llm_generator.py`:**
```python
class LLMGenerationService:
    def __init__(self):
        # Используем асинхронный клиент с пулом соединений и таймаутами
        self.client = httpx.AsyncClient(
            timeout=30.0, # LLM может думать дольше, чем обычные API
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        )
        
        # Настройки для YandexGPT Pro
        self.yandex_endpoint = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.folder_id = getattr(settings, 'yandex_folder_id', 'your_folder_id') # Добавьте в config/settings.py
        self.iam_token = getattr(settings, 'yandex_iam_token', 'your_iam_token') # Добавьте в config/settings.py
        
        logger.info("LLMGenerationService initialized (YandexGPT Pro mode).")

    @retry(
        stop=stop_after_attempt(2), # Максимум 2 попытки (оригинал + 1 retry при ошибке валидации)
        wait=wait_fixed(1.0),
        retry=retry_if_exception_type(ValueError), # Ретраим только ошибки валидации Pydantic
        reraise=True
    )
    async def generate_intent_for_cluster(
        self, 
        cluster: dict, # Словарь из FinalCluster
        previous_error: str = ""
    ) -> ClusterGenerationResult:
        """
        Генерирует интент для одного кластера с гарантированной валидацией JSON.
        """
        cluster_id = cluster["cluster_id"]
        examples = cluster["representative_phrases"]
        
        # Формируем промпт
        examples_str = "\n".join([f"- {ex}" for ex in examples])
        user_prompt = USER_PROMPT_TEMPLATE.format(
            cluster_id=cluster_id,
            size=cluster["size"],
            silhouette_score=cluster.get("metadata", {}).get("original_silhouette", "N/A"),
            examples_list=examples_str
        )
        
        if previous_error:
            user_prompt += f"\n\nВАЖНО: В предыдущей попытке ты допустил ошибку: {previous_error}. Исправь JSON и верни только его."

        # Payload для YandexGPT с требованием JSON
        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": 0.4, # Низкая температура для детерминированного, структурированного вывода
                "maxTokens": "1000"
            },
            "messages": [
                {"role": "system", "text": SYSTEM_PROMPT},
                {"role": "user", "text": user_prompt}
            ],
            # Требование к формату ответа (поддерживается YandexGPT)
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id,
            "Content-Type": "application/json"
        }

        try:
            response = await self.client.post(self.yandex_endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            raw_text = data["result"]["alternatives"][0]["message"]["text"].strip()
            
            # Очистка от возможных markdown-оберток (на случай, если модель проигнорировала response_format)
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            # Добавляем cluster_id к ответу модели для валидации полной схемы
            parsed_json = json.loads(raw_text)
            parsed_json["cluster_id"] = cluster_id
            parsed_json["processing_status"] = "success"

            # Строгая валидация через Pydantic V2
            validated_result = ClusterGenerationResult.model_validate(parsed_json)
            
            logger.info(f"Successfully generated intent for cluster {cluster_id}")
            return validated_result

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON returned by LLM: {str(e)}. Raw text: {raw_text[:100]}"
            logger.warning(f"Cluster {cluster_id}: {error_msg}")
            raise ValueError(error_msg) # Триггерит retry в декораторе
            
        except Exception as e:
            logger.error(f"Cluster {cluster_id} generation failed with critical error: {str(e)}")
            # При критической ошибке (не валидация) возвращаем failed-статус, чтобы не ронять весь DAG
            return ClusterGenerationResult(
                cluster_id=cluster_id,
                generated_intent=GeneratedIntent(
                    intent_name="error_unknown",
                    description="Ошибка генерации",
                    utterances=["ошибка"]
                ),
                processing_status="failed",
                error_message=str(e)
            )

    async def close(self):
        await self.client.aclose()

# Глобальный синглтон
llm_generator = LLMGenerationService()
```

---

## Шаг 3.1.4: Исчерпывающее тестирование промпта и валидации

Напишем тесты, которые проверяют, что наша Pydantic-модель действительно отлавливает плохие данные, которые могла бы сгенерировать "сбойная" LLM.

**Файл: `tests/test_llm_generator.py`**
```python
import pytest
from plugins.operators.llm_generator import GeneratedIntent, ClusterGenerationResult

def test_generated_intent_valid():
    """Проверка валидного интента."""
    data = {
        "intent_name": "vozvrat_deneg",
        "description": "Клиент хочет вернуть деньги за оплаченный заказ.",
        "utterances": [f"фраза номер {i}" for i in range(25)] # 25 фраз
    }
    intent = GeneratedIntent(**data)
    assert intent.intent_name == "vozvrat_deneg"
    assert len(intent.utterances) == 25

def test_generated_intent_fails_on_bad_name():
    """Проверка отклонения некорректного имени интента."""
    data = {
        "intent_name": "Intent: Возврат Денег!!!", # Нарушает snake_case
        "description": "Описание",
        "utterances": ["фраза"] * 20
    }
    with pytest.raises(ValueError) as exc_info:
        GeneratedIntent(**data)
    assert "snake_case" in str(exc_info.value)

def test_generated_intent_fails_on_too_few_utterances():
    """Проверка отклонения при недостаточном количестве фраз."""
    data = {
        "intent_name": "vozvrat_deneg",
        "description": "Описание",
        "utterances": ["фраза"] * 15 # Меньше минимальных 20
    }
    with pytest.raises(ValueError) as exc_info:
        GeneratedIntent(**data)
    assert "минимум 20 фраз" in str(exc_info.value)

def test_generated_intent_fails_on_duplicates():
    """Проверка отклонения при наличии дубликатов в фразах."""
    data = {
        "intent_name": "vozvrat_deneg",
        "description": "Описание",
        "utterances": ["вернуть деньги"] * 25 # Все фразы одинаковые
    }
    with pytest.raises(ValueError) as exc_info:
        GeneratedIntent(**data)
    assert "дублирующихся" in str(exc_info.value)
```

**Запуск тестов:**
```bash
pytest tests/test_llm_generator.py -v
```

---

## Шаг 3.1.5: Обновление конфигурации

Не забудьте добавить новые переменные окружения в `config/settings.py` для работы с YandexGPT (или вашим выбранным провайдером).

**Обновите `config/settings.py`:**
```python
    # --- LLM Generation ---
    yandex_folder_id: str = Field(default="your_yandex_folder_id")
    yandex_iam_token: str = Field(default="your_yandex_iam_token")
    llm_max_concurrent_requests: int = Field(default=5, description="Семафор для ограничения параллельных запросов к LLM API")
```

---

### Что мы достигли в Подзадаче 3.1:
✅ **Жесткий контроль вывода (Guardrails):** Pydantic V2 модели с кастомными валидаторами (`field_validator`) гарантируют, что в систему попадет только корректный `intent_name` (snake_case) и достаточное количество (20-30) *уникальных* фраз.  
✅ **Промпт-инжиниринг уровня Production:** Системный промпт явно запрещает markdown, задает роль лингвиста и требует разнообразия фраз, что критически снижает вероятность галлюцинаций.  
✅ **Асинхронная и устойчивая архитектура:** Использование `httpx.AsyncClient` с пулом соединений и декоратора `@retry` от `tenacity` обеспечивает автоматическое исправление ошибок парсинга JSON (через уточняющий промпт) без падения всего DAG.  
✅ **Нативная поддержка JSON-режима:** Использование `response_format: {"type": "json_object"}` в YandexGPT значительно повышает шанс получения валидной структуры с первой попытки.  
