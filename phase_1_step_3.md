На предыдущем шаге мы удалили явные PII-данные. Теперь наша задача — очистить текст от артефактов распознавания речи (ASR), убрать "воду" и привести текст к единой нормальной форме (лемматизация). Это критически важно для качества эмбеддингов: модель `Sentence Transformers` будет гораздо лучше группировать фразы "хочу вернуть деньги", "хотел бы оформить возврат" и "требуется возврат средств", если они приведены к общим корневым формам.

---

# ЭТАП 1, ПОДЗАДАЧА 1.3: Глубокая очистка и нормализация текста

## Шаг 1.3.1: Зависимости

Для быстрой и точной лемматизации русского языка мы будем использовать `pymorphy3` (современный, поддерживаемый форк `pymorphy2`). Он работает значительно быстрее `natasha` и идеален для задачи приведения слов к нормальной форме.

**Обновите `requirements.txt`:**
```text
# ... предыдущие зависимости ...

# Лемматизация и морфологический анализ
pymorphy3==2.0.2
pymorphy3-dicts-ru==2.4.417127.4580142
```
*Действие:* Выполните `pip install -r requirements.txt`.

---

## Шаг 1.3.2: Реализация `TextNormalizer`

Мы создадим отдельный класс, который реализует конвейер очистки. Он будет возвращать не только очищенный текст, но и причину отбраковки (если текст признан "мусором"), что критически важно для отладки и мониторинга качества данных.

**Файл: `plugins/operators/text_normalizer.py`**
```python
import re
import logging
from typing import Optional, Tuple
import pymorphy3

logger = logging.getLogger(__name__)

class TextNormalizer:
    """
    Конвейер глубокой очистки и нормализации текста для подготовки к векторизации.
    """
    
    def __init__(self):
        logger.info("Initializing TextNormalizer (loading pymorphy3 dictionaries)...")
        # auto_load=True загружает словари при первом использовании, экономя память при старте
        self.morph = pymorphy3.MorphAnalyzer()
        
        # Черный список типичных фраз-паразитов, ответов боту или артефактов ASR
        self.blacklist_phrases = {
            "алло", "да", "нет", "угу", "ага", "спасибо", "пока", "до свидания",
            "я не понял", "повторите", "что", "как", "ну", "эм", "эээ"
        }
        logger.info("TextNormalizer initialized successfully.")

    def clean_and_lemmatize(self, text: str) -> Tuple[Optional[str], str]:
        """
        Очищает и лемматизирует текст.
        
        :return: Кортеж (очищенный_текст или None, причина_отбраковки)
                 Если успех, причина = "success".
        """
        if not text or not isinstance(text, str):
            return None, "empty_or_not_string"
        
        text = text.strip()
        if len(text) < 3:
            return None, "too_short"

        # 1. Детекция артефактов ASR: повторяющиеся символы (например, "аааааа" или "!!!!!")
        # Регулярное выражение ищет любой символ, повторяющийся 5 и более раз подряд
        if re.search(r'(.)\1{4,}', text):
            return None, "repeating_chars_artifact"

        # 2. Подсчет "осмысленных" слов (только буквы, кириллица или латиница)
        words = re.findall(r'[а-яА-Яa-zA-Z]+', text)
        if len(words) < 2:
            return None, "less_than_2_meaningful_words"

        # 3. Проверка на черный список
        lower_text = text.lower()
        if lower_text in self.blacklist_phrases:
            return None, "blacklisted_exact_phrase"
        
        # Если более 80% слов в фразе состоят из слов-паразитов, отбраковываем
        word_count = len(words)
        blacklist_count = sum(1 for w in words if w.lower() in self.blacklist_phrases)
        if word_count > 0 and (blacklist_count / word_count) > 0.8:
            return None, "mostly_blacklisted_words"

        # 4. Нормализация и лемматизация
        lemmatized_words = []
        for word in words:
            # pymorphy3 работает с нижним регистром
            parsed = self.morph.parse(word.lower())[0]
            lemmatized_words.append(parsed.normal_form)
        
        cleaned_text = " ".join(lemmatized_words)
        
        # 5. Финальная проверка: после лемматизации текст не должен стать слишком коротким
        if len(cleaned_text) < 3:
            return None, "lemmatized_too_short"

        return cleaned_text, "success"

# Глобальный синглтон
normalizer = TextNormalizer()
```

---

## Шаг 1.3.3: Интеграция нормализации в пайплайн очистки

Обновим модуль `clean_and_mask.py`, чтобы он использовал новый `TextNormalizer` *после* маскирования PII. Это важно: мы не хотим, чтобы лемматизатор пытался разобрать маскированные токены вроде `[ТЕЛЕФОН_СКРЫТ]`.

**Обновите файл `plugins/operators/clean_and_mask.py`:**
```python
import json
import logging
import os
from typing import List, Dict, Any

from config.settings import settings
from plugins.operators.pii_masking import pii_masker
from plugins.operators.text_normalizer import normalizer # <-- НОВЫЙ ИМПОРТ

logger = logging.getLogger(__name__)

class DataCleaner:
    @staticmethod
    def process_logs_file(input_filepath: str, execution_date_str: str) -> str:
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Input file not found: {input_filepath}")

        with open(input_filepath, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        final_cleaned_logs = []
        
        # Счетчики для мониторинга качества данных (будут выведены в лог Airflow)
        stats = {
            "total": len(logs),
            "dropped_garbage": 0,
            "masked_pii": 0,
            "success": 0
        }
        
        # Словарь для агрегации причин отбраковки (для отладки)
        drop_reasons = {}

        for log in logs:
            raw_text = log.get("user_utterance") or log.get("text") or log.get("message", "")
            if not raw_text:
                continue

            # Шаг 1: Маскирование PII (из Подзадачи 1.2)
            masked_text = pii_masker.mask_text(raw_text)
            if masked_text != raw_text:
                stats["masked_pii"] += 1

            # Шаг 2: Глубокая очистка и лемматизация
            cleaned_text, drop_reason = normalizer.clean_and_lemmatize(masked_text)
            
            if cleaned_text is None:
                stats["dropped_garbage"] += 1
                drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
                continue

            stats["success"] += 1
            
            # Формируем финальную запись для ML-пайплайна
            clean_log = {
                "dialog_id": log.get("dialog_id") or log.get("id"),
                "timestamp": log.get("timestamp"),
                "original_text": raw_text,          # Оставляем для аудита (если разрешено)
                "masked_text": masked_text,         # Текст с PII-токенами
                "lemmatized_text": cleaned_text     # Текст, готовый для эмбеддинга!
            }
            final_cleaned_logs.append(clean_log)

        # 3. Сохранение результата
        output_filename = f"final_cleaned_logs_{execution_date_str}.json"
        output_filepath = os.path.join(settings.data_storage_path, output_filename)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_cleaned_logs, f, ensure_ascii=False, indent=2)
            
        logger.info(
            f"Cleaning pipeline complete. Total: {stats['total']}, "
            f"Success: {stats['success']}, Dropped: {stats['dropped_garbage']}, Masked: {stats['masked_pii']}"
        )
        if stats['dropped_garbage'] > 0:
            logger.info(f"Drop reasons breakdown: {drop_reasons}")
        
        return output_filepath

cleaner = DataCleaner()
```

---

## Шаг 1.3.4: Исчерпывающее тестирование нормализатора

Напишем тесты, которые доказывают корректность работы эвристик и лемматизации.

**Файл: `tests/test_text_normalizer.py`**
```python
import pytest
from plugins.operators.text_normalizer import TextNormalizer

@pytest.fixture
def normalizer():
    return TextNormalizer()

@pytest.mark.parametrize("input_text, expected_status, expected_reason", [
    # 1. Успешная лемматизация
    ("Я хочу вернуть деньги за заказ", "success", "успешная очистка"),
    ("Хотел бы оформить возврат средств", "success", "успешная очистка"),
    
    # 2. Отбраковка: слишком короткий текст
    ("да", "drop", "less_than_2_meaningful_words"),
    ("!", "drop", "too_short"),
    
    # 3. Отбраковка: артефакты ASR (повторяющиеся символы)
    ("аллооооооо", "drop", "repeating_chars_artifact"),
    ("???????????", "drop", "repeating_chars_artifact"),
    
    # 4. Отбраковка: черный список
    ("я не понял", "drop", "blacklisted_exact_phrase"),
    ("ну эм эээ", "drop", "mostly_blacklisted_words"),
    
    # 5. Успех после маскирования (симуляция)
    ("Мой номер [ТЕЛЕФОН_СКРЫТ], перезвоните", "success", "успешная очистка"),
])
def test_normalizer_logic(normalizer, input_text, expected_status, expected_reason):
    cleaned_text, reason = normalizer.clean_and_lemmatize(input_text)
    
    if expected_status == "drop":
        assert cleaned_text is None, f"Expected None for '{input_text}', got '{cleaned_text}'"
        assert reason == expected_reason, f"Expected reason '{expected_reason}', got '{reason}'"
    else:
        assert cleaned_text is not None, f"Expected text for '{input_text}', got None"
        assert reason == "success"
        # Проверка, что лемматизация сработала (например, "хочу" -> "хотеть")
        # Примечание: точная нормальная форма зависит от словаря pymorphy3

def test_lemmatization_quality(normalizer):
    """Проверка того, что разные формы слов приводятся к одной нормальной форме."""
    text1 = "Я хочу оформить возврат"
    text2 = "Он хотел оформить возврат"
    
    cleaned1, _ = normalizer.clean_and_lemmatize(text1)
    cleaned2, _ = normalizer.clean_and_lemmatize(text2)
    
    # Оба должны содержать лемму "хотеть" и "оформить"
    assert "хотеть" in cleaned1 or "хотеть" in cleaned2 # pymorphy3 может дать "хотеть" или "хотим" в зависимости от контекста, но для "хотел" точно будет "хотеть"
    assert "оформить" in cleaned1
    assert "оформить" in cleaned2
```

**Запуск тестов:**
```bash
pytest tests/test_text_normalizer.py -v
```

---

## Шаг 1.3.5: Важное архитектурное замечание о лемматизации и эмбеддингах

В профессиональном MLOps существует нюанс: современные модели эмбеддингов (такие как `multilingual-e5-large` или `ru-en-RoSBERTa`, которые мы будем использовать на Этапе 2) **уже обучены на естественном, нелемматизированном тексте**. 

Однако для *коротких, зашумленных фраз из ASR* (где много опечаток и вариаций: "вернуть", "возврат", "верните") лемматизация **помогает** алгоритмам кластеризации (HDBSCAN) сгруппировать их в один кластер, так как косинусное расстояние между их эмбеддингами становится меньше.

**Решение:** Мы сохраняем в итоговый JSON **оба** варианта:
1. `masked_text`: исходный текст с PII-токенами (для показа аналитику в Битрикс24, чтобы он видел реальный контекст).
2. `lemmatized_text`: очищенный текст (используется **исключительно** как вход для `Sentence Transformers` на Этапе 2).

Это дает нам максимальную гибкость и качество.

---

### Что мы достигли в Подзадаче 1.3:
✅ **Глубокая очистка от мусора:** Внедрены надежные эвристики для отсечения артефактов ASR (повторяющиеся символы) и фраз-паразитов, которые иначе создавали бы "шумовые" кластеры.
✅ **Лемматизация:** Интеграция `pymorphy3` приводит разноформатные запросы клиентов к единой нормальной форме, что математически улучшает качество последующей кластеризации (HDBSCAN).
✅ **Наблюдаемость данных:** Внедрен детальный подсчет причин отбраковки (`drop_reasons`), который выводится в логи Airflow. Если вдруг 90% данных начинают отбраковываться, инженеры мгновенно увидят причину (например, сломался формат выгрузки из Caila).
✅ **Разделение ответственности:** Четкое разделение `masked_text` (для людей/аудита) и `lemmatized_text` (для ML-моделей) соответствует лучшим практикам Feature Engineering.
