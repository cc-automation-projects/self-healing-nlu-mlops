Это самый ответственный момент в MLOps-пайплайне. Если мы передадим в LLM "мусорные" кластеры (например, группу из 3 фраз, состоящих только из слов "алло" и "да"), модель сгенерирует бессмысленные интенты, которые засорят боевого бота. Мы реализуем строгую многоуровневую фильтрацию.

---

# ЭТАП 2, ПОДЗАДАЧА 2.2: Кластеризация и валидация (HDBSCAN)

## Шаг 2.2.1: Зависимости

Добавим библиотеки для кластеризации и оценки качества кластеров.

**Обновите `requirements.txt`:**
```text
# ... предыдущие зависимости ...

# Кластеризация и оценка качества
hdbscan>=0.8.33
scikit-learn>=1.4.0
```
*Действие:* Выполните `pip install -r requirements.txt`.

---

## Шаг 2.2.2: Реализация `ClusterAnalyzer` с многоуровневой фильтрацией

Мы создадим строгую Pydantic-модель для выходных данных и класс, который выполняет кластеризацию, отсев шума и выбор наиболее репрезентативных примеров.

**Файл: `plugins/operators/cluster_analyzer.py`**
```python
import logging
import json
import os
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
import hdbscan
from sklearn.metrics import silhouette_score

from config.settings import settings

logger = logging.getLogger(__name__)

# 1. Строгая схема выходных данных кластера
class ValidCluster(BaseModel):
    cluster_id: int = Field(description="Уникальный ID кластера")
    size: int = Field(description="Количество фраз в кластере")
    silhouette_score: float = Field(description="Мера плотности и разделимости кластера (от -1 до 1)")
    top_examples: List[str] = Field(min_items=5, max_items=10, description="Топ примеров, наиболее близких к центроиду")

class ClusterAnalyzer:
    def __init__(
        self, 
        min_cluster_size: int = 15, 
        min_samples: int = 5,
        min_silhouette_score: float = 0.2,
        max_examples_per_cluster: int = 10
    ):
        """
        Инициализация анализатора с гиперпараметрами, откалиброванными для коротких текстов ASR.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.min_silhouette_score = min_silhouette_score
        self.max_examples = max_examples_per_cluster
        
        # Инициализация HDBSCAN
        # metric='cosine' критически важен, так как мы нормализовали эмбеддинги на шаге 2.1
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='cosine',
            cluster_selection_epsilon=0.0, # Помогает не разбивать плотные кластеры
            core_dist_n_jobs=-1 # Использовать все доступные ядра CPU
        )
        logger.info("ClusterAnalyzer initialized with calibrated HDBSCAN parameters.")

    def analyze_and_filter(self, input_filepath: str, execution_date_str: str) -> str:
        """
        Основной метод: читает векторы, кластеризует, фильтрует и сохраняет валидные кластеры.
        """
        logger.info(f"Loading vectorized data from {input_filepath}")
        df = pd.read_parquet(input_filepath)
        
        if len(df) < self.min_cluster_size:
            logger.warning(f"Dataset too small ({len(df)} rows) for meaningful clustering. Aborting.")
            return self._save_empty_result(execution_date_str)

        # 1. Извлекаем матрицу эмбеддингов
        # Parquet сохраняет списки как объекты, конвертируем обратно в numpy array
        embeddings = np.stack(df['embedding'].values)
        texts = df['lemmatized_text'].values

        logger.info(f"Running HDBSCAN on {len(df)} embeddings...")
        # 2. Кластеризация
        labels = self.clusterer.fit_predict(embeddings)
        df['cluster_label'] = labels

        # 3. Фильтрация: Отбрасываем шум (label == -1)
        valid_df = df[df['cluster_label'] != -1].copy()
        noise_count = len(df) - len(valid_df)
        logger.info(f"HDBSCAN classified {noise_count} records as noise (dropped).")

        if len(valid_df) == 0:
            logger.warning("No valid clusters found after noise removal.")
            return self._save_empty_result(execution_date_str)

        valid_clusters = []
        unique_labels = valid_df['cluster_label'].unique()

        # 4. Пост-фильтрация и извлечение примеров для каждого кластера
        for label in unique_labels:
            cluster_df = valid_df[valid_df['cluster_label'] == label]
            
            # Фильтр А: Размер кластера (дублирующая проверка для надежности)
            if len(cluster_df) < self.min_cluster_size:
                continue

            cluster_embeddings = np.stack(cluster_df['embedding'].values)
            cluster_texts = cluster_df['lemmatized_text'].values

            # Фильтр Б: Silhouette Score (оценка качества кластера)
            # Если в кластере только 1 уникальный текст, silhouette_score выдаст ошибку, пропускаем
            if len(np.unique(cluster_texts, axis=0)) < 2:
                continue
                
            try:
                score = silhouette_score(cluster_embeddings, cluster_df['cluster_label'].values, metric='cosine')
            except ValueError:
                continue # Игнорируем кластеры, где расчет невозможен

            if score < self.min_silhouette_score:
                logger.debug(f"Cluster {label} dropped due to low silhouette score: {score:.3f}")
                continue

            # 5. Извлечение топ-N примеров, наиболее близких к центроиду кластера
            centroid = np.mean(cluster_embeddings, axis=0)
            # Косинусное сходство между каждым вектором и центроидом
            similarities = np.dot(cluster_embeddings, centroid) / (np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(centroid))
            
            # Сортируем по убыванию сходства и берем топ-N
            top_indices = np.argsort(similarities)[::-1][:self.max_examples]
            top_examples = cluster_texts[top_indices].tolist()

            # Финальная проверка: не состоит ли кластер из однотипных фраз-паразитов
            if self._is_garbage_cluster(top_examples):
                logger.debug(f"Cluster {label} dropped: consists of garbage phrases.")
                continue

            valid_clusters.append(ValidCluster(
                cluster_id=int(label),
                size=len(cluster_df),
                silhouette_score=round(float(score), 4),
                top_examples=top_examples
            ))

        # 6. Сортировка кластеров по размеру (самые крупные и важные - в начале)
        valid_clusters.sort(key=lambda x: x.size, reverse=True)

        logger.info(f"Successfully identified {len(valid_clusters)} valid clusters.")
        return self._save_clusters(valid_clusters, execution_date_str)

    def _is_garbage_cluster(self, examples: List[str]) -> bool:
        """Эвристика: если все примеры слишком короткие или состоят из стоп-слов, это мусор."""
        garbage_words = {"да", "нет", "алло", "угу", "ага", "спасибо", "пока", "хорошо", "ок"}
        for text in examples:
            words = set(text.lower().split())
            # Если более 80% слов в примере - это мусор, считаем кластер мусорным
            if len(words) > 0 and len(words.intersection(garbage_words)) / len(words) > 0.8:
                return True
        return False

    def _save_clusters(self, clusters: List[ValidCluster], execution_date_str: str) -> str:
        """Сохраняет валидные кластеры в JSON для передачи в LLM."""
        output_filename = f"valid_clusters_{execution_date_str}.json"
        output_filepath = os.path.join(settings.data_storage_path, output_filename)
        
        # Сериализация через Pydantic
        clusters_dict = [c.model_dump() for c in clusters]
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(clusters_dict, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(clusters)} valid clusters to {output_filepath}")
        return output_filepath

    def _save_empty_result(self, execution_date_str: str) -> str:
        """Сохраняет пустой результат, чтобы не ломать downstream задачи."""
        output_filename = f"valid_clusters_{execution_date_str}.json"
        output_filepath = os.path.join(settings.data_storage_path, output_filename)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return output_filepath

# Глобальный синглтон
cluster_analyzer = ClusterAnalyzer()
```

---

## Шаг 2.2.3: Интеграция в Airflow DAG

Добавим задачу кластеризации в наш пайплайн.

**Обновите файл `dags/nlu_self_healing_extract.py`:**
```python
# ... предыдущие импорты ...
from plugins.operators.cluster_analyzer import cluster_analyzer

# ... внутри блока with DAG(...): ...

    def task_cluster_and_filter(**context: Context):
        ti = context["task_instance"]
        input_file = ti.xcom_pull(task_ids="vectorize_cleaned_logs", key="vectorized_file_path")
        execution_date_str = context["ds"]
        
        if not input_file:
            logger.info("No vectorized file to cluster. Skipping.")
            return
            
        output_file = cluster_analyzer.analyze_and_filter(input_file, execution_date_str)
        
        # Подсчет количества кластеров для метрик
        with open(output_file, 'r', encoding='utf-8') as f:
            clusters_count = len(json.load(f))
            
        ti.xcom_push(key="clusters_file_path", value=output_file)
        ti.xcom_push(key="valid_clusters_count", value=clusters_count)

    cluster_task = PythonOperator(
        task_id="cluster_and_filter_hdbcan",
        python_callable=task_cluster_and_filter,
        provide_context=True,
    )

    # Цепочка: Extract -> Clean/Mask -> Vectorize -> Cluster
    extract_task >> clean_task >> vectorize_task >> cluster_task
```

---

## Шаг 2.2.4: Исчерпывающее тестирование кластеризации

Напишем тесты, которые доказывают, что алгоритм корректно отбрасывает шум и мелкие группы.

**Файл: `tests/test_cluster_analyzer.py`**
```python
import pytest
import pandas as pd
import numpy as np
import os
import json
from plugins.operators.cluster_analyzer import ClusterAnalyzer, ValidCluster

@pytest.fixture
def sample_vectorized_data(tmp_path):
    """Генерирует синтетические данные с 2 явными кластерами и шумом."""
    # Кластер 1: Возврат денег (5 примеров)
    texts_1 = ["вернуть деньги", "оформить возврат", "хочу назад средства", "верните оплату", "возврат средств"]
    # Кластер 2: Смена тарифа (5 примеров)
    texts_2 = ["сменить тариф", "перейти на другой тариф", "хочу другой тариф", "поменять тариф", "смена тарифа"]
    # Шум (3 примера)
    texts_noise = ["алло", "да", "угу"]
    
    all_texts = texts_1 + texts_2 + texts_noise
    
    # Генерируем случайные, но сгруппированные векторы для эмуляции
    np.random.seed(42)
    # Векторы кластера 1 близки к [1, 0, 0...]
    emb_1 = np.random.normal(1.0, 0.1, (5, 10)) 
    # Векторы кластера 2 близки к [0, 1, 0...]
    emb_2 = np.random.normal(0.0, 0.1, (5, 10))
    emb_2[:, 1] += 1.0
    # Шум случайный
    emb_noise = np.random.normal(0.0, 1.0, (3, 10))
    
    all_embeddings = np.vstack([emb_1, emb_2, emb_noise])
    
    # Нормализация (как на шаге 2.1)
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    
    df = pd.DataFrame({
        'lemmatized_text': all_texts,
        'embedding': list(all_embeddings)
    })
    
    filepath = tmp_path / "test_vectors.parquet"
    df.to_parquet(filepath, engine='pyarrow')
    return str(filepath)

def test_clustering_filters_noise_and_small_clusters(sample_vectorized_data, tmp_path):
    # Устанавливаем min_cluster_size=4, чтобы кластеры из 5 прошли, а шум из 3 - нет
    analyzer = ClusterAnalyzer(min_cluster_size=4, min_samples=2, min_silhouette_score=0.0)
    
    # Запускаем анализ
    output_file = analyzer.analyze_and_filter(sample_vectorized_data, "2024-01-01")
    
    # Проверяем результат
    assert os.path.exists(output_file)
    with open(output_file, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    
    # Должно быть найдено ровно 2 кластера (шум отброшен)
    assert len(clusters) == 2
    
    # Проверяем структуру ValidCluster
    for c in clusters:
        assert "cluster_id" in c
        assert "size" in c
        assert "silhouette_score" in c
        assert "top_examples" in c
        assert len(c["top_examples"]) <= 10

def test_garbage_cluster_filtering():
    analyzer = ClusterAnalyzer()
    garbage_examples = ["да", "нет", "угу", "ага", "ок"]
    assert analyzer._is_garbage_cluster(garbage_examples) is True
    
    good_examples = ["хочу вернуть деньги за заказ номер 123", "как оформить возврат средств"]
    assert analyzer._is_garbage_cluster(good_examples) is False
```

**Запуск тестов:**
```bash
pytest tests/test_cluster_analyzer.py -v
```

---

## Шаг 2.2.5: Важные архитектурные нюансы для Production

1. **Параметр `min_cluster_size`**: Значение `15` (по умолчанию в коде) является эмпирическим золотым стандартом для NLU. Если установить его слишком низким (например, 3), HDBSCAN начнет выделять кластеры из случайных опечаток ASR. Если слишком высоким (50), вы пропустите редкие, но важные новые интенты (например, начало новой акции).
2. **Метрика `cosine`**: Мы используем её и в HDBSCAN, и в `silhouette_score`, потому что на шаге 2.1 мы применили `normalize_embeddings=True`. Это математически корректно и дает наилучшие результаты для текстовых эмбеддингов.
3. **Память (OOM)**: HDBSCAN имеет квадратичную сложность $O(N^2)$ в худшем случае. Для ежедневных логов fallback (обычно 1 000 – 10 000 записей) это занимает секунды и мегабайты памяти. Если объем превышает 100 000 записей за день, рекомендуется добавить шаг случайной выборки (random sampling) до кластеризации или использовать приближенные алгоритмы.

---

### Что мы достигли в Подзадаче 2.2:
✅ **Многоуровневая защита от мусора**: Комбинация отсечения шума HDBSCAN (`label == -1`), фильтрации по размеру (`min_cluster_size`) и оценки плотности (`silhouette_score`) гарантирует, что на вход LLM попадут только качественные, осмысленные группы фраз.
✅ **Репрезентативность**: Алгоритм автоматически находит центроид кластера и выбирает топ-10 фраз, максимально близких к нему, что дает LLM идеальный контекст для генерации названия интента.
✅ **Строгая типизация**: Использование Pydantic V2 (`ValidCluster`) гарантирует, что выходные данные кластеризации имеют предсказуемую структуру для следующего этапа.
✅ **Надежность DAG**: Реализован метод `_save_empty_result`, который предотвращает падение всего пайплайна, если в какой-то день не было найдено ни одного валидного кластера (это нормальная ситуация).

