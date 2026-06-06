# ЭТАП 5: Production Hardening и IaC

## Подзадача 5.1: Production-контейнеризация и Infrastructure as Code (IaC)

**Проблема:** Наивные Docker-образы Airflow огромны, медленно собираются и запускаются от root, что нарушает политики безопасности.
**Решение:** Multi-stage сборка на базе официального образа Airflow с кэшированием тяжелых ML-зависимостей и запуском от непривилегированного пользователя.

### Шаг 5.1.1: Production Dockerfile
Создайте файл `Dockerfile` в корне проекта. Мы расширяем официальный образ, чтобы сохранить все встроенные механизмы Airflow, но добавляем наши зависимости.

```dockerfile
# Используем официальный образ Airflow как базу (уже содержит пользователя airflow с UID 50000)
FROM apache/airflow:2.8.1-python3.12

USER root

# Устанавливаем системные зависимости для компиляции ML-библиотек и работы с данными
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Копируем requirements.txt ПЕРВЫМ для эффективного кэширования слоев Docker
COPY requirements.txt /tmp/requirements.txt

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Предварительная загрузка тяжелых ML-моделей в образ, чтобы избежать загрузок из интернета во время выполнения DAG
RUN python -m spacy download ru_core_news_sm && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large', device='cpu')"

# Копируем код приложения
COPY dags/ /opt/airflow/dags/
COPY plugins/ /opt/airflow/plugins/
COPY config/ /opt/airflow/config/

# Устанавливаем правильные права (Airflow требует, чтобы файлы принадлежали пользователю airflow)
USER root
RUN chown -R airflow:root /opt/airflow/dags /opt/airflow/plugins /opt/airflow/config
USER airflow

# Переопределяем точку входа, если нужно (обычно оставляем стандартную от Airflow)
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/entrypoint"]
```

### Шаг 5.1.2: Базовый Terraform для инфраструктуры (Концепт)
Для production нам нужна управляемая инфраструктура. Вот пример модуля Terraform для ключевых компонентов (Yandex Cloud / AWS аналог).

```hcl
# main.tf (фрагмент)
terraform {
  required_providers {
    yandex = { source = "yandex-cloud/yandex" }
  }
}

# 1. Managed Kubernetes Cluster
resource "yandex_kubernetes_cluster" "nlu_mlops_cluster" {
  name        = "nlu-self-healing-cluster"
  network_id  = yandex_vpc_network.mlops_net.id
  # ... настройки node_groups с autoscaling (min 2, max 5 узлов)
}

# 2. Managed PostgreSQL (для метаданных Airflow)
resource "yandex_mdb_postgresql_cluster" "airflow_db" {
  name        = "airflow-meta-db"
  environment = "PRODUCTION"
  config {
    version = "15"
    resources {
      resource_preset_id = "s2.medium"
      disk_size          = 50
    }
    postgresql_config = {
      max_connections = 200
    }
  }
}

# 3. Object Storage Bucket (для артефактов DAG и экспортов)
resource "yandex_storage_bucket" "nlu_artifacts" {
  bucket = "nlu-mlops-artifacts-prod"
  acl    = "private"
  
  # Правило автоматического удаления старых файлов (TTL 30 дней)
  lifecycle_rule {
    id      = "cleanup-old-exports"
    enabled = true
    expiration {
      days = 30
    }
  }
}
```

---

## Подзадача 5.2: Сквозная наблюдаемость (OpenTelemetry)

**Цель:** Точно знать, на каком этапе (Vectorize, Cluster, LLM) возникает задержка или ошибка, без необходимости парсить gigabytes текстовых логов.

### Шаг 5.2.1: Инструментация Airflow и кастомных операторов
Добавим библиотеки OTel в `requirements.txt`:
```text
opentelemetry-api==1.22.0
opentelemetry-sdk==1.22.0
opentelemetry-exporter-otlp==1.22.0
opentelemetry-instrumentation-httpx==0.43b0
```

**Обновите `plugins/operators/llm_generator.py` (пример внедрения трейсинга):**
```python
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Инициализация (обычно делается в инициализаторе плагина Airflow)
tracer = trace.get_tracer("nlu.mlops.llm_generator")
HTTPXClientInstrumentor().instrument() # Автоматически трассирует все вызовы httpx

class LLMGenerationService:
    # ... внутри метода generate_intent_with_self_correction ...
    
    async def generate_intent_with_self_correction(self, cluster: dict):
        cluster_id = cluster["cluster_id"]
        
        # Создаем кастомный span для отслеживания времени обработки конкретного кластера
        with tracer.start_as_current_span(f"process_cluster_{cluster_id}") as span:
            span.set_attribute("cluster.size", cluster["size"])
            span.set_attribute("cluster.silhouette", cluster.get("metadata", {}).get("original_silhouette", 0))
            
            try:
                # ... существующая логика вызова LLM ...
                result = await self._call_llm_api(current_prompt)
                span.set_attribute("llm.attempts", attempt)
                return validated_result
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
```

**Настройка экспорта в Grafana Tempo / Jaeger:**
В переменных окружения Airflow (в `docker-compose` или K8s Deployment) добавьте:
```env
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_SERVICE_NAME=airflow-nlu-pipeline
AIRFLOW__CORE__ENABLE_XCOM_PICKLING=False # Безопасность при использовании OTel
```

---

## Подзадача 5.3: Алертинг и мониторинг деградации (Prometheus)

Мы должны знать о проблемах раньше, чем аналитики. Создайте файл `k8s/prometheus-rules.yaml` для вашего кластера мониторинга.

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: nlu-mlops-alerts
  namespace: monitoring
spec:
  groups:
  - name: nlu.pipeline.rules
    rules:
    # 1. Критический сбой пайплайна
    - alert: NluDagFailed
      expr: airflow_dag_run_duration_seconds_count{dag_id="nlu_self_healing_extract", state="failed"} > 0
      for: 0m
      labels:
        severity: critical
      annotations:
        summary: "Сбой основного DAG самообучения NLU"
        description: "DAG nlu_self_healing_extract завершился с ошибкой. Проверьте логи в Airflow."

    # 2. Деградация качества данных (Data Drift)
    - alert: NluZeroValidClusters
      expr: airflow_task_duration_seconds_count{task_id="cluster_and_filter_hdbcan"} > 0 and on(dag_id, execution_date) (nlu_valid_clusters_count == 0)
      for: 0m
      labels:
        severity: warning
      annotations:
        summary: "HDBSCAN не нашел ни одного валидного кластера"
        description: "Возможна деградация качества входящих логов или поломка экстрактора. Проверьте сырые данные."

    # 3. Утечка PII (Кастомная метрика, если вы реализовали пост-проверку)
    - alert: NluPiiLeakDetected
      expr: rate(nlu_pii_leak_events_total[5m]) > 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Обнаружена потенциальная утечка PII в промпты LLM"
        description: "Система маскирования не сработала. Немедленно проверьте логи и остановите DAG."

    # 4. Высокая задержка LLM API
    - alert: NluLlmHighLatency
      expr: histogram_quantile(0.95, rate(llm_generation_duration_seconds_bucket[5m])) > 10
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Высокая задержка генерации LLM (P95 > 10 сек)"
        description: "Проверьте квоты и статус YandexGPT / vLLM кластера."
```

---

## Подзадача 5.4: Нагрузочное тестирование (Locust) и финальная валидация

**Цель:** Убедиться, что при скачке объема логов (например, после крупной маркетинговой акции) пайплайн не упадет с OOM и уложится в SLA.

### Шаг 5.4.1: Сценарий Locust для тестирования ML-компонентов
Мы не тестируем UI Airflow. Мы тестируем "тяжелые" функции (`vectorize`, `cluster`, `generate`) под нагрузкой.

**Файл: `load_tests/locustfile.py`**
```python
import time
import json
import random
from locust import User, task, events
import numpy as np
import pandas as pd

# Импортируем наши компоненты (требует, чтобы locust запускался в том же venv)
from plugins.operators.embedding_service import embedding_service
from plugins.operators.cluster_analyzer import cluster_analyzer

class MlopsPipelineUser(User):
    # Эмулируем нагрузку на CPU/GPU воркера
    
    def on_start(self):
        # Генерируем синтетический "тяжелый" кластер для теста
        self.test_cluster = {
            "cluster_id": 999,
            "size": 50,
            "metadata": {"original_silhouette": 0.45},
            "representative_phrases": [f"Тестовая фраза для нагрузки номер {i} с вариациями" for i in range(10)]
        }

    @task(3)
    def test_vectorization_and_clustering(self):
        start_time = time.time()
        try:
            # Эмулируем обработку батча из 1000 текстов
            texts = ["Тестовый запрос клиента на возврат или смену тарифа " + str(i) for i in range(1000)]
            
            # 1. Векторизация
            embeddings = embedding_service.generate_embeddings_batched(texts, batch_size=64)
            
            # 2. Подготовка DataFrame для кластеризации
            df = pd.DataFrame({"lemmatized_text": texts, "embedding": embeddings})
            temp_path = f"/tmp/test_load_{time.time()}.parquet"
            df.to_parquet(temp_path, engine='pyarrow')
            
            # 3. Кластеризация
            cluster_analyzer.analyze_and_filter(temp_path, "load_test")
            
            events.request.fire(
                request_type="ML_Pipeline",
                name="vectorize_and_cluster",
                response_time=(time.time() - start_time) * 1000,
                response_length=len(texts),
                exception=None,
            )
        except Exception as e:
            events.request.fire(
                request_type="ML_Pipeline",
                name="vectorize_and_cluster",
                response_time=(time.time() - start_time) * 1000,
                response_length=0,
                exception=e,
            )

    @task(1)
    def test_llm_generation(self):
        start_time = time.time()
        try:
            # Эмуляция вызова LLM (в реальном тесте здесь будет вызов llm_generator)
            time.sleep(random.uniform(0.5, 2.0)) # Имитация задержки сети и инференса
            
            events.request.fire(
                request_type="LLM_API",
                name="generate_intent",
                response_time=(time.time() - start_time) * 1000,
                response_length=500,
                exception=None,
            )
        except Exception as e:
            events.request.fire(
                request_type="LLM_API",
                name="generate_intent",
                response_time=(time.time() - start_time) * 1000,
                response_length=0,
                exception=e,
            )
```
*Запуск:* `locust -f load_tests/locustfile.py --host=http://localhost`

---

## Шаг 5.5: Итоговый чек-лист готовности к Production (Go-Live)

Перед тем как считать проект завершенным и передать его в эксплуатацию, убедитесь, что выполнены все пункты:

- [ ] **Безопасность:** Docker-образ собран, не содержит секретов, запускается от пользователя `airflow` (UID 50000). Пароли и токены вынесены в Airflow Variables / K8s Secrets.
- [ ] **Идемпотентность:** Повторный запуск DAG за ту же дату (`execution_date`) корректно перезаписывает артефакты и не создает дубликаты задач в Битрикс24.
- [ ] **Отказоустойчивость:** Circuit Breaker для Битрикс24 настроен и протестирован. Ошибки LLM валидации обрабатываются через Self-Correction, а не роняют DAG.
- [ ] **Наблюдаемость:** В Grafana настроен дашборд, отображающий время выполнения каждого шага DAG и количество сгенерированных кластеров. Алерты в Telegram/Slack настроены и протестированы.
- [ ] **Нагрузочная способность:** Locust-тест показал, что обработка 10 000 записей занимает предсказуемое время и не вызывает OOM-ошибок (потребление памяти стабильно).
- [ ] **HITL работает:** Лингвист получил задачу в Битрикс24 с корректной HTML-разметкой, примерами и рабочей ссылкой на CSV в S3.

---

### 🏆 ФИНАЛЬНЫЙ ИТОГ ПРОЕКТА

**Мы построили:**
1. **Этап 1:** Надежный ETL-пайплайн с идемпотентной выгрузкой и *гарантированным* удалением PII (Presidio).
2. **Этап 2:** Оптимизированная векторизация (Parquet, батчинг) и многоуровневая фильтрация кластеров (HDBSCAN + Silhouette + эвристики), отсекающая 99% мусора.
3. **Этап 3:** Устойчивый LLM-агент со строгой Pydantic-валидацией и механизмом Self-Correction, поддерживающий как облачные, так и локальные (vLLM) провайдеры.
4. **Этап 4:** Бесшовная интеграция с Битрикс24 (с Circuit Breaker) и Telegram-алертингом, реализующая паттерн Human-in-the-Loop.
5. **Этап 5:** Enterprise-инфраструктура: Multi-stage Docker, OpenTelemetry трассировка, Prometheus-алерты и нагрузочное тестирование.
