МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ
Федеральное государственное бюджетное образовательное учреждение
высшего образования «Челябинский государственный университет»
Факультет [название факультета]  |  Кафедра [название кафедры]

ДОПУСТИТЬ К ЗАЩИТЕ
Заведующий кафедрой, [учёная степень и звание]
__________ [И.О. Фамилия]   «___»___________ 2026 г.

Разработка программного комплекса для детектирования дрифта
данных и адаптивного переобучения моделей машинного обучения

ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА
ЧелГУ – 09.04.04.2026.XXX-XXX.ВКР

Научный руководитель, [должность, учёная степень]
__________ [И.О. Фамилия]

Автор работы, студент группы [номер группы]
__________ [И.О. Фамилия]

Нормоконтролёр _____________ [И.О. Фамилия]
«___»___________ 2026 г.

Челябинск, 2026 г.

-- 1 of 8 --

Аннотация

This thesis presents an open-source MLOps platform that unifies data
drift detection, automated retraining and model lifecycle management in a
single deployable system. The platform implements three statistical tests
(PSI, Kolmogorov–Smirnov, Chi-square), a configurable threshold policy,
an action chain (logging, webhook alerting, automated retraining), a
scikit-learn-based retraining pipeline with stratified hold-out evaluation,
and a lifecycle registry with stage promotion (development → staging →
production → archived). The back-end is Python (FastAPI + scikit-learn +
SQLite); the front-end is a React/MUI single-page dashboard; deployment
is via Docker Compose. Validated on two real datasets (Adult Census
Income and Kaggle Credit Card Fraud) under six drift scenarios,
confirming correct policy triggering, safe model promotion and complete
data-to-model provenance.

ОГЛАВЛЕНИЕ

ВВЕДЕНИЕ .............................................................................................. 3
1. ИССЛЕДОВАНИЕ ПРЕДМЕТНОЙ ОБЛАСТИ .............................. 4
2. РАЗРАБОТКА АРХИТЕКТУРЫ СИСТЕМЫ ................................. 5
3. РАЗРАБОТКА И ВНЕДРЕНИЕ СИСТЕМЫ ................................... 6
4. ЭКСПЕРИМЕНТАЛЬНАЯ ВАЛИДАЦИЯ ...................................... 7
ЗАКЛЮЧЕНИЕ ....................................................................................... 8
СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ ............................ 8

-- 2 of 8 --

ВВЕДЕНИЕ

Актуальность

Machine learning models deployed in production rarely maintain their
training-time accuracy. Input distributions shift over time due to changing
user behaviour, upstream pipeline modifications, seasonality and external
events. This phenomenon — data drift — silently degrades model quality
and, in regulated domains such as banking and fraud detection, can have
significant financial and reputational consequences. The need for
automated drift detection and adaptive retraining is therefore one of the
central problems of contemporary MLOps practice.

Existing solutions are strong but fragmented: Evidently AI and NannyML
specialise in monitoring; MLflow and Weights & Biases in model registries
and experiment tracking; Apache Airflow and Prefect in workflow
orchestration. Combining these tools into a coherent end-to-end loop
requires significant integration code and is fragile to maintain. A unified,
open-source platform covering the full drift-to-retraining-to-promotion
cycle in a single coherent stack would substantially reduce this operational
overhead and is the contribution of the present work.

Постановка задачи

Целью работы является разработка программного комплекса для
детектирования дрифта данных и адаптивного переобучения моделей
машинного обучения. Задачи: (1) анализ предметной области;
(2) проектирование архитектуры платформы; (3) реализация
механизмов детектирования дрифта, оркестрации, переобучения и
управления жизненным циклом; (4) разработка REST API и
веб-панели мониторинга; (5) экспериментальная валидация на
реальных наборах данных.

Структура работы: четыре главы, заключение и список источников.

-- 3 of 8 --

1. ИССЛЕДОВАНИЕ ПРЕДМЕТНОЙ ОБЛАСТИ

In supervised learning, a model learns f : X → Y on a training sample.
Performance degrades when P_train(X, Y) ≠ P_prod(X, Y). Distribution
shift manifests as covariate shift (P(X) changes), concept drift (P(Y|X)
changes) or prior probability shift (P(Y) changes). Covariate shift — data
drift — is the most common and operationally tractable form, since it can
be detected from input features alone without production labels.

Statistical foundations. Three tests cover tabular ML effectively:
(i) Population Stability Index (PSI) on binned distributions, with
interpretation bands < 0.1 (stable), 0.1–0.25 (moderate), ≥ 0.25 (high);
(ii) two-sample Kolmogorov–Smirnov (KS) on numeric empirical CDFs,
significance threshold α = 0.05; (iii) χ² test of homogeneity on categorical
features, with rare categories merged into «__other__» to satisfy minimum
expected-count requirements. Distance-based alternatives (Wasserstein,
MMD, KL-divergence) offer advantages for high-dimensional or non-
tabular data; for the tabular use case of this work the PSI/KS/χ² trio
offers the best balance of interpretability, established thresholds and
computational efficiency.

Таблица 1 — Сопоставление существующих инструментов

Инструмент          | Дрейф | Реестр | Оркестрация | Переобучение | Open-source
Evidently AI        |  да   |  нет   |     нет      |     нет      | да
NannyML             |  да   |  нет   |     нет      |     нет      | да
MLflow              |  нет* |  да    |     нет      |     нет      | да
Weights & Biases    |  нет* |  да    |     нет      |     нет      | нет
Apache Airflow      |  нет  |  нет   |     да       |     нет      | да
Fiddler / Aporia    |  да   |  да    |     да       |     частично | нет
Настоящая работа    |  да   |  да    |     да       |     да       | да
* плагины существуют, но не входят в ядро продукта.

The primary gap is the absence of a unified open-source platform covering
drift detection, policy-driven orchestration, automated retraining and
lifecycle management in a single deployable stack. The present work
addresses this gap.

-- 4 of 8 --

2. РАЗРАБОТКА АРХИТЕКТУРЫ СИСТЕМЫ

Функциональные требования: (1) вычислять per-feature PSI, KS и χ²
и формировать агрегированный отчёт; (2) хранить замороженный
базовый профиль (PSI-бины, списки признаков) в JSON; (3) оценивать
пороговую политику и формировать список причин срабатывания;
(4) при срабатывании выполнять цепочку действий: логирование →
опциональный webhook → автоматическое переобучение; (5) обучать
sklearn-пайплайн на объединённой выборке, оценивать на holdout,
версионировать артефакт и продвигать в production только при не-
ухудшении macro-F1; (6) предоставлять REST API и SPA-дашборд.

Нефункциональные требования: Python 3.11, FastAPI, scikit-learn;
React/MUI SPA; SQLite для всех хранилищ; Docker Compose с тремя
сервисами (api, scheduler, frontend); детерминированные операции при
фиксированном random_state.

Архитектура. Платформа состоит из пяти Python-пакетов:
drift_detection — статистические тесты и базовый профиль;
orchestration — Orchestrator, политика, действия, RunStore;
retraining — пайплайн обучения, ModelRegistry, правило продвижения;
lifecycle — LifecycleService, стадии, указатель production;
data_management — версии наборов данных, снимки базового профиля,
таблица провенанса. Поверх них — REST API (FastAPI) и SPA (React).

Взаимодействие модулей. Один запуск оркестрации: (1) загрузка
конфигурации и данных сценария; (2) загрузка или построение
базового профиля; (3) run_drift_analysis → DriftReport;
(4) DriftThresholdPolicy.evaluate → (triggered, reasons); (5) цепочка
действий; (6) при RetrainPipelineAction — обучение, регистрация в
ModelRegistry и lifecycle.db, запись провенанса; (7) RunStore.insert_run.

Пороги политики (по умолчанию): n_features_high_psi > 0;
n_numeric_ks_significant > 2; n_categorical_chi2_significant > 3.

-- 5 of 8 --

3. РАЗРАБОТКА И ВНЕДРЕНИЕ СИСТЕМЫ

Технологический стек. Back-end: Python 3.11, FastAPI, scikit-learn 1.4,
scipy 1.13, pandas 2.2, joblib, SQLite, uvicorn. Front-end: TypeScript,
React, Vite, MUI, axios, react-router-dom, @mui/x-data-grid, recharts.
Развёртывание: Docker Compose (api : 8000, scheduler, frontend : 8090),
общий том ./artifacts.

Детектирование дрифта. BaselineProfile замораживает квантильные
границы (10 бинов) при инициализации и сохраняется в JSON.
run_drift_analysis итерирует по численным (PSI + KS) и категориальным
(PSI + χ²) признакам, собирает построчный DataFrame и агрегированный
summary-словарь. Редкие категории объединяются в «__other__»
перед χ²-тестом. PSI считается по формуле Σ(Q_i − P_i)·ln(Q_i/P_i) с
клиппингом ε = 1e-6.

Оркестрация. Orchestrator.run_pipeline поддерживает шесть сценариев
(random_holdout, age_shift, incoming_csv, fraud_d1_vs_d2, fraud_d2_vs_d3,
fraud_d1_vs_d3). Расписание запускается командой python -m
src.orchestration serve и обёрнуто сервисом «scheduler» в Docker Compose.
Переменные окружения управляют интервалом, webhook-адресом и
флагом авто-переобучения.

Переобучение. run_retrain_pipeline строит sklearn Pipeline
(ColumnTransformer + HistGradientBoostingClassifier: max_depth = 6,
learning_rate = 0.08, max_iter = 200, random_state = 42), выполняет
стратифицированное разбиение 80/20, обучает, вычисляет accuracy /
macro-F1 / precision / recall. Новая версия продвигается в production
только если macro-F1 ≥ macro-F1 текущего чемпиона.

REST API и дашборд. 13 эндпоинтов: GET /api/overview, GET
/api/orchestration/runs, POST /api/orchestration/check-once, POST
/api/retraining/run, GET/POST /api/lifecycle/*, GET /api/data/*,
POST /api/inference/predict. SPA: 5 страниц (Overview, Workflows,
Models, Data, Inference).

-- 6 of 8 --

4. ЭКСПЕРИМЕНТАЛЬНАЯ ВАЛИДАЦИЯ

Таблица 2 — Характеристики наборов данных

Набор данных        | Записей | Признаки              | Целевой класс
Adult Census Income |  48 842 | 14 (6 чис. + 8 кат.)  | доход >50K (≈ 24%)
Credit Card Fraud   | 284 807 | 29 (V1–V28 + Amount)  | мошенничество (≈ 0.17%)

Стратегия. Шесть сценариев: random_holdout и age_shift (Adult);
incoming_csv (Adult, внешний CSV); fraud_d1_vs_d2, fraud_d2_vs_d3,
fraud_d1_vs_d3 (Fraud, темпоральные окна). Базовый профиль Adult
фиксируется по reference из random_holdout; базовый профиль Fraud —
по окну D1. Все split-операции: random_state = 42.

Таблица 3 — Сводка счётчиков дрейфа

Сценарий          | n_high_psi | n_ks_sig | n_chi2_sig | Триггер
random_holdout    |     0      |   0–2    |    0–1     | нет
age_shift         |     4      |    5     |     5      | да
fraud_d1_vs_d2    |     3      |    7     |    н/п     | да
fraud_d2_vs_d3    |     5      |    9     |    н/п     | да
fraud_d1_vs_d3    |     8      |   12     |    н/п     | да

Дрейф на Fraud-окнах монотонно усиливается с увеличением
темпорального расстояния. На random_holdout единичные значимые
тесты (0–2 KS, 0–1 χ²) — ожидаемый статистический шум при α = 0.05;
политика не срабатывает, поскольку пороги допускают до 2 и 3
значимых результатов соответственно.

Таблица 4 — Метрики переобученных моделей (holdout 20%)

Сценарий              | accuracy | macro-F1 | n_train | Продвинуто
random_holdout (Adult)|  0.866   |  0.819   |  39 074 | да
age_shift (Adult)     |  0.866   |  0.818   |  43 075 | нет*
fraud_retrain_d1_d2   |  0.9994  |  0.917   | 151 898 | да

* macro-F1 < чемпиона → production-указатель не изменён. Версия
зарегистрирована в реестре со стадией development.

Комплексное тестирование подтвердило: для каждого из шести
сценариев run-запись сохраняется в orchestration.db; при срабатывании
политики создаётся ровно одна новая версия в lifecycle.db и одна
запись провенанса в data_management.db; production-указатель
монотонен по macro-F1.

-- 7 of 8 --

ЗАКЛЮЧЕНИЕ

В рамках работы разработан программный комплекс, реализующий
полный цикл: детектирование дрифта (PSI, KS, χ²) → пороговая
политика → автоматическое переобучение → версионирование и
продвижение модели → провенанс данных и экспериментов. Цель и все
задачи выполнены. Экспериментальная валидация на двух реальных
наборах данных в шести сценариях подтвердила корректность
срабатывания политики и безопасность правила продвижения.

Направления дальнейшего развития: (1) обучаемые политики,
учитывающие бизнес-издержки ошибок; (2) детектирование дрейфа
для нетабличных данных (embedding drift); (3) интеграция с
промышленными системами оркестрации (Airflow, Prefect) и
реестрами моделей (MLflow); (4) staging-канареечное развёртывание
и human-in-the-loop подтверждение в качестве расширенных действий
политики.

СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ

1. Sculley D. et al. Hidden Technical Debt in Machine Learning Systems
   // Proc. NeurIPS. — 2015. — P. 2503–2511.
2. Huyen C. Designing Machine Learning Systems. — O'Reilly, 2022.
3. Siddiqi N. Credit Risk Scorecards. — Wiley, 2005.
4. Massey F. J. The KS Test for Goodness of Fit // JASA. — 1951. —
   Vol. 46. — P. 68–78.
5. Rabanser S. et al. Failing Loudly // Proc. NeurIPS. — 2019. —
   P. 1396–1408.
6. Zaharia M. et al. Accelerating the ML Lifecycle with MLflow //
   IEEE Data Engineering Bulletin. — 2018. — Vol. 41, № 4. — P. 39–45.
7. Pedregosa F. et al. Scikit-learn: ML in Python // JMLR. — 2011. —
   Vol. 12. — P. 2825–2830.
8. Amershi S. et al. Software Engineering for ML: A Case Study //
   Proc. ICSE-SEIP. — 2019. — P. 291–300.

-- 8 of 8 --
