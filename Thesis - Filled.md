МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ
Федеральное государственное бюджетное образовательное учреждение
высшего образования
«Челябинский государственный университет»
Факультет [название факультета]
Кафедра [название кафедры]

Разработка программного комплекса для детектирования дрифта
данных и адаптивного переобучения моделей машинного обучения

ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА
ЧелГУ – 09.04.04.2026.XXX-XXX.ВКР

Научный руководитель,
[должность, учёная степень и звание]                       [И.О. Фамилия]

Автор работы,
студент группы [номер группы]                              [И.О. Фамилия]

Челябинск, 2026 г.

ОГЛАВЛЕНИЕ

ВВЕДЕНИЕ ................................................................................................. 2
1. ИССЛЕДОВАНИЕ ПРЕДМЕТНОЙ ОБЛАСТИ ................................. 3
2. РАЗРАБОТКА АРХИТЕКТУРЫ СИСТЕМЫ .................................... 4
3. РАЗРАБОТКА И ВНЕДРЕНИЕ СИСТЕМЫ ..................................... 5
4. ЭКСПЕРИМЕНТАЛЬНАЯ ВАЛИДАЦИЯ ....................................... 6
ЗАКЛЮЧЕНИЕ ........................................................................................ 7
СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ .............................. 8

-- 1 of 8 --

2

ВВЕДЕНИЕ

Актуальность. The deployment of machine learning models in
production has accelerated over the past decade. Models trained on
historical data face inevitable distribution shifts in production, leading to
silent performance degradation. This mismatch between training and
production distributions — formally P_train(X, Y) ≠ P_prod(X, Y) — is a
fundamental challenge in production ML. Data drift (covariate shift) is the
most common and detectable form, and a key advantage of input-drift
detection is that it can be performed without production labels, which are
often unavailable or delayed in real-world systems.

Постановка задачи. The core problem is the absence of a unified, open-
source platform that automatically monitors production data for
distributional shifts, evaluates drift severity using statistical methods,
triggers automated retraining when critical drift is detected, and manages
the complete model lifecycle from detection to deployment. Existing
solutions (Evidently AI, NannyML, MLflow, Airflow, Prefect) cover
individual concerns, but combining them requires extensive custom
integration, manual cross-platform configuration, and ongoing maintenance,
all of which increase operational overhead and the risk of delayed
responses to drift.

Объект и предмет исследования. The object is production machine
learning systems exposed to data drift. The subject is methods and
architectures for unified, automated drift detection, adaptive retraining
and model lifecycle management.

Методы исследования. Statistical hypothesis testing (Population
Stability Index, Kolmogorov–Smirnov, Chi-square), supervised learning
on tabular data (gradient boosting), software architecture decomposition
into bounded components with explicit interfaces, and empirical evaluation
on real-world datasets (Adult Census Income, Kaggle Credit Card Fraud).

Научная новизна. A unified single-host MLOps platform is proposed
that joins drift detection, threshold-policy evaluation, automated
retraining and model lifecycle management into a single deterministic
workflow with end-to-end provenance from data to deployed model.

Практическая значимость. The platform is released as open-source
code and a Docker-Compose-deployable system, suitable for educational
demonstration, prototyping and small-team production use.

Структура работы. The thesis consists of an introduction, four chapters,
a conclusion and a list of sources. Total volume: 8 pages.

-- 2 of 8 --

3

1. ИССЛЕДОВАНИЕ ПРЕДМЕТНОЙ ОБЛАСТИ

Деградация моделей в эксплуатации. Production ML models degrade
through three main mechanisms: covariate shift (input distribution
changes while P(Y|X) is preserved), concept drift (the input–output
mapping itself changes), and prior probability shift (the marginal target
distribution changes). Temporal patterns include sudden, gradual,
incremental and recurring drift. Consequences span financial losses (fraud
models), reduced engagement (recommenders), safety risks (healthcare,
autonomous systems). Production monitoring is constrained by labelled-
data scarcity, high feature dimensionality, real-time efficiency
requirements, and the cost asymmetry between false positives
(unnecessary retraining) and false negatives (silent degradation).

Подходы к мониторингу распределений. Approaches fall into
statistical testing (PSI, KS, Chi-square), distance-based methods
(Wasserstein, MMD, KL-divergence, total variation), model-based
methods (two-sample classifiers, performance monitoring, density-ratio
estimation), and streaming approaches (adaptive windowing, change-
point detection). Statistical tests dominate industrial practice because
they are interpretable, computationally efficient and well-established.

Сопоставление платформ. Evidently AI and NannyML excel at drift
detection but lack automated retraining and orchestration. MLflow and
Weights & Biases excel at experiment tracking and model registry but
provide only ad-hoc drift detection. Apache Airflow and Prefect provide
orchestration but no built-in drift detection. Alibi Detect is a detection
library without lifecycle or orchestration. Fiddler AI and Aporia are
commercial and proprietary. None provide a fully integrated open-source
solution.

Методологическая база. PSI compares binned proportions:
PSI = Σ (q_i − p_i) · ln(q_i / p_i), with industry-standard bands
< 0.10 stable, 0.10–0.25 moderate, ≥ 0.25 high. The two-sample
Kolmogorov–Smirnov test compares empirical CDFs