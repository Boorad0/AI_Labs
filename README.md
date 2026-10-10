# Лабораторные работы

ФИО: *Лебедев Даниил*
Группа: *БИб-23Э1*
Дисциплина: *Искусственный интеллект и защита информации*

Преподаватель: *Мазур Вадим Анатольевич*

## Выполненные работы

|Работа|Тема|Статус|
|-|-|-|
|[Лабораторная №1](./lab01/)|Тестирование локальной LLM (LM Studio): проверка подключения, набор промптов, задержка ответа, оценка стабильности|Выполнена|
|[Лабораторная №2](./lab02/)|Влияние параметров генерации LLM (system prompt, temperature, max\_tokens) на качество и стабильность ответов|Выполнена|
|[Лабораторная №3](./lab03/)|Проектирование промптов и системных инструкций<br />для больших языковых моделей|Выполнена|
|[Лабораторная №4](./lab04/)|Оценка качества больших языковых моделей|Выполнена|
|[Лабораторная №5](./lab05/)|Развертывание и исследование локальной большой языковой модели|Выполнена|
|[Лабораторная №6](./lab06/)|Развертывание и исследование локальной большой языковой модели.Качество, производительность, стоимость и безопасность|Выполнена|
|[Лабораторная №7](./lab07/)|Эмбеддинги и семантический поиск. Векторные представления текста, сходство и поиск по смыслу|Выполнена|

## Структура репозитория

```text
AI_Labs/
│
├── README.md
│
├── lab01/
│   ├── .env.example
│   ├── main.py
│   ├── README.md
│   ├── requirements.txt
│   └── results.json
│
├── lab02/
│   ├── .env.example
│   ├── main.py
│   ├── README.md
│   ├── requirements.txt
│   └── results.json
│
├── lab03/
│   ├── \prompts
│   ├── .env.example
│   ├── main.py
│   ├── README.md
│   ├── requirements.txt
│   ├── results.json
│   └── tests.json
│
├── lab04/
│   ├── .env.example
│   ├── prompts.json
│   ├── tests.json
│   ├── requirements.txt
│   ├── README.md
│   ├── main.py
│   └── results.json
│
├── lab05/
│   ├── .env.example
│   ├── prompts.json
│   ├── requirements.txt
│   ├── README.md
│   ├── src/
│   │   ├── client.py
│   │   └── analyze_log.py
│   ├── data/
│   │   └── sample_auth.log
│   ├── results/
│   │   ├── results.json
│   │   ├── benchmark.csv
│   │   ├── examples.md
│   │   └── analysis.json
│   └── screenshots/
│ 
├── lab06/
│   ├── .env.example
│   ├── requirements.txt
│   ├── README.md
│   ├── src/
│   │   ├── benchmark.py
│   │   ├── providers.py
│   │   ├── metrics.py
│   │   └── charts.py
│   ├── data/
│   │   └── test_cases.json
│   ├── results/
│   │   ├── raw_results.csv
│   │   ├── summary.csv
│   │   ├── summary_by_type.csv
│   │   ├── config.json
│   │   └── repeats.csv
│   └── screenshots/
│
├── lab07/
│   ├── .env.example
│   ├── requirements.txt
│   ├── README.md
│   ├── src/
│   │   ├── common.py
│   │   ├── build_index.py
│   │   ├── search.py
│   │   └── evaluate.py
│   ├── data/
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   ├── long_docs.jsonl
│   │   └── long_queries.jsonl
│   ├── results/
│   │   ├── index_info.json
│   │   ├── search_results.csv
│   │   └── metrics.json
│   │
│   └── screenshots/
└── .gitignore
```

## Как запустить любую из работ

Инструкция запуска и все детали (используемые технологии, ход решения,
эксперименты, результаты и выводы) описаны в README.md соответствующей
лабораторной работы.

## Примечание про секреты

В репозитории **нет** файлов `.env` с реальными ключами — только
`.env.example` с образцом переменных без значений. Реальные ключи
(если появятся) хранятся только локально и в `.gitignore`.

