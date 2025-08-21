# Загрузчик котировок с Yahoo Finance

📈 Репозиторий содержит Python-скрипт [`yahoo.py`](./yahoo.py) для загрузки исторических котировок с Yahoo Finance.  
Скрипт позволяет выгружать данные по акциям, индексам, валютным парам (Forex), криптовалютам и производить базовую обработку:  
- учёт сплитов и дивидендов,  
- ресемплинг таймфреймов,  
- фильтрация по внутридневным часам,  
- расчёт технических индикаторов (SMA, RSI, MACD),  
- сохранение данных в форматы **CSV, JSON, Parquet**,  
- построение простых графиков в формате PNG.  

---

## ⚡ Возможности
- Загрузка котировок для одного или нескольких тикеров.
- Поддержка периодов (`1d, 1mo, 1y, max`) и дат (`--start/--end`).
- Поддержка интервалов от минутных (`1m`) до квартальных (`3mo`).
- Автоматическая корректировка сплитов/дивидендов (`--adjust`).
- Ресемплинг данных (например, агрегация минуток в дневные свечи).
- Заполнение пропусков методами `ffill`/`bfill`.
- Локализация в выбранную таймзону (`--tz Europe/Moscow`).
- Фильтрация данных по торговым часам (например, `--intraday-hours 09:00-17:30`).
- Индикаторы:
  - SMA (скользящая средняя),
  - RSI (Relative Strength Index),
  - MACD.
- Сохранение данных в **CSV**, **JSON**, **Parquet**.
- Автоматическая генерация графиков (PNG).

---

## 🛠️ Установка

```bash
git clone https://github.com/FX84/Yahoo-Finance.git
cd Yahoo-Finance
python3 -m venv venv
source venv/bin/activate   # для Linux/MacOS
venv\\Scripts\\activate    # для Windows
pip install -r requirements.txt
````

---

## 🚀 Использование

### Общий синтаксис

```bash
python yahoo.py --tickers <ТИКЕР[,ТИКЕР2,...]> [опции]
```

### Примеры

#### 1. Загрузка годовых дневных данных по AAPL и построение графика

```bash
python yahoo.py --tickers AAPL --period 1y --interval 1d --plot --out-format csv
```

#### 2. Форекс EUR/USD за год, часовые данные, локализация в Madrid, с индикаторами SMA и RSI

```bash
python yahoo.py --tickers EURUSD=X --start 2023-01-01 --end 2024-01-01 \
  --interval 1h --tz Europe/Madrid --indicators sma,rsi --sma-window 50 --out-format parquet
```

#### 3. Два тикера (AAPL, MSFT), агрегация минуток в дневные свечи

```bash
python yahoo.py --tickers AAPL,MSFT --period 5d --interval 1m --resample 1D --out-format csv --plot
```

#### 4. Биткоин, сохранение только колонок Close и Volume в JSON

```bash
python yahoo.py --tickers BTC-USD --period 1mo --interval 1h --columns close,volume --out-format json
```

---

## 📂 Структура вывода

* Данные сохраняются в каталог `./data` (по умолчанию или указанный через `--out-dir`).

* Формат имени файлов:

  ```
  {ТИКЕР}_{ИНТЕРВАЛ}_{ПЕРИОД или START_END}.{расширение}
  ```

  Примеры:

  * `AAPL_1d_1y.csv`
  * `EURUSD=X_1h_2023-01-01_2024-01-01.parquet`
  * `BTC-USD_1h_1mo.json`

* Графики сохраняются в том же каталоге в PNG.

---

## ⚙️ Аргументы командной строки

* `--tickers` — тикеры через запятую (**обязательно**).
* `--period` — готовый период (например, `1y`).
* `--start`, `--end` — даты выборки (альтернатива `--period`).
* `--interval` — интервал данных (`1m`, `1d`, `1wk` и т.д.).
* `--adjust / --no-adjust` — учитывать или нет сплиты/дивиденды.
* `--resample` — правило ресемплинга (например, `1D`, `1H`).
* `--fill` — метод заполнения пропусков (`none`, `ffill`, `bfill`).
* `--tz` — таймзона для данных.
* `--columns` — список колонок для сохранения.
* `--intraday-hours` — фильтр часов торгов (например, `09:00-17:30`).
* `--indicators` — список индикаторов (`sma,rsi,macd`).
* `--sma-window`, `--rsi-window`, `--macd-fast/slow/signal` — параметры индикаторов.
* `--plot` — построить график.
* `--out-format` — форматы вывода (можно несколько).
* `--out-dir` — каталог сохранения файлов.
* `--rate-limit` — пауза между загрузками тикеров.
* `--retries`, `--retry-wait` — параметры повторных запросов.
* `--log-level` — уровень логов (`DEBUG, INFO, WARNING, ERROR`).
* `--version` — версия скрипта.

---

## 📜 Лицензия

Проект распространяется по лицензии [MIT](./LICENSE).
