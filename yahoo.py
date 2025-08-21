#!/usr/bin/env python3
"""
Загрузчик котировок с Yahoo Finance (yahoo.py)

Однофайловый CLI-скрипт для выгрузки исторических котировок с Yahoo Finance,
предобработки (TZ, ресемплинг, заполнение пропусков), расчёта индикаторов
(SMA, RSI, MACD), сохранения в CSV/JSON/Parquet и построения простого графика.

Соответствует ТЗ из предыдущего шага.

Примеры запуска:
  python yahoo.py --tickers AAPL --period 1y --interval 1d --plot --out-format csv
  python yahoo.py --tickers EURUSD=X --start 2023-01-01 --end 2024-01-01 \
      --interval 1h --tz Europe/Madrid --indicators sma,rsi --sma-window 50 --out-format parquet
  python yahoo.py --tickers AAPL,MSFT --period 5d --interval 1m --resample 1D --out-format csv --plot
  python yahoo.py --tickers BTC-USD --period 1mo --interval 1h --columns close,volume --out-format json
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# matplotlib: используем бэкэнд без GUI (на случай запуска на сервере)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# yfinance — официальный клиент к публичным данным Yahoo Finance
import yfinance as yf

__version__ = "1.0.0"

# --- Допустимые интервалы и периоды ---
ALLOWED_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo"
}
ALLOWED_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "ytd", "max"}
ALLOWED_FILL = {"none", "ffill", "bfill"}
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
ALLOWED_OUT_FORMATS = {"csv", "json", "parquet"}


@dataclass
class Args:
    tickers: List[str]
    interval: str
    period: Optional[str]
    start: Optional[str]
    end: Optional[str]
    adjust: bool
    resample: Optional[str]
    fill: str
    tz: Optional[str]
    columns: Optional[List[str]]
    intraday_hours: Optional[Tuple[str, str]]
    indicators: List[str]
    sma_window: int
    rsi_window: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    plot: bool
    out_formats: List[str]
    out_dir: str
    rate_limit: float
    retries: int
    retry_wait: float
    log_level: str


# ------------------ Вспомогательные функции ------------------

def parse_indicators(ind: Optional[str]) -> List[str]:
    """Парсим список индикаторов из строки вида "sma,rsi,macd".
    Возвращаем упорядоченный список из допустимых значений.
    """
    if not ind:
        return []
    items = [x.strip().lower() for x in ind.split(",") if x.strip()]
    valid = []
    for x in items:
        if x in {"sma", "rsi", "macd"} and x not in valid:
            valid.append(x)
    return valid


def parse_columns(cols: Optional[str]) -> Optional[List[str]]:
    """Парсим список колонок пользователя (open,high,low,close,volume,adj close)."""
    if not cols:
        return None
    mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adj_close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }
    result = []
    for raw in cols.split(","):
        key = raw.strip().lower()
        if key in mapping:
            result.append(mapping[key])
        else:
            raise ValueError(f"Неизвестная колонка: {raw}")
    # Убираем дубликаты, сохраняя порядок
    seen = set()
    out: List[str] = []
    for c in result:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def parse_intraday_hours(val: Optional[str]) -> Optional[Tuple[str, str]]:
    """Парсим диапазон часов формата HH:MM-HH:MM. Возвращаем (start, end) или None."""
    if not val:
        return None
    try:
        start, end = [x.strip() for x in val.split("-")]
        # простая валидация HH:MM
        for t in (start, end):
            hh, mm = t.split(":")
            if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                raise ValueError
        return start, end
    except Exception:
        raise ValueError("--intraday-hours должен быть в формате HH:MM-HH:MM")


# ------------------ Индикаторы ------------------

def add_sma(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.DataFrame:
    """Простая скользящая средняя (SMA)."""
    col = f"SMA_{window}"
    df[col] = df[price_col].rolling(window=window, min_periods=window).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.DataFrame:
    """RSI по методу Уайлдера (Wilder)."""
    col = f"RSI_{window}"
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing: alpha = 1/window
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[col] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int, price_col: str = "Close") -> pd.DataFrame:
    """MACD (EMA fast/slow) + сигнальная линия и гистограмма."""
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    df[f"MACD_{fast}_{slow}"] = macd
    df[f"MACD_signal_{signal}"] = sig
    df[f"MACD_hist_{fast}_{slow}_{signal}"] = hist
    return df


# ------------------ Загрузка и обработка данных ------------------

def fetch_data_once(ticker: str, args: Args) -> pd.DataFrame:
    """Одиночный запрос к Yahoo через yfinance.download с заданными параметрами."""
    kwargs = {
        "tickers": ticker,
        "interval": args.interval,
        "auto_adjust": args.adjust,  # авто-учёт сплитов/дивидендов
        "progress": False,
        "threads": False,
    }
    # Период: либо period, либо start/end
    if args.period:
        kwargs["period"] = args.period
    else:
        if args.start:
            kwargs["start"] = args.start
        if args.end:
            kwargs["end"] = args.end
    logging.debug(f"yfinance.download kwargs: {kwargs}")
    df = yf.download(**kwargs)
    # yfinance может вернуть пустой DataFrame
    if isinstance(df, pd.DataFrame) and df.empty:
        return df
    # Убедимся, что индекс — DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    # Если индекс без TZ — локализуем как UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        # На всякий случай конвертируем в UTC
        df.index = df.index.tz_convert("UTC")
    return df


def fetch_data(ticker: str, args: Args) -> pd.DataFrame:
    """Загрузка с повторами при сетевых ошибках и логированием."""
    attempt = 0
    wait = args.retry_wait
    while True:
        try:
            df = fetch_data_once(ticker, args)
            return df
        except Exception as e:
            attempt += 1
            logging.warning(f"Ошибка загрузки для {ticker}: {e}")
            if attempt > args.retries:
                logging.error(f"Провалено после {args.retries} повторов для {ticker}")
                raise
            logging.info(f"Повтор {attempt}/{args.retries} через {wait:.1f}с...")
            time.sleep(wait)
            wait *= 2  # экспоненциальная пауза


def apply_timezone_and_filter(df: pd.DataFrame, args: Args) -> pd.DataFrame:
    """Применяем локализацию TZ и фильтр по внутридневным часам, если задано."""
    if df.empty:
        return df
    # Перевод в целевую таймзону (если задана)
    if args.tz:
        try:
            df = df.tz_convert(args.tz)
        except Exception:
            # Если вдруг индекс без TZ (бывает на старых версиях) — локализуем как UTC и конвертим
            df.index = df.index.tz_localize("UTC").tz_convert(args.tz)
    # Фильтр по внутридневным часам
    if args.intraday_hours:
        start_str, end_str = args.intraday_hours
        t = df.index.time
        start_h, start_m = map(int, start_str.split(":"))
        end_h, end_m = map(int, end_str.split(":"))
        start_time = pd.Timestamp(df.index[0].date(), hour=start_h, minute=start_m, tz=df.index.tz).time()
        end_time = pd.Timestamp(df.index[0].date(), hour=end_h, minute=end_m, tz=df.index.tz).time()
        mask = (t >= start_time) & (t <= end_time)
        df = df[mask]
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Ресемплинг OHLC и объёма. Adj Close берём последним значением интервала."""
    cols = df.columns
    out = {}
    # OHLC
    if "Open" in cols:
        out["Open"] = "first"
    if "High" in cols:
        out["High"] = "max"
    if "Low" in cols:
        out["Low"] = "min"
    if "Close" in cols:
        out["Close"] = "last"
    if "Adj Close" in cols:
        out["Adj Close"] = "last"
    if "Volume" in cols:
        out["Volume"] = "sum"
    return df.resample(rule).agg(out).dropna(how="all")


def fill_missing(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """Заполнение пропусков по выбору пользователя."""
    if how == "ffill":
        return df.ffill()
    elif how == "bfill":
        return df.bfill()
    return df


def postprocess(df: pd.DataFrame, args: Args) -> pd.DataFrame:
    """Комбинированная постобработка: TZ, фильтр, ресемплинг, fill, отбор колонок."""
    if df.empty:
        return df
    # TZ и фильтр часов
    df = apply_timezone_and_filter(df, args)
    # Ресемплинг (выполняем до индикаторов)
    if args.resample:
        df = resample_ohlc(df, args.resample)
    # Заполнение пропусков
    df = fill_missing(df, args.fill)
    # Отбор колонок (пока только базовые, индикаторы добавятся позже)
    base_cols = df.columns.tolist()
    if args.columns:
        keep = [c for c in args.columns if c in base_cols]
        df = df[keep]
    return df


def compute_indicators(df: pd.DataFrame, args: Args) -> pd.DataFrame:
    """Расчёт индикаторов по выбранному списку."""
    if df.empty or not args.indicators:
        return df
    if "sma" in args.indicators:
        df = add_sma(df, args.sma_window, price_col="Close")
    if "rsi" in args.indicators:
        df = add_rsi(df, args.rsi_window, price_col="Close")
    if "macd" in args.indicators:
        df = add_macd(df, args.macd_fast, args.macd_slow, args.macd_signal, price_col="Close")
    return df


# ------------------ Сохранение и графики ------------------

def make_filename(ticker: str, args: Args, ext: str) -> str:
    """Формируем имя файла по правилам ТЗ."""
    interval = args.interval
    if args.period:
        suffix = args.period
    else:
        start = args.start or ""
        end = args.end or ""
        suffix = f"{start}_{end}".strip("_") or "range"
    safe_ticker = ticker.replace("/", "-")
    return f"{safe_ticker}_{interval}_{suffix}.{ext}"


def save_table(df: pd.DataFrame, ticker: str, args: Args) -> None:
    """Сохраняем таблицу в выбранные форматы."""
    path_base = args.out_dir.rstrip("/\\")
    filename_base = make_filename(ticker, args, ext="")[:-1]  # без точки в конце

    # Создаём каталог, если нужно
    import os
    os.makedirs(path_base, exist_ok=True)

    for fmt in args.out_formats:
        if fmt == "csv":
            path = os.path.join(path_base, f"{filename_base}.csv")
            df.to_csv(path, index=True)
            logging.info(f"CSV сохранён: {path}")
        elif fmt == "json":
            path = os.path.join(path_base, f"{filename_base}.json")
            df.to_json(path, orient="table", date_format="iso")
            logging.info(f"JSON сохранён: {path}")
        elif fmt == "parquet":
            path = os.path.join(path_base, f"{filename_base}.parquet")
            try:
                df.to_parquet(path, index=True)
            except Exception as e:
                logging.warning(f"Parquet не сохранён (нет pyarrow/fastparquet?): {e}")
                continue
            logging.info(f"Parquet сохранён: {path}")
        else:
            logging.warning(f"Неизвестный формат: {fmt}")


def plot_chart(df: pd.DataFrame, ticker: str, args: Args) -> None:
    """Простой график: линия Close + поверх SMA (если есть). MACD — одной линией в легенде.
    По ТЗ рисуем в один холст для простоты.
    """
    if df.empty:
        logging.warning("Пустой DataFrame — график не построен")
        return

    # Заготовим подписи
    title_parts = [f"{ticker} ({args.interval})"]
    if args.period:
        title_parts.append(args.period)
    else:
        title_parts.append(f"{args.start or ''}..{args.end or ''}")
    title = " ".join(title_parts)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    ax.plot(df.index, df["Close"], label="Close")

    # Найдём все SMA-колонки, чтобы добавить на график
    for c in df.columns:
        if c.startswith("SMA_"):
            ax.plot(df.index, df[c], label=c)

    # MACD — добавим как линию, если посчитан (не рисуем отдельную панель)
    macd_cols = [c for c in df.columns if c.startswith("MACD_") and "hist" not in c and "signal" not in c]
    for c in macd_cols:
        ax.plot(df.index, df[c], label=c)

    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.autofmt_xdate()

    # Сохраняем PNG
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    fname = make_filename(ticker, args, ext="png")
    path = os.path.join(args.out_dir, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"График сохранён: {path}")


# ------------------ CLI и main ------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Загрузчик котировок Yahoo Finance с обработкой и индикаторами",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", required=True, help="Тикер(ы) через запятую")

    # Периоды
    parser.add_argument("--period", choices=sorted(ALLOWED_PERIODS), help="Период пресетом")
    parser.add_argument("--start", help="Начальная дата YYYY-MM-DD")
    parser.add_argument("--end", help="Конечная дата YYYY-MM-DD")

    parser.add_argument("--interval", default="1d", choices=sorted(ALLOWED_INTERVALS), help="Таймфрейм")

    # Обработка
    parser.add_argument("--adjust", dest="adjust", action="store_true", help="Учитывать сплиты/дивиденды")
    parser.add_argument("--no-adjust", dest="adjust", action="store_false", help="Не учитывать сплиты/дивиденды")
    parser.set_defaults(adjust=True)
    parser.add_argument("--resample", help="Правило ресемплинга pandas, напр. 1D, 1H, 1W")
    parser.add_argument("--fill", default="none", choices=sorted(ALLOWED_FILL), help="Заполнение пропусков")
    parser.add_argument("--tz", help="Таймзона, напр. Europe/Madrid")
    parser.add_argument("--columns", help="Список колонок: open,high,low,close,volume,adj close")
    parser.add_argument("--intraday-hours", dest="intraday_hours", help="Фильтр часов HH:MM-HH:MM")

    # Индикаторы
    parser.add_argument("--indicators", help="Список индикаторов: sma,rsi,macd")
    parser.add_argument("--sma-window", type=int, default=20)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)

    # Вывод
    parser.add_argument("--plot", action="store_true", help="Сохранить PNG-график")
    parser.add_argument(
        "--out-format",
        action="append",
        choices=sorted(ALLOWED_OUT_FORMATS),
        help="Форматы вывода (можно несколько флагов)",
    )
    parser.add_argument("--out-dir", default="./data", help="Каталог вывода")

    # Надёжность/ограничения
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Задержка между тикерами (сек)")
    parser.add_argument("--retries", type=int, default=3, help="Кол-во повторов при ошибке сети")
    parser.add_argument("--retry-wait", type=float, default=2.0, help="Стартовая пауза перед повтором (сек)")

    # Логи и версия
    parser.add_argument("--log-level", default="INFO", choices=sorted(ALLOWED_LOG_LEVELS))
    parser.add_argument("--version", action="version", version=f"yahoo.py {__version__}")

    return parser


def validate_args(ns: argparse.Namespace) -> Args:
    """Базовая валидация и преобразование типов."""
    # Тикеры
    tickers = [t.strip() for t in ns.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("Укажите хотя бы один тикер через --tickers")

    # Период: либо period, либо start/end
    if ns.period and (ns.start or ns.end):
        raise SystemExit("Используйте либо --period, либо --start/--end, но не вместе")
    if not ns.period and not (ns.start or ns.end):
        logging.info("Не указан период — будет использован пресет по умолчанию yfinance (1mo)")

    # Колонки
    try:
        columns = parse_columns(ns.columns) if ns.columns else None
    except ValueError as e:
        raise SystemExit(str(e))

    # Часы
    try:
        intraday_hours = parse_intraday_hours(ns.intraday_hours) if ns.intraday_hours else None
    except ValueError as e:
        raise SystemExit(str(e))

    # Индикаторы
    indicators = parse_indicators(ns.indicators)

    # Форматы вывода: по умолчанию CSV, если не указано
    out_formats = ns.out_format or ["csv"]

    # Приведение лог-уровня
    log_level = ns.log_level.upper()

    return Args(
        tickers=tickers,
        interval=ns.interval,
        period=ns.period,
        start=ns.start,
        end=ns.end,
        adjust=ns.adjust,
        resample=ns.resample,
        fill=ns.fill,
        tz=ns.tz,
        columns=columns,
        intraday_hours=intraday_hours,
        indicators=indicators,
        sma_window=ns.sma_window,
        rsi_window=ns.rsi_window,
        macd_fast=ns.macd_fast,
        macd_slow=ns.macd_slow,
        macd_signal=ns.macd_signal,
        plot=ns.plot,
        out_formats=out_formats,
        out_dir=ns.out_dir,
        rate_limit=ns.rate_limit,
        retries=ns.retries,
        retry_wait=ns.retry_wait,
        log_level=log_level,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def process_ticker(ticker: str, args: Args) -> None:
    logging.info(f"Тикер: {ticker} | interval={args.interval} | period={args.period} | start={args.start} | end={args.end}")
    df = fetch_data(ticker, args)
    if df.empty:
        logging.warning(f"Пустые данные для {ticker} — пропуск")
        return

    logging.debug(f"Загружено строк: {len(df)}. Колонки: {list(df.columns)}")

    df = postprocess(df, args)
    if df.empty:
        logging.warning(f"После обработки данных ничего не осталось для {ticker}")
        return

    df = compute_indicators(df, args)

    # Сохранение таблиц
    save_table(df, ticker, args)

    # Графики (опционально)
    if args.plot:
        try:
            plot_chart(df, ticker, args)
        except Exception as e:
            logging.warning(f"Не удалось построить график для {ticker}: {e}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)

    args = validate_args(ns)
    setup_logging(args.log_level)

    logging.info("Старт yahoo.py")

    for i, ticker in enumerate(args.tickers):
        if i > 0 and args.rate_limit > 0:
            time.sleep(args.rate_limit)
        try:
            process_ticker(ticker, args)
        except Exception as e:
            logging.error(f"Ошибка обработки {ticker}: {e}")
            # продолжаем другие тикеры, но фиксируем ошибку

    logging.info("Готово")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(130)