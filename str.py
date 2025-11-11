import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Структура данных для хранения информации о сделке"""
    entry_time: datetime      # Время входа
    exit_time: datetime       # Время выхода
    direction: str            # 'long' или 'short'
    entry_price: float        # Цена входа
    exit_price: float         # Цена выхода
    stop_loss: float          # Уровень стоп-лосса
    take_profit: float        # Уровень тейк-профита
    profit_loss: float        # Прибыль/убыток в пунктах
    profit_loss_pct: float    # Прибыль/убыток в процентах
    exit_reason: str          # Причина выхода: 'TP', 'SL', 'Signal'
    commission_paid: float    # Комиссия за сделку


class TechnicalAnalysis:
    """
    Базовый класс для технического анализа и бэктестинга.
    
    УЛУЧШЕНИЯ В ЭТОЙ ВЕРСИИ:
    - Добавлены комиссии на вход и выход
    - Комиссия учитывается при расчёте прибыли/убытка
    - Более детальная статистика с учётом комиссий
    
    Этот класс содержит:
    - Методы расчёта технических индикаторов (RSI, EMA, Engulfing)
    - Базовую инфраструктуру для бэктестинга
    - Методы управления позициями и расчёта результатов
    
    Наследуйте этот класс для создания конкретных торговых стратегий.
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0, 
                 commission_rate: float = 0.0008):
        """
        Инициализация класса технического анализа.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame с колонками: datetime, open, high, low, close, volume
        initial_capital : float
            Начальный капитал для бэктестинга
        commission_rate : float
            Комиссия за сделку в долях (0.0008 = 0.08% = 8 базисных пунктов)
            Комиссия списывается дважды: на вход и на выход
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate  # ✨ НОВОЕ: комиссия
        
        # Список всех завершённых сделок
        self.trades: List[Trade] = []
        
        # Информация о текущей открытой позиции
        self.current_position: Optional[Dict] = None
        
        # Конвертируем datetime в datetime объект если это строка
        if 'datetime' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
    # ============================================================
    # РАЗДЕЛ 1: РАСЧЁТ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ
    # ============================================================
    
    def calculate_ema(self, period: int, price_column: str = 'close') -> pd.Series:
        """
        Расчёт экспоненциальной скользящей средней (EMA).
        
        EMA даёт больший вес последним ценам по формуле:
        EMA(t) = α * Price(t) + (1 - α) * EMA(t-1)
        где α = 2 / (period + 1)
        
        Parameters:
        -----------
        period : int
            Период для расчёта EMA
        price_column : str
            Название колонки с ценами (по умолчанию 'close')
            
        Returns:
        --------
        pd.Series
            Серия значений EMA
        """
        return self.data[price_column].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period: int = 14, price_column: str = 'close') -> pd.Series:
        """
        Расчёт индекса относительной силы (RSI).
        
        RSI показывает перекупленность/перепроданность актива:
        RSI = 100 - (100 / (1 + RS))
        где RS = Среднее_значение_роста / Среднее_значение_падения
        
        RSI > 70 — перекупленность
        RSI < 30 — перепроданность
        
        Parameters:
        -----------
        period : int
            Период для расчёта RSI (стандартно 14)
        price_column : str
            Название колонки с ценами
            
        Returns:
        --------
        pd.Series
            Серия значений RSI (от 0 до 100)
        """
        # Вычисляем изменения цен
        delta = self.data[price_column].diff()
        
        # Разделяем на прибыли и убытки
        gain = delta.where(delta > 0, 0)  # Положительные изменения
        loss = -delta.where(delta < 0, 0)  # Отрицательные изменения (берём модуль)
        
        # Считаем экспоненциальные средние для прибылей и убытков
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # Относительная сила
        rs = avg_gain / avg_loss
        
        # Формула RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def detect_bullish_engulfing(self) -> pd.Series:
        """
        Обнаружение паттерна Bullish Engulfing (бычье поглощение).
        
        Условия паттерна:
        1. Предыдущая свеча медвежья (close < open)
        2. Текущая свеча бычья (close > open)
        3. Тело текущей свечи полностью поглощает тело предыдущей:
           - open(текущая) <= close(предыдущая)
           - close(текущая) >= open(предыдущая)
        
        Returns:
        --------
        pd.Series
            Boolean серия, True — обнаружен Bullish Engulfing
        """
        # Текущая и предыдущая свечи
        curr_open = self.data['open']
        curr_close = self.data['close']
        prev_open = self.data['open'].shift(1)
        prev_close = self.data['close'].shift(1)
        
        # Условие 1: предыдущая свеча медвежья
        prev_bearish = prev_close < prev_open
        
        # Условие 2: текущая свеча бычья
        curr_bullish = curr_close > curr_open
        
        # Условие 3: поглощение
        engulfing = (curr_open <= prev_close) & (curr_close >= prev_open)
        
        # Все условия должны выполняться одновременно
        return prev_bearish & curr_bullish & engulfing
    
    def detect_bearish_engulfing(self) -> pd.Series:
        """
        Обнаружение паттерна Bearish Engulfing (медвежье поглощение).
        
        Условия паттерна:
        1. Предыдущая свеча бычья (close > open)
        2. Текущая свеча медвежья (close < open)
        3. Тело текущей свечи полностью поглощает тело предыдущей:
           - open(текущая) >= close(предыдущая)
           - close(текущая) <= open(предыдущая)
        
        Returns:
        --------
        pd.Series
            Boolean серия, True — обнаружен Bearish Engulfing
        """
        curr_open = self.data['open']
        curr_close = self.data['close']
        prev_open = self.data['open'].shift(1)
        prev_close = self.data['close'].shift(1)
        
        # Условие 1: предыдущая свеча бычья
        prev_bullish = prev_close > prev_open
        
        # Условие 2: текущая свеча медвежья
        curr_bearish = curr_close < curr_open
        
        # Условие 3: поглощение
        engulfing = (curr_open >= prev_close) & (curr_close <= prev_open)
        
        return prev_bullish & curr_bearish & engulfing
    
    def add_indicators(self, ema_periods: List[int] = [25, 50, 200], rsi_period: int = 14):
        """
        Добавляет все индикаторы в DataFrame.
        
        Parameters:
        -----------
        ema_periods : List[int]
            Список периодов для EMA (по умолчанию [25, 50, 200])
        rsi_period : int
            Период для RSI (по умолчанию 14)
        """
        # Добавляем EMA разных периодов
        for period in ema_periods:
            self.data[f'ema_{period}'] = self.calculate_ema(period)
        
        # Добавляем RSI
        self.data['rsi'] = self.calculate_rsi(rsi_period)
        
        # Добавляем паттерны поглощения
        self.data['bullish_engulfing'] = self.detect_bullish_engulfing()
        self.data['bearish_engulfing'] = self.detect_bearish_engulfing()
        
        print(f"✅ Индикаторы добавлены: EMA{ema_periods}, RSI({rsi_period}), Engulfing")
    
    # ============================================================
    # РАЗДЕЛ 2: ИНФРАСТРУКТУРА БЭКТЕСТИНГА С КОМИССИЯМИ
    # ============================================================
    
    def open_position(self, index: int, direction: str, entry_price: float,
                     stop_loss: float, take_profit: float):
        """
        Открытие торговой позиции.
        
        ✨ УЧИТЫВАЕМ КОМИССИЮ НА ВХОД
        
        Parameters:
        -----------
        index : int
            Индекс строки в DataFrame
        direction : str
            Направление: 'long' или 'short'
        entry_price : float
            Цена входа
        stop_loss : float
            Уровень стоп-лосса
        take_profit : float
            Уровень тейк-профита
        """
        self.current_position = {
            'index': index,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': self.data.iloc[index]['datetime']
        }
    
    def close_position(self, index: int, exit_price: float, exit_reason: str):
        """
        Закрытие текущей позиции и запись результата в список сделок.
        
        ✨ УЧИТЫВАЕМ КОМИССИЮ НА ВХОД И ВЫХОД
        
        Формула расчёта:
        ----------------
        Для LONG:
            profit_loss = (exit_price - entry_price) - (entry_price * 2 * commission)
        Для SHORT:
            profit_loss = (entry_price - exit_price) - (entry_price * 2 * commission)
        
        Комиссия списывается ДВАЖДЫ: на вход и на выход.
        
        Parameters:
        -----------
        index : int
            Индекс строки выхода
        exit_price : float
            Цена выхода
        exit_reason : str
            Причина выхода ('TP', 'SL', 'Signal', 'End')
        """
        if self.current_position is None:
            return
        
        pos = self.current_position
        direction = pos['direction']
        entry_price = pos['entry_price']
        
        # ✨ РАСЧЁТ КОМИССИИ
        # Комиссия = цена_входа × commission_rate × 2 (вход + выход)
        commission_total = entry_price * self.commission_rate * 2
        
        # Расчёт прибыли/убытка в пунктах (БЕЗ комиссии)
        if direction == 'long':
            profit_loss_gross = exit_price - entry_price
        else:  # short
            profit_loss_gross = entry_price - exit_price
        
        # ✨ ЧИСТАЯ ПРИБЫЛЬ/УБЫТОК (вычитаем комиссию)
        profit_loss_net = profit_loss_gross - commission_total
        
        # Расчёт прибыли/убытка в процентах
        profit_loss_pct = (profit_loss_net / entry_price) * 100
        
        # Создаём объект сделки
        trade = Trade(
            entry_time=pos['entry_time'],
            exit_time=self.data.iloc[index]['datetime'],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=pos['stop_loss'],
            take_profit=pos['take_profit'],
            profit_loss=profit_loss_net,  # ✨ Чистая прибыль
            profit_loss_pct=profit_loss_pct,
            exit_reason=exit_reason,
            commission_paid=commission_total  # ✨ Сохраняем комиссию
        )
        
        # Обновляем капитал (предполагаем, что торгуем всем капиталом)
        self.current_capital *= (1 + profit_loss_pct / 100)
        
        # Сохраняем сделку
        self.trades.append(trade)
        
        # Сбрасываем текущую позицию
        self.current_position = None
    
    def check_exit_conditions(self, index: int) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Проверка условий выхода из позиции (стоп-лосс и тейк-профит).
        
        Parameters:
        -----------
        index : int
            Индекс текущей строки
            
        Returns:
        --------
        Tuple[bool, Optional[float], Optional[str]]
            (должны_ли_выйти, цена_выхода, причина)
        """
        if self.current_position is None:
            return False, None, None
        
        row = self.data.iloc[index]
        pos = self.current_position
        
        if pos['direction'] == 'long':
            # Проверка стоп-лосса (low пробил уровень)
            if row['low'] <= pos['stop_loss']:
                return True, pos['stop_loss'], 'SL'
            
            # Проверка тейк-профита (high достиг уровня)
            if row['high'] >= pos['take_profit']:
                return True, pos['take_profit'], 'TP'
        
        else:  # short
            # Проверка стоп-лосса (high пробил уровень)
            if row['high'] >= pos['stop_loss']:
                return True, pos['stop_loss'], 'SL'
            
            # Проверка тейк-профита (low достиг уровня)
            if row['low'] <= pos['take_profit']:
                return True, pos['take_profit'], 'TP'
        
        return False, None, None
    
    # ============================================================
    # РАЗДЕЛ 3: ГЕНЕРАЦИЯ СИГНАЛОВ (переопределяется в наследнике)
    # ============================================================
    
    def generate_signal(self, index: int) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Генерация торгового сигнала.
        
        ЭТОТ МЕТОД ДОЛЖЕН БЫТЬ ПЕРЕОПРЕДЕЛЁН В НАСЛЕДУЮЩЕМ КЛАССЕ!
        
        Parameters:
        -----------
        index : int
            Индекс текущей строки
            
        Returns:
        --------
        Tuple[Optional[str], Optional[float], Optional[float]]
            (направление, стоп_лосс, тейк_профит) или (None, None, None)
        """
        raise NotImplementedError("Метод generate_signal должен быть реализован в дочернем классе!")
    
    # ============================================================
    # РАЗДЕЛ 4: ЗАПУСК БЭКТЕСТА
    # ============================================================
    
    def run_backtest(self):
        """
        Запуск бэктестинга стратегии.
        
        Проходит по всем барам данных и:
        1. Проверяет условия выхода из существующей позиции
        2. Генерирует новые сигналы на вход
        3. Открывает новые позиции
        """
        print("=" * 60)
        print("🚀 ЗАПУСК БЭКТЕСТИНГА")
        print("=" * 60)
        print(f"Начальный капитал: ${self.initial_capital:.2f}")
        print(f"Комиссия: {self.commission_rate * 100:.3f}% (на вход + выход)")
        print(f"Количество баров: {len(self.data)}")
        print(f"Период: {self.data.iloc[0]['datetime']} - {self.data.iloc[-1]['datetime']}")
        print("=" * 60)
        
        # Проходим по всем барам
        for i in range(len(self.data)):
            # Пропускаем первые бары, пока не накопятся данные для индикаторов
            if i < 200:  # Минимум для EMA(200)
                continue
            
            # Шаг 1: Проверяем условия выхода, если есть открытая позиция
            if self.current_position is not None:
                should_exit, exit_price, exit_reason = self.check_exit_conditions(i)
                
                if should_exit:
                    self.close_position(i, exit_price, exit_reason)
                    continue  # Переходим к следующему бару
            
            # Шаг 2: Генерируем сигналы на вход, если нет открытой позиции
            if self.current_position is None:
                direction, stop_loss, take_profit = self.generate_signal(i)
                
                if direction is not None:
                    entry_price = self.data.iloc[i]['close']
                    self.open_position(i, direction, entry_price, stop_loss, take_profit)
        
        # Закрываем позицию, если она осталась открытой в конце
        if self.current_position is not None:
            last_index = len(self.data) - 1
            exit_price = self.data.iloc[last_index]['close']
            self.close_position(last_index, exit_price, 'End')
        
        print(f"\n✅ Бэктестинг завершён! Всего сделок: {len(self.trades)}")
    
    # ============================================================
    # РАЗДЕЛ 5: АНАЛИЗ РЕЗУЛЬТАТОВ С УЧЁТОМ КОМИССИЙ
    # ============================================================
    
    def calculate_statistics(self) -> Dict:
        """
        Расчёт статистики по результатам бэктестинга.
        
        ✨ СТАТИСТИКА ТЕПЕРЬ УЧИТЫВАЕТ КОМИССИИ
        
        Returns:
        --------
        Dict
            Словарь с метриками производительности
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit_loss': 0,
                'total_profit_loss_pct': 0,
                'total_commission_paid': 0,  # ✨ НОВОЕ
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'final_capital': self.initial_capital
            }
        
        # Базовые метрики
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        losing_trades = [t for t in self.trades if t.profit_loss <= 0]
        
        total_profit = sum(t.profit_loss for t in winning_trades)
        total_loss = abs(sum(t.profit_loss for t in losing_trades))
        
        # ✨ ОБЩАЯ СУММА КОМИССИЙ
        total_commission = sum(t.commission_paid for t in self.trades)
        
        # Расчёт максимальной просадки
        capital_curve = [self.initial_capital]
        for trade in self.trades:
            capital_curve.append(capital_curve[-1] * (1 + trade.profit_loss_pct / 100))
        
        peak = capital_curve[0]
        max_drawdown = 0
        for value in capital_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            'total_profit_loss': sum(t.profit_loss for t in self.trades),
            'total_profit_loss_pct': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'total_commission_paid': total_commission,  # ✨ НОВОЕ
            'avg_profit': total_profit / len(winning_trades) if winning_trades else 0,
            'avg_loss': total_loss / len(losing_trades) if losing_trades else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
            'max_drawdown': max_drawdown,
            'final_capital': self.current_capital
        }
    
    def save_results(self, filename: str = 'backtest_results.txt'):
        """
        Сохранение результатов бэктестинга в текстовый файл.
        
        Parameters:
        -----------
        filename : str
            Имя файла для сохранения результатов
        """
        stats = self.calculate_statistics()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 РЕЗУЛЬТАТЫ БЭКТЕСТИНГА\n")
            f.write("=" * 80 + "\n\n")
            
            # Общая статистика
            f.write("ОБЩАЯ СТАТИСТИКА:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Начальный капитал:        ${self.initial_capital:,.2f}\n")
            f.write(f"Конечный капитал:         ${stats['final_capital']:,.2f}\n")
            f.write(f"Прибыль/Убыток:           ${stats['final_capital'] - self.initial_capital:,.2f} "
                   f"({stats['total_profit_loss_pct']:.2f}%)\n")
            f.write(f"Комиссия (всего):         ${stats['total_commission_paid']:,.2f}\n")  # ✨
            f.write(f"Максимальная просадка:    {stats['max_drawdown']:.2f}%\n\n")
            
            # Статистика сделок
            f.write("СТАТИСТИКА СДЕЛОК:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Всего сделок:             {stats['total_trades']}\n")
            f.write(f"Прибыльных сделок:        {stats['winning_trades']} "
                   f"({stats['win_rate']:.2f}%)\n")
            f.write(f"Убыточных сделок:         {stats['losing_trades']} "
                   f"({100 - stats['win_rate']:.2f}%)\n")
            f.write(f"Средняя прибыль:          ${stats['avg_profit']:.2f}\n")
            f.write(f"Средний убыток:           ${stats['avg_loss']:.2f}\n")
            f.write(f"Profit Factor:            {stats['profit_factor']:.2f}\n\n")
            
            # Детали всех сделок
            f.write("ДЕТАЛИ ВСЕХ СДЕЛОК:\n")
            f.write("=" * 80 + "\n\n")
            
            for i, trade in enumerate(self.trades, 1):
                f.write(f"Сделка #{i}:\n")
                f.write(f"  Направление:      {trade.direction.upper()}\n")
                f.write(f"  Вход:             {trade.entry_time} @ ${trade.entry_price:.2f}\n")
                f.write(f"  Выход:            {trade.exit_time} @ ${trade.exit_price:.2f}\n")
                f.write(f"  Stop Loss:        ${trade.stop_loss:.2f}\n")
                f.write(f"  Take Profit:      ${trade.take_profit:.2f}\n")
                f.write(f"  Результат:        ${trade.profit_loss:.2f} ({trade.profit_loss_pct:.2f}%)\n")
                f.write(f"  Комиссия:         ${trade.commission_paid:.2f}\n")  # ✨
                f.write(f"  Причина выхода:   {trade.exit_reason}\n")
                f.write("-" * 80 + "\n")
        
        print(f"\n💾 Результаты сохранены в файл: {filename}")


# ============================================================
# НОВЫЙ КЛАСС: СТРАТЕГИЯ EMA CROSSOVER
# ============================================================

class EMA_CrossoverStrategy(TechnicalAnalysis):
    """
    Стратегия на основе пересечения EMA25 и EMA50 с фильтрами.
    
    ЛОГИКА LONG ПОЗИЦИИ:
    -------------------
    1. Цена (close) выше EMA200 — восходящий тренд
    2. EMA25 пересекает EMA50 СНИЗУ ВВЕРХ — сигнал на покупку
    3. Размер свечи (close - low) > 2 × commission — фильтр ликвидности
    4. Stop Loss = low текущей свечи
    5. Take Profit = entry + 2 × (entry - stop_loss) — риск/прибыль 1:2
    
    ЛОГИКА SHORT ПОЗИЦИИ:
    --------------------
    1. Цена (close) ниже EMA200 — нисходящий тренд
    2. EMA25 пересекает EMA50 СВЕРХУ ВНИЗ — сигнал на продажу
    3. Размер свечи (high - close) > 2 × commission — фильтр ликвидности
    4. Stop Loss = high текущей свечи
    5. Take Profit = entry - 2 × (stop_loss - entry) — риск/прибыль 1:2
    
    МАТЕМАТИКА:
    -----------
    Для LONG:
        Risk = Entry - Stop_Loss
        Take_Profit = Entry + 2 × Risk
        
    Для SHORT:
        Risk = Stop_Loss - Entry
        Take_Profit = Entry - 2 × Risk
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0,
                 commission_rate: float = 0.0008):
        """
        Инициализация стратегии EMA Crossover.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для бэктестинга
        initial_capital : float
            Начальный капитал
        commission_rate : float
            Комиссия (по умолчанию 0.08%)
        """
        super().__init__(data, initial_capital, commission_rate)
        
        # Добавляем индикаторы: EMA25, EMA50, EMA200
        print("🔧 Инициализация стратегии EMA Crossover...")
        self.add_indicators(ema_periods=[25, 50, 200], rsi_period=14)
    
    def generate_signal(self, index: int) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Генерация торгового сигнала для EMA Crossover стратегии.
        
        Parameters:
        -----------
        index : int
            Индекс текущей строки
            
        Returns:
        --------
        Tuple[Optional[str], Optional[float], Optional[float]]
            (направление, stop_loss, take_profit) или (None, None, None)
        """
        # Нужно минимум 2 бара для определения пересечения
        if index < 1:
            return None, None, None
        
        # Текущая и предыдущая строки
        current = self.data.iloc[index]
        previous = self.data.iloc[index - 1]
        
        # Извлекаем значения индикаторов
        close = current['close']
        low = current['low']
        high = current['high']
        
        ema25_curr = current['ema_25']
        ema50_curr = current['ema_50']
        ema200_curr = current['ema_200']
        
        ema25_prev = previous['ema_25']
        ema50_prev = previous['ema_50']
        
        # Минимальный размер свечи для входа (фильтр ликвидности)
        # Должен быть больше 2 × commission
        min_candle_size = 2 * self.commission_rate * close
        
        # ============================================================
        # ПРОВЕРКА УСЛОВИЙ ДЛЯ LONG ПОЗИЦИИ
        # ============================================================
        
        # Условие 1: Цена выше EMA200 (восходящий тренд)
        price_above_ema200 = close > ema200_curr
        
        # Условие 2: EMA25 пересекает EMA50 СНИЗУ ВВЕРХ
        # Было: EMA25 < EMA50 (предыдущий бар)
        # Стало: EMA25 > EMA50 (текущий бар)
        ema_bullish_cross = (ema25_prev < ema50_prev) and (ema25_curr > ema50_curr)
        
        # Условие 3: Размер свечи (close - low) > 2 × commission
        candle_size_long = close - low
        candle_filter_long = candle_size_long > min_candle_size
        
        # Если все условия выполнены — открываем LONG
        if price_above_ema200 and ema_bullish_cross and candle_filter_long:
            # Entry = Close текущей свечи
            entry_price = close
            
            # Stop Loss = Low текущей свечи
            stop_loss = low
            
            # Риск = Entry - Stop Loss
            risk = entry_price - stop_loss
            
            # Take Profit = Entry + 2 × Risk (соотношение 1:2)
            take_profit = entry_price + 2 * risk
            
            return 'long', stop_loss, take_profit
        
        # ============================================================
        # ПРОВЕРКА УСЛОВИЙ ДЛЯ SHORT ПОЗИЦИИ
        # ============================================================
        
        # Условие 1: Цена ниже EMA200 (нисходящий тренд)
        price_below_ema200 = close < ema200_curr
        
        # Условие 2: EMA25 пересекает EMA50 СВЕРХУ ВНИЗ
        # Было: EMA25 > EMA50 (предыдущий бар)
        # Стало: EMA25 < EMA50 (текущий бар)
        ema_bearish_cross = (ema25_prev > ema50_prev) and (ema25_curr < ema50_curr)
        
        # Условие 3: Размер свечи (high - close) > 2 × commission
        candle_size_short = high - close
        candle_filter_short = candle_size_short > min_candle_size
        
        # Если все условия выполнены — открываем SHORT
        if price_below_ema200 and ema_bearish_cross and candle_filter_short:
            # Entry = Close текущей свечи
            entry_price = close
            
            # Stop Loss = High текущей свечи
            stop_loss = high
            
            # Риск = Stop Loss - Entry
            risk = stop_loss - entry_price
            
            # Take Profit = Entry - 2 × Risk (соотношение 1:2)
            take_profit = entry_price - 2 * risk
            
            return 'short', stop_loss, take_profit
        
        # Нет сигнала
        return None, None, None


# ============================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ СТРАТЕГИИ
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК СТРАТЕГИИ EMA CROSSOVER")
    print("=" * 80 + "\n")
    
    
    data = pd.read_csv('past_your_data.csv')
    
    print(f"✅ Данные готовы: {len(data)} баров")
    print(f"Период: {data.iloc[0]['datetime']} - {data.iloc[-1]['datetime']}\n")
    
    # ============================================================
    # СОЗДАЁМ И ЗАПУСКАЕМ СТРАТЕГИЮ
    # ============================================================
    
    # Параметры
    INITIAL_CAPITAL = 100.0  
    COMMISSION = 0.000      
    
    # Создаём экземпляр стратегии
    strategy = EMA_CrossoverStrategy(
        data=data,
        initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION
    )
    
    # Запускаем бэктестинг
    strategy.run_backtest()
    
    # Выводим статистику в консоль
    stats = strategy.calculate_statistics()
    
    print("\n" + "=" * 80)
    print("📈 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"💰 Начальный капитал:       ${stats['final_capital'] - stats['total_profit_loss']:,.2f}")
    print(f"💰 Конечный капитал:        ${stats['final_capital']:,.2f}")
    print(f"📊 Прибыль/Убыток:          ${stats['total_profit_loss']:,.2f} ({stats['total_profit_loss_pct']:.2f}%)")
    print(f"💸 Комиссия (всего):        ${stats['total_commission_paid']:,.2f}")
    print(f"📉 Макс. просадка:          {stats['max_drawdown']:.2f}%")
    print(f"🎯 Всего сделок:            {stats['total_trades']}")
    print(f"✅ Прибыльных:              {stats['winning_trades']} ({stats['win_rate']:.2f}%)")
    print(f"❌ Убыточных:               {stats['losing_trades']} ({100 - stats['win_rate']:.2f}%)")
    print(f"💵 Средняя прибыль:         ${stats['avg_profit']:.2f}")
    print(f"💔 Средний убыток:          ${stats['avg_loss']:.2f}")
    print(f"⚖️  Profit Factor:           {stats['profit_factor']:.2f}")
    print("=" * 80)
    
    # Сохраняем детальный отчёт в файл
    strategy.save_results('ema_crossover_results.txt')
    
    print("\n🎉 Готово! Проверь файл 'ema_crossover_results.txt' для детальной информации.")
    print("\n💡 СОВЕТ: Теперь можешь загрузить свои реальные данные и протестировать!")
    print("   Просто замени генерацию тестовых данных на загрузку из CSV:\n")
    print("   data = pd.read_csv('your_data.csv')")
    print("   strategy = EMA_CrossoverStrategy(data, initial_capital=10000)")
    print("   strategy.run_backtest()")
    print("   strategy.save_results('my_results.txt')\n")
    
