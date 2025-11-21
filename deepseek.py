import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AggressiveProfitStrategy(bt.Strategy):
    params = (
        # Trend Parameters - More sensitive
        ('ema_fast', 13),
        ('ema_slow', 34),
        ('ema_trend', 89),
        
        # Momentum Parameters - More responsive
        ('rsi_period', 11),
        ('rsi_oversold', 35),
        ('rsi_overbought', 75),
        
        # Volatility Parameters
        ('atr_period', 10),
        ('atr_multiplier', 1.2),  # Tighter stops for more trades
        ('volatility_lookback', 15),
        
        # Volume Parameters
        ('volume_ema', 15),
        ('volume_spike', 1.3),
        
        # Position Sizing - More aggressive
        ('risk_per_trade', 0.02),  # 2% risk per trade
        ('max_portfolio_risk', 0.08),  # 8% total portfolio risk
        ('pyramid_factor', 0.6),  # Add more to winning positions
        
        # Exit Parameters - Faster exits
        ('trail_atr_mult', 1.5),
        ('profit_targets', [1.0, 2.0, 3.0]),  # Closer profit targets
        ('time_stop', 15),  # Shorter holding period
        ('break_even_at', 0.8),  # Move to breakeven faster
    )

    def __init__(self):
        # Trend Indicators - Faster EMAs
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.params.ema_trend)
        
        # Momentum Indicators - More sensitive
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(self.data.close, period_me1=8, period_me2=21, period_signal=9)
        self.stochastic = bt.indicators.Stochastic(self.data, period=14, period_dfast=3)
        
        # Volatility Indicators
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2)
        
        # Volume Indicators
        self.volume_ema = bt.indicators.EMA(self.data.volume, period=self.params.volume_ema)
        
        # Additional momentum
        self.momentum = bt.indicators.Momentum(self.data.close, period=10)
        self.adosc = bt.indicators.AO(self.data)
        
        # Trade Tracking
        self.entry_price = 0
        self.entry_time = None
        self.stop_price = 0
        self.trail_price = 0
        self.position_size = 0
        self.trade_count = 0
        self.win_count = 0
        self.profit_targets = []
        self.partial_exits = 0
        self.break_even_move = False
        
        # Performance
        self.equity_curve = []
        self.max_equity = 100000

    def calculate_position_size(self, stop_distance):
        """More aggressive position sizing"""
        if stop_distance <= 0:
            return 0
            
        risk_amount = self.broker.getvalue() * self.params.risk_per_trade
        base_size = risk_amount / stop_distance
        
        # More aggressive in strong trends
        trend_strength = abs(self.ema_fast[0] - self.ema_slow[0]) / self.atr[0]
        if trend_strength > 1.5:
            base_size *= 1.5
        elif trend_strength > 2.5:
            base_size *= 2.0
            
        max_size = (self.broker.getvalue() * self.params.max_portfolio_risk) / stop_distance
        return min(int(base_size), int(max_size))

    def next(self):
        current_value = self.broker.getvalue()
        self.equity_curve.append(current_value)
        
        if len(self) < 100:
            return

        # ENTRY LOGIC - More signals
        if not self.position:
            signals = self.generate_entry_signals()
            
            for signal in signals:
                if signal:
                    stop_distance = self.calculate_stop_distance(signal["type"])
                    self.position_size = self.calculate_position_size(stop_distance)
                    
                    if self.position_size > 0:
                        self.execute_entry(signal, stop_distance)
                        break  # Take only one signal per bar

        # POSITION MANAGEMENT
        else:
            self.manage_position()

    def generate_entry_signals(self):
        """Generate multiple entry signal types"""
        signals = []
        current_price = self.data.close[0]
        
        # Signal 1: Trend Pullback
        if (current_price > self.ema_trend[0] and
            self.ema_fast[0] > self.ema_slow[0] and
            self.rsi[0] < 55 and self.rsi[0] > 40 and
            current_price < self.ema_fast[0] and
            self.data.volume[0] > self.volume_ema[0] * 1.1):
            signals.append({"type": "PULLBACK", "strength": "HIGH"})
        
        # Signal 2: Momentum Breakout
        if (self.macd.macd[0] > self.macd.signal[0] and
            self.macd.macd[0] > 0 and
            current_price > self.bb.top[0] and
            self.data.volume[0] > self.volume_ema[0] * self.params.volume_spike and
            self.momentum[0] > 0):
            signals.append({"type": "BREAKOUT", "strength": "HIGH"})
        
        # Signal 3: RSI Oversold Bounce
        if (self.rsi[0] < self.params.rsi_oversold and
            self.rsi[0] > self.rsi[-1] and  # RSI rising
            self.stochastic.percK[0] < 25 and
            self.stochastic.percK[0] > self.stochastic.percK[-1] and
            current_price > self.bb.bot[0]):  # Above lower BB
            signals.append({"type": "BOUNCE", "strength": "MEDIUM"})
        
        # Signal 4: MACD Zero-line Cross
        if (self.macd.macd[0] > 0 and self.macd.macd[-1] <= 0 and
            self.ema_fast[0] > self.ema_slow[0] and
            self.data.volume[0] > self.volume_ema[0]):
            signals.append({"type": "MACD_CROSS", "strength": "HIGH"})
        
        # Signal 5: Trend Continuation
        if (current_price > self.ema_trend[0] and
            self.ema_fast[0] > self.ema_slow[0] > self.ema_trend[0] and
            self.macd.macd[0] > self.macd.signal[0] and
            self.macd.macd[0] > self.macd.macd[-1] and  # MACD rising
            self.adosc[0] > 0):
            signals.append({"type": "TREND_CONTINUATION", "strength": "HIGH"})
        
        return signals

    def execute_entry(self, signal, stop_distance):
        """Execute entry with aggressive parameters"""
        self.entry_price = self.data.close[0]
        self.stop_price = self.entry_price - stop_distance
        self.trail_price = self.entry_price
        self.entry_time = len(self)
        self.profit_targets = [
            self.entry_price * (1 + target/100) for target in self.params.profit_targets
        ]
        self.partial_exits = 0
        self.break_even_move = False
        
        self.buy(size=self.position_size)
        self.trade_count += 1
        
        print(f"🎯 {signal['type']} ENTRY | Price: {self.entry_price:.2f} | "
              f"Size: {self.position_size} | Stop: {self.stop_price:.2f}")

    def manage_position(self):
        """Aggressive position management"""
        current_price = self.data.close[0]
        days_in_trade = len(self) - self.entry_time
        
        # Move to breakeven quickly
        if (not self.break_even_move and 
            current_price >= self.entry_price * (1 + self.params.break_even_at/100)):
            self.stop_price = self.entry_price  # Break-even
            self.break_even_move = True
            print(f"🛡️ BREAK-EVEN STOP @ {self.stop_price:.2f}")

        # Partial profit taking
        for i, target in enumerate(self.profit_targets):
            if current_price >= target and i >= self.partial_exits:
                exit_size = int(self.position_size * 0.3)  # Take 30% profit
                if exit_size > 0 and self.position.size > exit_size:
                    self.sell(size=exit_size)
                    self.partial_exits += 1
                    profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                    print(f"🎯 TARGET {i+1} HIT | Price: {current_price:.2f} | Profit: {profit_pct:.2f}%")

        # Aggressive trailing stop
        new_trail = current_price - (self.atr[0] * self.params.trail_atr_mult)
        self.trail_price = max(self.trail_price, new_trail)
        self.stop_price = max(self.stop_price, self.trail_price)

        # Add to winning positions more aggressively
        if (current_price > self.entry_price * 1.015 and 
            self.partial_exits == 0 and 
            self.position.size < self.position_size * 3):
            
            add_size = int(self.position_size * self.params.pyramid_factor)
            current_pos_size = self.position.size
            if add_size > 0 and (current_pos_size + add_size) <= self.position_size * 3:
                self.buy(size=add_size)
                new_avg_price = (self.entry_price * current_pos_size + current_price * add_size) / (current_pos_size + add_size)
                self.entry_price = new_avg_price  # Update average entry price
                print(f"📈 PYRAMIDING | Price: {current_price:.2f} | Added: {add_size}")

        # Exit conditions
        if current_price <= self.stop_price:
            self.close_trade(current_price, "STOP LOSS")
        elif days_in_trade >= self.params.time_stop:
            self.close_trade(current_price, "TIME EXIT")
        # Emergency exit if momentum reverses
        elif (self.macd.macd[0] < self.macd.signal[0] and 
              current_price < self.ema_fast[0] and
              self.partial_exits >= 1):  # Only if we've taken some profit
            self.close_trade(current_price, "MOMENTUM EXIT")

    def close_trade(self, price, reason):
        """Close trade with detailed reporting"""
        profit_pct = (price - self.entry_price) / self.entry_price * 100
        if profit_pct > 0:
            self.win_count += 1
        
        self.close()
        print(f"{'🎯' if profit_pct > 0 else '🛑'} {reason} | "
              f"Price: {price:.2f} | P&L: {profit_pct:+.2f}%")

    def calculate_stop_distance(self, signal_type):
        """Tighter stops for more aggressive trading"""
        base_atr = self.atr[0]
        
        if signal_type in ["PULLBACK", "BOUNCE"]:
            return base_atr * 1.0
        elif signal_type == "BREAKOUT":
            return base_atr * 1.1
        else:
            return base_atr * self.params.atr_multiplier

    def stop(self):
        """Final performance report"""
        if self.trade_count > 0:
            win_rate = (self.win_count / self.trade_count) * 100
            total_return = (self.broker.getvalue() / 100000 - 1) * 100
            
            print(f"\n{'='*60}")
            print(f"🚀 AGGRESSIVE PROFIT STRATEGY RESULTS")
            print(f"{'='*60}")
            print(f"Total Trades: {self.trade_count}")
            print(f"Winning Trades: {self.win_count}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Total Return: {total_return:.1f}%")
            print(f"Final Portfolio: ${self.broker.getvalue():.2f}")
            
            if total_return > 0:
                sharpe = self.calculate_sharpe_ratio()
                print(f"Sharpe Ratio: {sharpe:.2f}")

    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

# Optimized data download
def download_data(symbol, years=2):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    df = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.index = pd.to_datetime(df.index)
    return df

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AggressiveProfitStrategy)
    
    # Focus on QQQ for maximum momentum
    symbols = ['QQQ']  # Concentrate on highest-momentum ETF
    
    print("Downloading Market Data...")
    for symbol in symbols:
        df = download_data(symbol)
        data = bt.feeds.PandasData(dataname=df, name=symbol)
        cerebro.adddata(data)
    
    # More aggressive configuration
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0005)  # Lower commission
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='monthly_returns')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    print("Backtesting Aggressive Profit Strategy...")
    
    results = cerebro.run()
    strat = results[0]
    
    # Generate performance report
    print(f"\n{'='*50}")
    print("📊 AGGRESSIVE TRADING PERFORMANCE")
    print(f"{'='*50}")
    
    # Monthly returns
    monthly_returns = strat.analyzers.monthly_returns.get_analysis()
    print(f"\n📅 MONTHLY PERFORMANCE:")
    positive_months = 0
    total_months = len(monthly_returns)
    
    for date, ret in monthly_returns.items():
        pct = ret * 100
        icon = "🟢" if pct > 0 else "🔴"
        if pct > 0:
            positive_months += 1
        print(f"{date.strftime('%Y-%m')}: {icon} {pct:6.2f}%")
    
    if total_months > 0:
        positive_rate = (positive_months / total_months) * 100
        print(f"Positive Months: {positive_rate:.1f}%")