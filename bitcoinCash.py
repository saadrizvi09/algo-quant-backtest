# bch_trend_rider.py
import backtrader as bt
import yfinance as yf
import pandas as pd

class BCHTrendRider(bt.Strategy):
    params = (
        ('trend_ema', 150),          # Between BTC (200) and DOGE (100)
        ('fast_ema', 30),
        ('slow_ema', 60),
        ('atr_period', 14),
        ('vol_ma_period', 20),
        ('risk_per_trade', 0.0075),  # 0.75% risk (moderate)
        ('stop_mult', 2.8),          # Wider than BTC
        ('trail_mult', 3.5),         # Wide trailing stop
        ('use_volume_filter', True), # Recommended for BCH
    )

    def __init__(self):
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.p.trend_ema)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.fast_ema)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.slow_ema)
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        
        if self.p.use_volume_filter:
            self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.vol_ma_period)

        self.entry_price = None
        self.trailing_stop = None
        self.trade_count = 0
        self.win_count = 0

    def next(self):
        if len(self) < self.p.trend_ema:
            return

        current_price = self.data.close[0]
        atr_val = self.atr[0]

        # Volume filter
        volume_ok = True
        if self.p.use_volume_filter:
            volume_ok = self.data.volume[0] > self.vol_ma[0] * 0.8

        # Exit logic
        if self.position:
            new_trail = current_price - (self.p.trail_mult * atr_val)
            if self.trailing_stop is None or new_trail > self.trailing_stop:
                self.trailing_stop = new_trail

            if current_price <= self.trailing_stop:
                self.close()
                self._record_trade(current_price, "TRAILING STOP")
                return

        # Entry logic
        if not self.position:
            # Uptrend: price > EMA150 AND EMA30 > EMA60
            trend_ok = (
                current_price > self.ema_trend[0] and
                self.ema_fast[0] > self.ema_slow[0]
            )
            # Momentum confirmation
            momentum_ok = current_price > self.data.close[-1]

            if trend_ok and momentum_ok and volume_ok:
                risk_usd = self.broker.getvalue() * self.p.risk_per_trade
                stop_dist = self.p.stop_mult * atr_val
                if stop_dist <= 0:
                    return

                size = int(risk_usd / stop_dist)
                if size <= 0:
                    return

                self.buy(size=size)
                self.entry_price = current_price
                self.trailing_stop = current_price - stop_dist
                self.trade_count += 1
                print(f"🚀 LONG @ {current_price:.4f} | Stop: {self.trailing_stop:.4f}")

    def _record_trade(self, exit_price, reason):
        pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        if pnl_pct > 0:
            self.win_count += 1
            icon = "✅"
        else:
            icon = "❌"
        print(f"{icon} {reason} @ {exit_price:.4f} | PnL: {pnl_pct:.2f}%")

    def stop(self):
        if self.trade_count > 0:
            win_rate = self.win_count / self.trade_count * 100
            print(f"\n📊 FINAL STATS")
            print(f"Total Trades: {self.trade_count}")
            print(f"Win Rate: {win_rate:.1f}%")


# ======================
# RUN BACKTEST FOR BCH
# ======================
if __name__ == '__main__':
    print("📥 Downloading BCH-USD data (2018–2025)...")
    df = yf.download(
        "BCH-USD",
        start="2018-01-01",
        end="2025-11-19",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise RuntimeError("❌ Failed to download BCH data")

    # Fix column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    else:
        df.columns = [
            col[0] if isinstance(col, tuple) and len(col) == 1 else col
            for col in df.columns
        ]

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Cerebro setup
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BCHTrendRider)

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0015)  # 0.15% (between BTC and DOGE)

    # Slippage: BCH has moderate liquidity → 0.2%
    cerebro.broker.set_slippage_perc(0.002)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', tann=252)

    print(f'💵 Starting Capital: ${cerebro.broker.getvalue():,.2f}')
    results = cerebro.run()

    final_value = cerebro.broker.getvalue()
    total_return_pct = (final_value / 100_000 - 1) * 100
    print(f'💰 Final Portfolio Value: ${final_value:,.2f}')
    print(f'📈 Total Return: {total_return_pct:.2f}%')

    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()

    print(f"\n🎯 PERFORMANCE METRICS")
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A'):.2f}")
    print(f"Max Drawdown: {dd.max.drawdown:.2f}%")
    print(f"Annual Return: {returns.get('rnorm100', 'N/A'):.2f}%")

    # Buy & Hold comparison
    bh_return = (df['Close'][-1] / df['Close'][0] - 1) * 100
    print(f"\n🔍 Buy & Hold Return: {bh_return:.2f}%")