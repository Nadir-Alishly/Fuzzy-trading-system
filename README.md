# Fuzzy-trading-system

This is trading algorithm based on fuzzy logic. Code written in python language.

Repository contains two main python files, the first one (fuzzy_trading_system_strategy) contains only fuzzy logic code, the second one (fuzzy_trading_system_backtest) contains fuzzy logic and implements backtesting.

Only necessary libraries needed to run code. Libraries: numpy, pandas, pandas_ta, matplotlib, skfuzzy, yfinance, datetime.

Fuzzy system uses pythons skfuzzy library to implement necessary fuzzy operations. Fuzzy system has 3 inputs (rsi, macd and adx signals) and 2 outputs (bullish and bearish).
Rsi, Adx can accept values between 0 and 100, when Macd can accept any values, but scales itself to [-5, 5] range. Outputs are also return values between 0 and 100 range.

Trading algoritm gets data from yahoo finance. After calculating signals (rsi, macd, adx), feeds them to fuzzy system. Based on the output algoritm makes desicion to buy, sell, wait or hold stock. At the end algoritm calculates profit in given period of time and returns plot of trading with profits.
