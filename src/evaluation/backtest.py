import numpy as np
import pandas as pd

def simulate_trading_strategy(true_returns, pred_returns, pred_directions, transaction_cost=0.001):
    """
    Simulates realistic long/cash or long/short trading strategy based on model directional and return predictions.
    Accounts for slippage / transaction costs.
    """
    # Signal generation: Long (+1) when directional probability > 0.50 and predicted return > 0
    signals = np.where((pred_directions > 0.5) & (pred_returns > 0), 1.0, 0.0)
    
    # Trade execution costs (charged when signal flips)
    position_changes = np.abs(np.diff(signals, prepend=0))
    cost_deductions = position_changes * transaction_cost
    
    # Strategy daily returns
    strategy_returns = signals * true_returns - cost_deductions
    benchmark_returns = true_returns # Buy & Hold
    
    # Cumulative returns
    cum_strategy = np.cumprod(1.0 + strategy_returns) - 1.0
    cum_benchmark = np.cumprod(1.0 + benchmark_returns) - 1.0
    
    # Financial Performance Metrics
    total_return_strat = cum_strategy[-1] * 100.0
    total_return_bench = cum_benchmark[-1] * 100.0
    
    # Annualized Sharpe Ratio (assuming 365 days for crypto)
    strat_mean = np.mean(strategy_returns)
    strat_std = np.std(strategy_returns) + 1e-8
    sharpe_strat = (strat_mean / strat_std) * np.sqrt(365)
    
    bench_mean = np.mean(benchmark_returns)
    bench_std = np.std(benchmark_returns) + 1e-8
    sharpe_bench = (bench_mean / bench_std) * np.sqrt(365)
    
    # Maximum Drawdown
    wealth_strat = np.cumprod(1.0 + strategy_returns)
    peak_strat = np.maximum.accumulate(wealth_strat)
    drawdown_strat = (wealth_strat - peak_strat) / peak_strat
    max_dd_strat = np.min(drawdown_strat) * 100.0
    
    wealth_bench = np.cumprod(1.0 + benchmark_returns)
    peak_bench = np.maximum.accumulate(wealth_bench)
    drawdown_bench = (wealth_bench - peak_bench) / peak_bench
    max_dd_bench = np.min(drawdown_bench) * 100.0
    
    # Win Rate of Trades
    active_trades = signals > 0
    if np.sum(active_trades) > 0:
        win_rate = np.mean(true_returns[active_trades] > 0) * 100.0
    else:
        win_rate = 0.0

    return {
        'Strategy Total Return (%)': float(total_return_strat),
        'Buy & Hold Return (%)': float(total_return_bench),
        'Strategy Sharpe Ratio': float(sharpe_strat),
        'Benchmark Sharpe Ratio': float(sharpe_bench),
        'Strategy Max Drawdown (%)': float(max_dd_strat),
        'Benchmark Max Drawdown (%)': float(max_dd_bench),
        'Trade Win Rate (%)': float(win_rate),
        'cum_strategy': cum_strategy,
        'cum_benchmark': cum_benchmark
    }
