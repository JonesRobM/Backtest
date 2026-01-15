import streamlit as st
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

from backtest.data import get_data
from backtest.engine import run_backtest
from backtest.portfolio import Portfolio
from backtest.metrics import calculate_cagr, calculate_sharpe_ratio, calculate_max_drawdown, calculate_volatility
from backtest.risk import generate_risk_report
from backtest.plotting import generate_contribution_plots
from backtest.forecasting import run_monte_carlo_simulation
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="Simple Portfolio Backtester",
    page_icon="📈",
    layout="wide"
)

# --- Main App ---
st.title("📈 Simple Portfolio Backtester")
st.caption("A lightweight tool to visualize the performance of different portfolio allocation, rebalancing, and investment strategies.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Configuration")

    # --- Tickers Input ---
    tickers_input = st.text_input(
        "Enter stock tickers (comma-separated)",
        "AAPL, GOOG, MSFT"
    )
    st.caption("Use suffixes for non-US stocks, e.g., 'VMID.L' for London, 'VOW3.DE' for Frankfurt.")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

    # --- Weights Input ---
    st.subheader("Portfolio Weights")
    weights = {}
    if tickers:
        default_weight = 100 // len(tickers)
        for i, ticker in enumerate(tickers):
            remainder = 100 - (default_weight * len(tickers))
            value = default_weight + 1 if i < remainder else default_weight
            weights[ticker] = st.number_input(f"Weight for {ticker} (%)", min_value=0, max_value=100, value=value, key=ticker)
    
    total_weight = sum(weights.values())
    if total_weight != 100 and tickers:
        st.warning(f"Total weights must sum to 100. Current total: {total_weight}%")

    # --- Rebalancing Strategy ---
    st.subheader("Rebalancing Strategy")
    rebalance_freq_map = {"Buy and Hold": None, "Monthly": "monthly", "Quarterly": "quarterly", "Annually": "annually"}
    rebalance_selection = st.selectbox("Select rebalancing frequency", options=list(rebalance_freq_map.keys()))
    rebalance_frequency = rebalance_freq_map[rebalance_selection]

    # --- Currency Selection ---
    currency_symbol = st.selectbox("Select Currency", ["$", "£", "€"])

    # --- Date Range Input ---
    st.subheader("Backtest Period")
    start_date = st.date_input("Start Date", date(2020, 1, 1))
    end_date = st.date_input("End Date", date.today())

    # --- Investment Inputs ---
    st.subheader("Investment Schedule")
    initial_investment = st.number_input(f"Initial Investment ({currency_symbol})", min_value=0, value=10000)
    monthly_topup = st.number_input(f"Monthly Top-up ({currency_symbol})", min_value=0, value=500)
    annual_increase = st.number_input("Annual Top-up Increase (%)", min_value=0.0, value=5.0, step=0.5, format="%.1f")

    # --- Future Projection ---
    st.subheader("Future Projection")
    projection_years = st.number_input("Projection Period (Years)", min_value=1, max_value=50, value=10)
    st.caption("❗ Projections are based on historical performance and are not a guarantee of future returns.")

    # --- Run Button ---
    run_button = st.button("Run Backtest", type="primary", use_container_width=True, disabled=(total_weight != 100 or not tickers))

# --- Main Content Area ---
if run_button:
    with st.spinner("Running backtest..."):
        # --- Data Fetching & Validation ---
        all_tickers = list(set(tickers + ['SPY']))
        try:
            price_data = get_data(all_tickers, start_date, end_date)
            if price_data.empty or not all(t in price_data.columns for t in all_tickers):
                raise ValueError(f"Failed to fetch complete data for: {', '.join(all_tickers)}")
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
            st.stop()

        user_tickers = [t for t in tickers if t in price_data.columns]
        portfolio_data = price_data[user_tickers].dropna()
        benchmark_data = price_data[['SPY']].dropna()

        if portfolio_data.empty:
            st.error("Not enough data for the selected portfolio tickers and date range.")
            st.stop()

        # --- Portfolio & Backtest Execution ---
        normalized_weights = {ticker: w / 100 for ticker, w in weights.items()}
        user_portfolio = Portfolio(target_weights=normalized_weights, rebalance_frequency=rebalance_frequency)
        benchmark_portfolio = Portfolio(target_weights={'SPY': 1.0}, rebalance_frequency=None)

        portfolio_value, final_holdings, final_cash = run_backtest(portfolio_data, user_portfolio, initial_investment, monthly_topup, annual_increase)
        benchmark_value, _, _ = run_backtest(benchmark_data, benchmark_portfolio, initial_investment, 0, 0)
        benchmark_value.columns = ['Benchmark Value (SPY)']

        # --- Display Results ---
        st.header("Backtest Results")
        combined_values = pd.concat([portfolio_value, benchmark_value], axis=1).dropna()
        st.line_chart(combined_values)

        # --- Performance Metrics ---
        st.header("Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your Portfolio")
            st.metric("CAGR", f"{calculate_cagr(portfolio_value['Portfolio Value']):.2%}")
            st.metric("Annual Volatility", f"{calculate_volatility(portfolio_value['Portfolio Value']):.2%}")
            st.metric("Sharpe Ratio", f"{calculate_sharpe_ratio(portfolio_value['Portfolio Value']):.2f}")
            st.metric("Max Drawdown", f"{calculate_max_drawdown(portfolio_value['Portfolio Value']):.2%}")
        with col2:
            st.subheader("Benchmark (SPY)")
            st.metric("CAGR", f"{calculate_cagr(benchmark_value['Benchmark Value (SPY)']):.2%}")
            st.metric("Annual Volatility", f"{calculate_volatility(benchmark_value['Benchmark Value (SPY)']):.2%}")
            st.metric("Sharpe Ratio", f"{calculate_sharpe_ratio(benchmark_value['Benchmark Value (SPY)']):.2f}")
            st.metric("Max Drawdown", f"{calculate_max_drawdown(benchmark_value['Benchmark Value (SPY)']):.2%}")

        # --- Risk Analysis ---
        if len(user_tickers) > 1:
            st.header("Portfolio Risk Analysis")
            if len(portfolio_data) < 252:
                st.warning("Time period is short (< 1 year). Risk metrics may be less reliable.")
            
            risk_metrics, corr_fig = generate_risk_report(portfolio_data, normalized_weights)
            if risk_metrics and corr_fig:
                r_col1, r_col2 = st.columns(2)
                r_col1.metric("Portfolio Annualized Volatility", f"{risk_metrics['volatility']:.2%}")
                r_col2.metric("Risk from Main Driver (PC1)", f"{risk_metrics['pc1_contribution']:.2%}")
                st.pyplot(corr_fig)
            else:
                st.info("Risk analysis could not be performed due to insufficient data.")
        
        # --- Contribution Analysis Section ---
        st.header("Contribution Analysis")
        
        def get_total_capital_invested(start, end, initial, monthly, increase_pa):
            total = initial
            current_monthly = monthly
            
            # This logic approximates the number of contributions more simply.
            num_months = (end.year - start.year) * 12 + end.month - start.month
            
            if num_months > 0:
                last_year = start.year
                # Loop through each month of the investment period
                for i in range(num_months):
                    current_date = start + relativedelta(months=i+1)
                    if current_date.year > last_year:
                        current_monthly *= (1 + increase_pa / 100.0)
                        last_year = current_date.year
                    total += current_monthly
            return total
            
        total_invested = get_total_capital_invested(start_date, end_date, initial_investment, monthly_topup, annual_increase)
        final_prices = portfolio_data.iloc[-1]
        
        comp_fig, cont_fig = generate_contribution_plots(final_holdings, final_prices, total_invested, normalized_weights)
        
        plot_col1, plot_col2 = st.columns(2)
        plot_col1.pyplot(comp_fig)
        plot_col2.pyplot(cont_fig)

        # --- Future Projection Section ---
        st.header("Future Projection")

        with st.spinner("Running Monte Carlo simulation..."):
            # Calculate historical returns for the simulation
            historical_returns = portfolio_value['Portfolio Value'].pct_change().dropna()
            last_known_value = portfolio_value['Portfolio Value'].iloc[-1]

            # Run the simulation
            forecast_df = run_monte_carlo_simulation(historical_returns, projection_years, last_known_value)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot historical data
            ax.plot(portfolio_value.index, portfolio_value['Portfolio Value'], label='Historical Performance', color='black', linewidth=2)
            
            # Plot forecast data
            ax.plot(forecast_df.index, forecast_df['median'], label='Median Forecast', color='blue', linestyle='--')
            
            # Add uncertainty bounds (cone of uncertainty)
            ax.fill_between(
                forecast_df.index, 
                forecast_df['lower_bound_05'], 
                forecast_df['upper_bound_95'],
                color='gray', alpha=0.3, label='90% Confidence Interval'
            )
            ax.fill_between(
                forecast_df.index, 
                forecast_df['lower_bound_25'], 
                forecast_df['upper_bound_75'],
                color='gray', alpha=0.5, label='50% Confidence Interval'
            )
            
            ax.set_title("Portfolio Future Projection (Monte Carlo Simulation)")
            ax.set_ylabel(f"Portfolio Value ({currency_symbol})")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)

else:
    st.info("Configure your portfolio in the sidebar and click 'Run Backtest' to see the results.")
