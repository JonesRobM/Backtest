import pandas as pd
import matplotlib.pyplot as plt

def generate_contribution_plots(
    final_holdings: dict, 
    final_prices: pd.Series, 
    total_capital_invested: float,
    target_weights: dict
):
    """
    Generates plots for portfolio composition and gain/loss contribution.

    Returns:
        A tuple of two matplotlib Figure objects: (composition_fig, contribution_fig).
    """
    # --- Calculate Final Portfolio Composition ---
    final_asset_values = {ticker: final_holdings[ticker] * final_prices[ticker] for ticker in final_holdings}
    final_total_value = sum(final_asset_values.values())
    
    # Filter out assets with negligible value for cleaner charts
    final_asset_values = {k: v for k, v in final_asset_values.items() if v / final_total_value > 0.001}
    
    labels = final_asset_values.keys()
    sizes = final_asset_values.values()

    # --- 1. Composition Pie Chart ---
    comp_fig, comp_ax = plt.subplots()
    comp_ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    comp_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    comp_ax.set_title("Final Portfolio Composition")

    # --- 2. Gain/Loss Contribution Bar Chart ---
    # Approximate an asset's contribution to gain/loss
    asset_contribution = {}
    for ticker, final_value in final_asset_values.items():
        # Approximate capital invested in this asset based on its target weight
        capital_invested_in_asset = total_capital_invested * target_weights.get(ticker, 0)
        gain_loss = final_value - capital_invested_in_asset
        asset_contribution[ticker] = gain_loss
        
    contribution_df = pd.DataFrame.from_dict(asset_contribution, orient='index', columns=['Contribution'])
    contribution_df.sort_values('Contribution', ascending=False, inplace=True)
    
    cont_fig, cont_ax = plt.subplots()
    colors = ['g' if x > 0 else 'r' for x in contribution_df['Contribution']]
    contribution_df['Contribution'].plot(kind='bar', ax=cont_ax, color=colors)
    cont_ax.set_ylabel("Gain / Loss Contribution")
    cont_ax.set_title("Asset Contribution to Overall Gain/Loss")
    plt.tight_layout()

    return comp_fig, cont_fig
