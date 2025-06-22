# analysis/stock_recommender.py
import pandas as pd

class StockRecommender:
    def __init__(self, scored_df, portfolio_df):
        self.scored_df = scored_df.copy()
        self.portfolio_df = portfolio_df.copy()

    def recommend_additions(self, top_n=5, min_score=60, exclude_owned=True):
        df = self.scored_df.copy()

        if exclude_owned:
            owned_stocks = self.portfolio_df['Stock'].tolist()
            df = df[~df['Stock'].isin(owned_stocks)]

        df = df[df['Final Score'] >= min_score]
        df = df.sort_values(by='Final Score', ascending=False).head(top_n)

        return df[['Stock', 'Ticker', 'Sector', 'PER', 'PBV', 'Yield', 'ROE', 'Final Score']].reset_index(drop=True)
