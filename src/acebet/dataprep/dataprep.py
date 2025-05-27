import pandas as pd
from pathlib import Path


def prepare_data():
    """
    Prepare the ATP data for modeling.

    Returns
    -------
    df : pandas.DataFrame
        The prepared data.

    """
    data_path = Path(__file__).resolve().parents[2] / "data" / "atp_data.csv"
    data = pd.read_csv(data_path, low_memory=False)

    # Data cleaning and feature renaming
    data["Date"] = pd.to_datetime(data["Date"])
    data.sort_values("Date", inplace=True)

    df = data.copy()
    df.columns = df.columns.str.lower()
    df.rename(
        columns={
            "wrank": "rank_p1",
            "lrank": "rank_p2",
            "wsets": "sets_p1",
            "lsets": "sets_p2",
            "psw": "ps_p1",
            "psl": "ps_p2",
            "b365w": "b365_p1",
            "b365l": "b365_p2",
        },
        inplace=True,
    )
    df.columns = df.columns.str.lower()
    df.rename(
        columns=lambda x: x.replace("winner", "p1").replace("loser", "p2"), inplace=True
    )

    # Swap player columns and adjust the target column
    # This section ensures that player 1 (p1) is not always the winner.
    # It swaps p1 and p2 columns for roughly half of the matches (where index is odd).
    # This helps in creating a balanced dataset for training a model
    # where the model needs to predict the winner irrespective of their original designation as winner or loser.
    p1_columns = df.filter(like="p1").columns  # Select columns related to player 1
    p2_columns = df.filter(like="p2").columns  # Select columns related to player 2
    mask = df.index % 2 == 1  # Create a boolean mask for odd rows (every other match)
    # Swap p1 and p2 columns for the selected rows
    df.loc[mask, p1_columns], df.loc[mask, p2_columns] = (
        df.loc[mask, p2_columns].values,
        df.loc[mask, p1_columns].values,
    )
    # Adjust 'proba_elo' for the swapped rows. If p1 and p2 are swapped,
    # the elo probability of p1 winning becomes 1 - original elo probability.
    df.loc[mask, "proba_elo"] = 1 - df.loc[mask, "proba_elo"].values
    # Create the target variable 'target'.
    # 'target' is True if p1 is the winner (original winner, so mask is False),
    # and False if p2 is the winner (original loser, so mask is True, hence ~mask is False).
    df["target"] = ~mask

    # Naive feature engineering
    # Extract date components for potential use in time-based analysis or feature creation.
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    # Calculate the difference in ranks between player 1 and player 2.
    # This can be a strong predictor of match outcome.
    df["rank_diff"] = df["rank_p1"] - df["rank_p2"]
    # this column is forbidden,
    # to illustrate data leakage leading to 98% accuracy
    # df["p1_won_more_sets"] = 1*(df['sets_p1'] > df['sets_p2'])
    # Identify which player has the better (lower) rank.
    df["best_ranked"] = "p1"  # Assume p1 is best ranked by default
    # If rank_diff is positive, it means rank_p1 > rank_p2, so p2 is better ranked.
    df.loc[df["rank_diff"] > 0, "best_ranked"] = "p2"
    df = df.reset_index(drop=True)  # Reset index after manipulations

    # Write the "production" data to a feather file
    production_data_path = (
        Path(__file__).resolve().parents[2] / "data" / "atp_data_production.feather"
    )
    df.to_feather(production_data_path)


# if __name__ == "__main__":
#     prepare_data()
