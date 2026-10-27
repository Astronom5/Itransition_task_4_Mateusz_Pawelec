def count_daily_revenue(df):
    return (
        df
        .groupby('date', as_index=False)['paid_price'].sum()
        .rename(columns={'paid_price': 'daily_amount'})
        .assign(rank=lambda df: df["daily_amount"].rank(ascending=False, method='dense'))
        .sort_values(by='date')
    )


def count_unique_authors(df):
    return (
        df["author"]
        .dropna()
        .str.strip()
        .str.lower()
        .str.split(r"\s*,\s*")
        .apply(frozenset)
        .nunique()
    )


def find_top_customer(orders_df, users_df, ids):
    return (
        orders_df[["user_id", "paid_price"]]
        .merge(users_df, left_on="user_id", right_on="id")
        .groupby(ids)
        .agg(total=("paid_price", "sum"),
             ids=("user_id", lambda s: sorted(s.unique())))
        .loc[lambda df: df["total"] == df["total"].max()]
    )


def find_top_authors(orders_df, books_df):
    return (
        orders_df
        .groupby('book_id', as_index=False)['quantity'].sum()
        .rename(columns={'quantity': 'books_sold'})
        .loc[lambda d: d["books_sold"] == d["books_sold"].max()]
        .merge(books_df, left_on="book_id", right_on="id", how="left")
    )


def filter_by_store(df, store_number):
    return df[df["store_number"] == store_number].drop(columns="store_number")
