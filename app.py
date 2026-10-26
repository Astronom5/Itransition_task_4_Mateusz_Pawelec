import pandas as pd
from Cleaner import Cleaner
import streamlit as st
from Loader import read


def filter_by_store(df, store_number):
    return df[df["store_number"] == store_number].drop(columns="store_number")


@st.cache_data
def load_data():
    books_list, users_list, orders_list = [], [], []
    for n in range(1, 6):
        books, users, orders = (
            read(f"stores/STORE{n}/books.yaml").assign(store_number=n),
            Cleaner.clean_users(read(f"stores/STORE{n}/users.csv")).assign(store_number=n),
            Cleaner.clean_orders(read(f"stores/STORE{n}/orders.parquet")).assign(store_number=n))

        books.columns = books.columns.str.replace(r":([A-Za-z]+)", r"\1", regex=True)
        books_list.append(books)
        users_list.append(users)
        orders_list.append(orders)

    books = pd.concat(books_list, ignore_index=True)
    users = pd.concat(users_list, ignore_index=True)
    orders = pd.concat(orders_list, ignore_index=True)
    return books, users, orders


books, users, orders = load_data()

store = st.sidebar.radio(
    "Store",
    options=sorted(orders["store_number"].unique()),
    index=0,
)

orders_f = filter_by_store(orders, store)
users_f = filter_by_store(users, store)
books_f = filter_by_store(books, store)

tab1, tab2, tab3 = st.tabs(["Revenue", "Authors", "Customers"])

with tab1:
    daily_revenue = (
        orders_f
        .groupby('date', as_index=False)['paid_price'].sum()
        .rename(columns={'paid_price': 'daily_amount'})
        .assign(rank=lambda df: df["daily_amount"].rank(ascending=False, method='dense'))
        .sort_values(by='date')
    )

    st.line_chart(daily_revenue, x="date", y="daily_amount")
    st.dataframe(
        daily_revenue
        .loc[lambda df: df["rank"] <= 5]
        .sort_values(by='rank'),
        hide_index=True
    )

with tab2:
    unique_authors = (
        books_f["author"]
        .dropna()
        .str.strip()
        .str.lower()
        .str.split(r"\s*,\s*")
        .apply(frozenset)
        .nunique()
    )

    st.metric(f'Unique authors in store {store}', unique_authors)

    with st.container(border=True):
        st.subheader("Top Selling author (with their book)")
        st.dataframe(
            orders_f
            .groupby('book_id', as_index=False)['quantity'].sum()
            .rename(columns={'quantity': 'books_sold'})
            .loc[lambda d: d["books_sold"] == d["books_sold"].max()]
            .merge(books_f, left_on="book_id", right_on="id", how="left")
        )

with tab3:
    identifying_fields = st.multiselect(
        "Identifying fields",
        options=users_f.columns,
        default=['phone']
    )

    if identifying_fields:
        unique_users = len(users_f.drop_duplicates(subset=identifying_fields))
    else:
        unique_users = 0

    st.metric(f'Unique customers in store {store}', unique_users)

    try:
        top_customer = (
            orders_f[["user_id", "paid_price"]]
            .merge(users_f, left_on="user_id", right_on="id")
            .groupby(identifying_fields)
            .agg(total=("paid_price", "sum"),
                 ids=("user_id", lambda s: sorted(s.unique())))
            .loc[lambda df: df["total"] == df["total"].max()]
        )
    except ValueError:
        top_customer = None

    with st.container(border=True):
        st.subheader("Top Customer")
        try:
            st.dataframe(users_f.merge(top_customer, on=identifying_fields), hide_index=True)
        except TypeError:
            st.write("No identifying fields provided")
