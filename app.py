import pandas as pd
from Cleaner import Cleaner
import streamlit as st
from Loader import read
import analytics as an
import assistant
from openai import OpenAI, RateLimitError

client = OpenAI(api_key=st.secrets["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")


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
orders_t = orders.assign(date=pd.to_datetime(orders["date"]))
tables = {"books": books, "users": users,
          "orders": orders_t.assign(year=orders_t["date"].dt.year, month=orders_t["date"].dt.month)}

store = st.sidebar.radio(
    "Store",
    options=sorted(orders["store_number"].unique()),
    index=0,
)

orders_f = an.filter_by_store(orders, store)
users_f = an.filter_by_store(users, store)
books_f = an.filter_by_store(books, store)

tab1, tab2, tab3, tab4 = st.tabs(["Revenue", "Authors", "Customers", "Assistant"])

with tab1:
    daily_revenue = an.count_daily_revenue(orders_f)

    st.line_chart(daily_revenue, x="date", y="daily_amount")
    st.dataframe(
        daily_revenue
        .loc[lambda df: df["rank"] <= 5]
        .sort_values(by='rank'),
        hide_index=True
    )

with tab2:
    unique_authors = an.count_unique_authors(books_f)

    st.metric(f'Unique authors in store {store}', unique_authors)

    with st.container(border=True):
        st.subheader("Top Selling author")
        st.dataframe(
            an.find_top_authors(orders_f, books_f),
            hide_index=True
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
        top_customer = an.find_top_customer(orders_f, users_f, identifying_fields)

    except ValueError:
        top_customer = None

    with st.container(border=True):
        st.subheader("Top Customer")
        try:
            st.dataframe(users_f.merge(top_customer, on=identifying_fields), hide_index=True)
        except TypeError:
            st.write("No identifying fields provided")

with tab4:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])

    question = st.chat_input("Ask about the data")
    if question:
        st.chat_message("user").write(question)
        try:
            reply = assistant.ask(client, question, st.session_state.chat[-6:], tables, store)
        except RateLimitError:
            reply = "Rate limit reached, try again in a minute."
        st.chat_message("assistant").write(reply)
        st.session_state.chat += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": reply}]
