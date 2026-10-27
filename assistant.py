import json
import analytics as an

MODEL = "openai/gpt-oss-120b"
MAX_ROUNDS = 6


def _to_python(x):
    return x.item() if hasattr(x, "item") else x


def query(tables, table, store="all", where=None, group_by=None, metric="count", column=None):
    df = tables[table]
    if store not in (None, "all"):
        df = df[df["store_number"] == int(store)]
    if where:
        df = df.query(where)
    if not group_by:
        value = len(df) if metric == "count" else df[column].agg(metric)
        return {"value": _to_python(value)}
    groups = df.groupby(group_by)
    result = groups.size() if metric == "count" else groups[column].agg(metric)
    result = result.sort_values(ascending=False).head(20).rename("value").reset_index()
    return result.to_dict("records")


def unique_authors(tables, store):
    books = an.filter_by_store(tables["books"], int(store))
    return {"unique_author_sets": an.count_unique_authors(books)}


def top_customers(tables, store, identifying_fields):
    orders = an.filter_by_store(tables["orders"], int(store))
    users = an.filter_by_store(tables["users"], int(store))
    top = an.find_top_customer(orders, users, identifying_fields).reset_index()
    return top.head(20).to_dict("records")


FUNCTIONS = {"query": query, "unique_authors": unique_authors, "top_customers": top_customers}

TOOLS = [
    {"type": "function", "function": {
        "name": "query",
        "description": "Count, sum, mean, min, max or count-distinct over one table, optionally filtered "
                       "and grouped. 'where' is a pandas query string, e.g. \"quantity > 2 and currency == 'EUR'\". "
                       "Revenue = sum of orders.paid_price (USD).",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string", "enum": ["orders", "users", "books"]},
            "store": {"type": "string", "description": "store number or 'all'"},
            "where": {"type": "string"},
            "group_by": {"type": "array", "items": {"type": "string"}},
            "metric": {"type": "string", "enum": ["count", "sum", "mean", "min", "max", "nunique"]},
            "column": {"type": "string", "description": "column to aggregate, not needed for count"}},
                       "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "unique_authors",
        "description": "Number of distinct authors or author sets in one store.",
        "parameters": {"type": "object", "properties": {"store": {"type": "integer"}},
                       "required": ["store"]}}},
    {"type": "function", "function": {
        "name": "top_customers",
        "description": "Highest-spending customers in one store ('total' in USD, 'ids' = their user ids). "
                       "A customer = a unique combination of identifying_fields (user columns).",
        "parameters": {"type": "object", "properties": {
            "store": {"type": "integer"},
            "identifying_fields": {"type": "array", "items": {"type": "string"}}},
                       "required": ["store", "identifying_fields"]}}},
]


def ask(client, question, history, tables, store):
    schema = "\n".join(f"{name}: {list(df.columns)}" for name, df in tables.items())
    system = (
        "Answer questions about a bookstore chain's data using the tools only. "
        "Never calculate or recall numbers yourself; quote tool results exactly. "
        "If the tools cannot answer, say so. State your assumptions.\n"
        f"Tables and columns:\n{schema}\n"
        f"The dashboard shows store {store}; use it unless the question names other stores or says all."
    )
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": question}]

    for _ in range(MAX_ROUNDS):
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, temperature=0
        ).choices[0].message

        if not msg.tool_calls:  # the model wrote the final answer
            return msg.content

        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [c.model_dump(include={"id", "type", "function"})
                                        for c in msg.tool_calls]})
        for call in msg.tool_calls:  # the model asked for lookups
            try:
                args = json.loads(call.function.arguments or "{}")
                result = FUNCTIONS[call.function.name](tables, **args)
            except Exception as e:
                result = {"error": str(e)}  # the model sees the error and can retry
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, default=str)})
    return "Sorry, I could not answer that."
