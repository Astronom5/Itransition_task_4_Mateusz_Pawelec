import re
import pandas as pd


class Cleaner:
    @staticmethod
    def clean_phone(phone_number):
        return re.sub(r'\D', '', phone_number)

    @staticmethod
    def get_currency(string, currencies):
        for code, symbols in currencies.items():
            if any(symbol in string for symbol in symbols):
                return code
        return None

    @staticmethod
    def clean_users(data):
        data = (
            data
            .assign(phone=lambda df: df["phone"].apply(Cleaner.clean_phone))
        )
        return data

    @staticmethod
    def clean_orders(data):
        data = (
            data
            .assign(currency=lambda df: df["unit_price"].apply(Cleaner.get_currency,
                                                               currencies={"USD": ["$", "USD"], "EUR": ["€", "EUR"]}))
            .assign(unit_price=lambda df: df["unit_price"].str.replace(r"\D", ".", regex=True))
            .assign(unit_price=lambda df: df["unit_price"].str.replace(r"^\.+|\.+$", "", regex=True))
            .assign(timestamp=lambda df: df["timestamp"].str.replace(r"([AaPp])\.[Mm]\.", r"\1M", regex=True))
            .assign(
                hour_format=lambda df: df["timestamp"].str.extract(r"([AaPp][Mm])", expand=False).str.upper().fillna(
                    "AM"))
            .assign(hour=lambda df: df["timestamp"].str.extract(
                r"""([0-9]+:[0-9]+:[0-9]+\.[0-9]+|[0-9]+:[0-9]+:[0-9]+|[0-9]+:[0-9]+)"""))
            .assign(hour=lambda df: pd.to_datetime(df["hour"], format='mixed'))
            .assign(hour=lambda df: df['hour'].mask(df['hour_format'] == "PM", df['hour'] + pd.Timedelta(hours=12)))
            .assign(hour=lambda df: df["hour"].dt.time)
            .assign(timestamp=lambda df: df["timestamp"].str.replace(
                r"([0-9]+:[0-9]+:[0-9]+\.[0-9]+|[0-9]+:[0-9]+:[0-9]+|[0-9]+:[0-9]+|[APap][Mm]|[,;]+)",
                "", regex=True)
                    .str.strip()
                    .str.replace(r"T$", "", regex=True))
            .assign(date=lambda df: pd.to_datetime(df["timestamp"], format='mixed').dt.date)
            .astype({'unit_price': float})
            .drop(columns=['timestamp', 'hour_format'])
            .assign(price_in_USD=lambda df: df['unit_price'].mask(df['currency'] == "EUR", df['unit_price'] * 1.2))
            .assign(paid_price=lambda df: df['price_in_USD'] * df['quantity'])
        )
        return data
