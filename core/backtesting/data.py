import datetime
import os, os.path
import pandas as pd

from core.backtesting.event import MarketEvent

class DataHandler:
    """
    DataHandler is an abstract base class providing an interface for
    all inherited data handlers.

    The goal of a DataHandler object is to output a continuous stream
    of the latest market data for all symbols requested. This will
    power the Portfolio and Strategy objects.
    """
    def __init__(self, events):
        self.events = events
        self.continue_backtest = True

    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    def get_latest_bar_datetime(self, symbol):
        """
        Returns the datetime of the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the latest bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_value()")

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest bar.
        """
        raise NotImplementedError("Should implement get_latest_bars_values()")

    def update_bars(self):
        """
        Pushes the latest bar to the queue.
        """
        raise NotImplementedError("Should implement update_bars()")

class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each symbol and provide an interface to obtain the latest
    "bar" of information for each symbol.
    """
    def __init__(self, csv_dir, symbol_list, start_date, end_date, events_queue):
        super().__init__(events_queue)
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converts
        them into pandas DataFrames within a symbol dictionary.
        """
        comb_index = None
        for s in self.symbol_list:
            # Adjust path for Windows
            filepath = os.path.join(self.csv_dir, f"{s}.csv")
            print(f"Attempting to read: {filepath}")
            try:
                df = pd.read_csv(
                    filepath,
                    header=0,
                    index_col=0,
                    parse_dates=True
                )
                df.index = df.index.tz_localize('UTC')
            except FileNotFoundError:
                print(f"Error: CSV file not found for symbol {s} at {filepath}")
                self.continue_backtest = False
                return
            except Exception as e:
                print(f"Error reading CSV for symbol {s}: {e}")
                self.continue_backtest = False
                return

            # Filter by date range
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            self.symbol_data[s] = df.sort_index()

            # Combine the index to ensure all symbols have the same time index
            if comb_index is None:
                comb_index = df.index
            else:
                comb_index = comb_index.union(df.index)

        # Set the combined index for all symbols
        self.bar_stream = pd.DataFrame(index=comb_index)
        for s in self.symbol_list:
            self.bar_stream = self.bar_stream.join(
                pd.DataFrame(self.symbol_data[s]['close']), how='outer'
            )
        self.bar_stream.sort_index(inplace=True)
        self.bar_stream.ffill(inplace=True)
        self.bar_stream.bfill(inplace=True) # Fill any remaining NaNs at the beginning
        self.bar_stream = self.bar_stream.iterrows() # Convert to iterator here

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for row in self.symbol_data[symbol]:
            yield row

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol_data dictionary.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"Symbol {symbol} is not available in the latest bar data.")
            return None
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol_data dictionary,
        or N-k if less than N bars are available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"Symbol {symbol} is not available in the latest bar data.")
            return []
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns the datetime of the last bar from the latest_symbol_data dictionary.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"Symbol {symbol} is not available in the latest bar data.")
            return None
        else:
            return bars_list[-1][0] # Index 0 is datetime

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the latest bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"Symbol {symbol} is not available in the latest bar data.")
            return None
        else:
            return bars_list[-1][1][val_type] # Index 1 is the Series, then access by column name

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"Symbol {symbol} is not available in the latest bar data.")
            return []
        else:
            return [bar[1][val_type] for bar in bars_list[-N:]]

    def update_bars(self):
        """
        Pushes the latest bar to the event queue.
        """
        try:
            index, row = next(self.bar_stream)
        except StopIteration:
            self.continue_backtest = False
            return
        
        # Update latest_symbol_data for each symbol
        for s in self.symbol_list:
            # Ensure the symbol exists in the row (it should if comb_index was used)
            if s in row.index:
                bar_data = (index, row[s]) # (datetime, close_price)
                if s not in self.latest_symbol_data:
                    self.latest_symbol_data[s] = []
                self.latest_symbol_data[s].append(bar_data)
        
        self.events.append(MarketEvent())