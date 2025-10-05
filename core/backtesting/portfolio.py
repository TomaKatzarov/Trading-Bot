import datetime
import numpy as np
import pandas as pd

from core.backtesting.event import OrderEvent, FillEvent
from core.backtesting.data import DataHandler

class Portfolio(object):
    """
    The Portfolio object handles the positions and account of all
    symbols, as well as providing an interface to both Signal and
    Fill events.
    """
    def __init__(self, data_handler, events, initial_capital=100000.0, start_date=None, strategy=None):
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = self.data_handler.symbol_list
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.strategy = strategy # Store strategy for potential future use (e.g., getting config)

        self.all_positions = self._construct_all_positions()
        self.current_positions = self._construct_current_positions()
        self.all_holdings = self._construct_all_holdings()
        self.current_holdings = self._construct_current_holdings()

        self.equity_curve = pd.DataFrame() # To store equity curve
        self.open_positions_details = {} # To track details of currently open positions
        self.closed_trades = [] # To store details of closed trades

    def _construct_all_positions(self):
        """
        Constructs the positions list using the start_date to ensure
        all symbols are present from the outset.
        """
        d = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        return [d]

    def _construct_current_positions(self):
        """
        Constructs the current positions list, which will be updated
        with each new market bar.
        """
        d = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        return d

    def _construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date to ensure
        all symbols are present from the outset.
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def _construct_current_holdings(self):
        """
        Constructs the current holdings list, which will be updated
        with each new market bar.
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar's holdings
        prior to any new orders being executed.
        """
        latest_datetime = self.data_handler.get_latest_bar_datetime(
            self.symbol_list[0]
        )
        
        # Update positions
        dp = dict((k, v) for k, v in [(s, self.current_positions[s]) for s in self.symbol_list])
        dp['datetime'] = latest_datetime
        self.all_positions.append(dp)

        # Update holdings
        dh = dict((k, v) for k, v in [(s, self.current_holdings[s]) for s in self.symbol_list])
        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for s in self.symbol_list:
            # Approximate the market value of the positions
            market_value = self.current_positions[s] * \
                           self.data_handler.get_latest_bar_value(s, "close")
            dh[s] = market_value
            dh['total'] += market_value
        self.all_holdings.append(dh)

    def update_positions_from_fill(self, fill_event):
        """
        Updates the positions list based on the FillEvent.
        """
        fill_direction = 0
        if fill_event.direction == 'BUY':
            fill_direction = 1
        if fill_event.direction == 'SELL':
            fill_direction = -1

        self.current_positions[fill_event.symbol] += fill_direction * fill_event.quantity

    def update_holdings_from_fill(self, fill_event):
        """
        Updates the holdings list based on the FillEvent.
        """
        fill_direction = 0
        if fill_event.direction == 'BUY':
            fill_direction = 1
        if fill_event.direction == 'SELL':
            fill_direction = -1

        fill_cost = self.data_handler.get_latest_bar_value(
            fill_event.symbol, "close"
        )
        cost = fill_direction * fill_cost * fill_event.quantity
        self.current_holdings['cash'] -= (cost + fill_event.commission)
        self.current_holdings['commission'] += fill_event.commission

    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)

            symbol = event.symbol
            current_time = self.data_handler.get_latest_bar_datetime(symbol)
            current_price = self.data_handler.get_latest_bar_value(symbol, "close")

            # Determine if the fill is opening, adding to, or closing a position
            # Calculate old_position before self.current_positions is updated by update_positions_from_fill
            # The quantity in fill_event is the change, not the new total.
            # So, old_position = new_position - (change if BUY else -change)
            old_position = self.current_positions[symbol] - (event.quantity if event.direction == 'BUY' else -event.quantity)
            new_position = self.current_positions[symbol]

            # Opening a new position
            if old_position == 0 and new_position != 0:
                self.open_positions_details[symbol] = {
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'quantity': new_position,
                    'direction': event.direction # Store 'BUY' or 'SELL'
                }
            # Adding to an existing position (same direction)
            elif (old_position > 0 and new_position > 0 and event.direction == 'BUY') or \
                 (old_position < 0 and new_position < 0 and event.direction == 'SELL'):
                # Weighted average entry price
                old_entry_price = self.open_positions_details[symbol]['entry_price']
                old_quantity = self.open_positions_details[symbol]['quantity']
                
                new_entry_price = ((old_entry_price * abs(old_quantity)) + (current_price * event.quantity)) / abs(new_position)
                self.open_positions_details[symbol]['entry_price'] = new_entry_price
                self.open_positions_details[symbol]['quantity'] = new_position
                self.open_positions_details[symbol]['entry_time'] = current_time # Update entry time to last addition
            
            # Closing or partially closing a position
            elif (old_position > 0 and new_position <= 0 and event.direction == 'SELL') or \
                 (old_position < 0 and new_position >= 0 and event.direction == 'BUY'):
                
                if symbol in self.open_positions_details:
                    entry_price = self.open_positions_details[symbol]['entry_price']
                    entry_time = self.open_positions_details[symbol]['entry_time']
                    # entry_quantity = self.open_positions_details[symbol]['quantity'] # Quantity when position was opened

                    # Calculate PnL for the closed portion
                    # The quantity for PnL calculation should be the quantity of the fill event
                    # that caused the close, not the original entry quantity.
                    
                    # If old_position was positive (long), and now selling, it's a close
                    if old_position > 0 and event.direction == 'SELL':
                        pnl = (current_price - entry_price) * event.quantity
                        trade_type = 'LONG_CLOSED'
                    # If old_position was negative (short), and now buying, it's a close
                    elif old_position < 0 and event.direction == 'BUY':
                        pnl = (entry_price - current_price) * event.quantity
                        trade_type = 'SHORT_CLOSED'
                    else:
                        pnl = 0 # Should not happen with above logic
                        trade_type = 'UNKNOWN'

                    holding_period = current_time - entry_time

                    self.closed_trades.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'quantity': event.quantity, # Quantity of this specific fill that closed part/all
                        'pnl': pnl,
                        'holding_period': holding_period,
                        'trade_type': trade_type
                    })

                    # If position is fully closed, remove from open_positions_details
                    if new_position == 0:
                        del self.open_positions_details[symbol]
                    else: # Partial close, update remaining quantity
                        self.open_positions_details[symbol]['quantity'] = new_position
                        # For partial closes, the entry price and time for the remaining position
                        # should ideally be re-evaluated based on FIFO/LIFO or average cost.
                        # For simplicity, we'll keep the original entry details for the remaining
                        # position, which might not be perfectly accurate for complex scenarios.
                        # A more advanced system would track individual lots.
                else:
                    # This case might occur if a position was opened before backtest start or
                    # if there's a complex scenario not covered by simple open/close.
                    # For now, we'll just log a warning or ignore.
                    print(f"Warning: Closing fill for {symbol} but no open position details found.")
            
            # Reducing an existing position (same direction, but quantity reduced)
            # This case is implicitly handled by the 'closing or partially closing' logic
            # if the quantity goes to zero or changes sign. If it's just a reduction
            # without closing, the PnL is not realized yet.

            symbol = event.symbol
            current_time = self.data_handler.get_latest_bar_datetime(symbol)
            current_price = self.data_handler.get_latest_bar_value(symbol, "close")

            # Determine if the fill is opening, adding to, or closing a position
            old_position = self.current_positions[symbol] - (event.quantity if event.direction == 'BUY' else -event.quantity)
            new_position = self.current_positions[symbol]

            # Opening a new position
            if old_position == 0 and new_position != 0:
                self.open_positions_details[symbol] = {
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'quantity': new_position,
                    'direction': event.direction # Store 'BUY' or 'SELL'
                }
            # Adding to an existing position (same direction)
            elif (old_position > 0 and new_position > 0 and event.direction == 'BUY') or \
                 (old_position < 0 and new_position < 0 and event.direction == 'SELL'):
                # Weighted average entry price
                old_entry_price = self.open_positions_details[symbol]['entry_price']
                old_quantity = self.open_positions_details[symbol]['quantity']
                
                new_entry_price = ((old_entry_price * abs(old_quantity)) + (current_price * event.quantity)) / abs(new_position)
                self.open_positions_details[symbol]['entry_price'] = new_entry_price
                self.open_positions_details[symbol]['quantity'] = new_position
                self.open_positions_details[symbol]['entry_time'] = current_time # Update entry time to last addition
            
            # Closing or partially closing a position
            elif (old_position > 0 and new_position <= 0 and event.direction == 'SELL') or \
                 (old_position < 0 and new_position >= 0 and event.direction == 'BUY'):
                
                if symbol in self.open_positions_details:
                    entry_price = self.open_positions_details[symbol]['entry_price']
                    entry_time = self.open_positions_details[symbol]['entry_time']
                    entry_quantity = self.open_positions_details[symbol]['quantity'] # Quantity when position was opened

                    # Calculate PnL for the closed portion
                    # If old_position was positive (long), and now selling, it's a close
                    if old_position > 0 and event.direction == 'SELL':
                        pnl = (current_price - entry_price) * event.quantity
                        trade_type = 'LONG_CLOSED'
                    # If old_position was negative (short), and now buying, it's a close
                    elif old_position < 0 and event.direction == 'BUY':
                        pnl = (entry_price - current_price) * event.quantity
                        trade_type = 'SHORT_CLOSED'
                    else:
                        pnl = 0 # Should not happen with above logic
                        trade_type = 'UNKNOWN'

                    holding_period = current_time - entry_time

                    self.closed_trades.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'quantity': event.quantity, # Quantity of this specific fill that closed part/all
                        'pnl': pnl,
                        'holding_period': holding_period,
                        'trade_type': trade_type
                    })

                    # If position is fully closed, remove from open_positions_details
                    if new_position == 0:
                        del self.open_positions_details[symbol]
                    else: # Partial close, update remaining quantity
                        self.open_positions_details[symbol]['quantity'] = new_position
                        # For partial closes, the entry price and time for the remaining position
                        # should ideally be re-evaluated based on FIFO/LIFO or average cost.
                        # For simplicity, we'll keep the original entry details for the remaining
                        # position, which might not be perfectly accurate for complex scenarios.
                        # A more advanced system would track individual lots.
                else:
                    # This case might occur if a position was opened before backtest start or
                    # if there's a complex scenario not covered by simple open/close.
                    # For now, we'll just log a warning or ignore.
                    print(f"Warning: Closing fill for {symbol} but no open position details found.")
            
            # Reducing an existing position (same direction, but quantity reduced)
            # This case is implicitly handled by the 'closing or partially closing' logic
            # if the quantity goes to zero or changes sign. If it's just a reduction
            # without closing, the PnL is not realized yet.

    def generate_naive_order(self, signal):
        """
        Simply takes a Signal object and converts it into an
        Order object. This is a very simplistic approach to
        sizing.
        """
        order = None
        symbol = signal.symbol
        signal_type = signal.signal_type
        mkt_quantity = 100 # Fixed quantity for simplicity
        cur_quantity = self.current_positions[symbol]

        if signal_type == 'LONG' and cur_quantity == 0:
            order = OrderEvent(symbol, 'MKT', mkt_quantity, 'BUY')
        if signal_type == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(symbol, 'MKT', mkt_quantity, 'SELL')
        if signal_type == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, 'MKT', abs(cur_quantity), 'SELL')
        if signal_type == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, 'MKT', abs(cur_quantity), 'BUY')
        return order

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders based on
        the portfolio logic.
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            if order_event:
                self.events.append(order_event)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings list of dictionaries.
        """
        print(f"DEBUG: all_holdings before DataFrame creation: {self.all_holdings[:5]}...") # Print first 5 entries
        print(f"DEBUG: Number of holdings entries: {len(self.all_holdings)}")
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve