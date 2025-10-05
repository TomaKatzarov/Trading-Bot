from __future__ import print_function

from core.backtesting.event import SignalEvent

class Strategy(object):
    """
    Strategy is an abstract base class providing an interface for
    all inherited strategy handling objects.

    The goal of a Strategy object is to generate SignalEvent
    objects for the Portfolio.

    This is designed to work in a event-driven backtesting system.
    """
    def __init__(self, data_handler, events):
        self.data_handler = data_handler
        self.events = events
        self.symbol_list = self.data_handler.symbol_list

    def calculate_signals(self, event):
        """
        Provides the mechanisms to calculate the list of signals.
        """
        raise NotImplementedError("Should implement calculate_signals()")

class BuyAndHoldStrategy(Strategy):
    """
    A testing strategy that simply purchases a fixed quantity of a
    security and holds it until the end of the backtest.
    """
    def __init__(self, data_handler, events):
        super().__init__(data_handler, events)
        self.bought = self._setup_bought()

    def _setup_bought(self):
        """
        Adds keys to the bought dictionary for each symbol
        and sets the initial value to False.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = False
        return bought

    def calculate_signals(self, event):
        """
        Calculates the signals for a simple Buy and Hold strategy.
        """
        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.data_handler.get_latest_bars(s, N=1)
                if bars is not None and len(bars) > 0:
                    if self.bought[s] == False:
                        # (Symbol, Datetime, Type, Strength)
                        signal = SignalEvent(s, bars[0][0], 'LONG')
                        self.events.append(signal)
                        self.bought[s] = True