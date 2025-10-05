class Event:
    """
    Base class for all events in the backtesting system.
    """
    pass

class MarketEvent(Event):
    """
    Handles the event of receiving a new market update (bar).
    """
    def __init__(self):
        self.type = 'MARKET'

class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    def __init__(self, symbol, datetime, signal_type, strength=1.0):
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type # 'LONG', 'SHORT', 'EXIT'
        self.strength = strength # Not used in this basic implementation

class OrderEvent(Event):
    """
    Handles the event of sending an Order to an ExecutionHandler.
    """
    def __init__(self, symbol, order_type, quantity, direction):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type # 'MKT' or 'LMT'
        self.quantity = quantity
        self.direction = direction # 'BUY' or 'SELL'

class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as received from a
    Brokerage or Simulated Execution Handler. Stores information
    about the quantity of an asset filled and at what price.
    """
    def __init__(self, datetime, symbol, exchange, quantity,
                 direction, fill_cost, commission=None):
        self.type = 'FILL'
        self.datetime = datetime
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost

        if commission is None:
            self.commission = self.calculate_commission()
        else:
            self.commission = commission

    def calculate_commission(self):
        """
        Calculates the commission for the fill.
        """
        # Example commission calculation (e.g., fixed fee or percentage)
        # This can be made more sophisticated based on brokerage fees
        commission = 0.0
        if self.quantity > 0:
            commission = 0.001 * self.quantity # Example: 0.1% of quantity
        return commission