import datetime

from core.backtesting.event import FillEvent, OrderEvent

class ExecutionHandler(object):
    """
    The ExecutionHandler abstract class handles the interaction
    between a set of order objects and the actual fill that
    takes place in the market.

    The goal of a ExecutionHandler is to receive Order objects
    from a Portfolio, execute them in the market and receive
    Fill objects in return.

    This is designed to work in a event-driven backtesting system.
    """
    def __init__(self, events):
        self.events = events

    def execute_order(self, event):
        """
        Takes an Order event and executes it, producing
        a Fill event that is placed onto the events queue.
        """
        raise NotImplementedError("Should implement execute_order()")

class SimulatedExecutionHandler(ExecutionHandler):
    """
    The SimulatedExecutionHandler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.

    This is used for backtesting purposes.
    """
    def __init__(self, events):
        super().__init__(events)

    def execute_order(self, event):
        """
        Converts OrderEvents into FillEvents.
        """
        if event.type == 'ORDER':
            fill_event = FillEvent(
                datetime.datetime.now(), # Use current datetime for simplicity in simulation
                event.symbol,
                'ARCA', # Example exchange
                event.quantity,
                event.direction,
                # Simulate fill price as current close price (requires data handler access)
                # For simplicity, we'll assume a perfect fill at the requested price for now.
                # In a real system, this would come from market data.
                # For now, we'll just use a placeholder and assume the portfolio
                # will get the actual price from its data handler.
                fill_cost=1.0 # Placeholder, actual fill_cost will be determined by portfolio
            )
            self.events.append(fill_event)