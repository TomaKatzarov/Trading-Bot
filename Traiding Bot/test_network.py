# test_network.py
import socket
sock = socket.create_connection(("stream.data.alpaca.markets", 443), timeout=5)
print("Connection successful" if sock else "Failed")