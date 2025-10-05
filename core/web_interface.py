# web_interface.py
import sys
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import json
import os
from pathlib import Path
from datetime import datetime
from utils.db_setup import SessionLocal
from utils.audit_models import AuditLog
from core.decision_engine import DecisionEngine
from dotenv import load_dotenv

# Load .env with explicit path
env_path = Path('..') / 'Credential.env'
load_dotenv(dotenv_path=env_path, override=True)


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


app = Flask(__name__, static_folder='static', static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*")
decision_engine = DecisionEngine()
# Attach the decision engine to the account manager.
decision_engine.account_manager.set_decision_engine(decision_engine)

def get_symbols_config():
    config_path = Path(__file__).parent.parent / 'config' / 'symbols.json'
    with open(config_path) as f:
        return json.load(f)

@app.route('/')
def dashboard():
    symbols_config = get_symbols_config()
    return render_template('dashboard.html',
                           current_symbol='AAPL',
                           current_timeframe='1H',
                           alerts=decision_engine.account_manager.get_recent_alerts(),
                           sectors=symbols_config['sectors'],
                           indices=symbols_config.get('indices', {}),
                           timeframes=symbols_config['timeframes'],
                           lora_status=decision_engine.get_metrics().get('lora_status', {}),
                           adapter_version=decision_engine.get_metrics().get('lora_version', 'N/A'))

@app.route('/set_symbol', methods=['POST'])
def set_symbol():
    try:
        data = request.json
        symbol = data.get("symbol")
        config = get_symbols_config()
        allowed_symbols = []
        for sector in config.get("sectors", {}).values():
            allowed_symbols.extend(sector)
        for index in config.get("indices", {}).values():
            allowed_symbols.extend(index)
        if symbol not in allowed_symbols:
            return jsonify({'status': 'error', 'message': 'Symbol not allowed'})
        decision_engine.account_manager.set_trading_symbol(symbol)
        return jsonify({'status': 'success', 'symbol': symbol})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/close_position', methods=['POST'])
def close_position():
    try:
        data = request.json
        symbol = data.get("symbol")
        exit_price = float(data.get("exit_price"))
        reason = data.get("reason", "User manual closure")
        # This call will now delegate to the decision engine via the account manager.
        pnl = decision_engine.account_manager.manual_close_position(symbol, exit_price, reason)
        return jsonify({"status": "success", "symbol": symbol, "pnl": pnl})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/account_summary', methods=['GET'])
def account_summary():
    try:
        summary = decision_engine.account_manager.get_account_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def send_reasoning_update(reasoning_data):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "reasoning": reasoning_data["analysis"],
        "entry_points": reasoning_data.get("entries", [])
    }
    socketio.emit('reasoning_update', json.dumps(payload))

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
