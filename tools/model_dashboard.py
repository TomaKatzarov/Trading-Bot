import os
import sys
import logging
# Set environment variables for Streamlit BEFORE importing streamlit
os.environ["STREAMLIT_WATCH_MODULES"] = ""
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = ""
# Disable PyTorch warning messages unrelated to model operation
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create logs directory if it doesn't exist
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True, parents=True)

# Configure logging to file AND console for redundancy
try:
    log_file = logs_dir / 'dashboard.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()  # Also log to console as backup
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
except Exception as e:
    # Fallback to basic console logging if file logging fails
    print(f"Warning: Could not set up file logging: {e}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

# Check if running as a standalone script or through streamlit
# This needs to be done BEFORE importing streamlit to avoid circular imports
is_streamlit = 'streamlit' in sys.modules or 'STREAMLIT_RUN_PATH' in os.environ

if not is_streamlit:
    print("=" * 80)
    print("ERROR: This is a Streamlit app and must be run with 'streamlit run'")
    print("Please use one of the following methods:")
    print()
    print("1. Use the launcher script:")
    print("   python tools/launch_dashboard.py")
    print()
    print("2. Run directly with streamlit:")
    print("   streamlit run tools/model_dashboard.py")
    print("=" * 80)
    sys.exit(1)

# Now import streamlit (only after we've checked)
import streamlit as st

# Import project modules
from models.llm_handler import LLMHandler

def load_training_data_distribution():
    """Load and analyze training data distribution."""
    data_file = project_root / "data" / "training data" / "training_data_pnl_v1.jsonl"
    if not data_file.exists():
        return None
    
    classes = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                classes.append(data.get('target_signal'))
            except Exception:
                continue
    
    return pd.Series(classes).value_counts().sort_index()

def main():
    st.set_page_config(page_title="Trading Model Analysis Dashboard", layout="wide")
    st.title("PnL-based Trading Model Analysis Dashboard")
    
    # Initialize model and data in session state if needed
    if 'llm_handler' not in st.session_state:
        with st.spinner("Loading model and data..."):
            try:
                st.session_state.llm_handler = LLMHandler(use_lora=True)
                st.session_state.training_dist = load_training_data_distribution()
            except Exception as e:
                st.error(f"Error loading model or data: {str(e)}")
                st.session_state.llm_handler = None
                st.session_state.training_dist = None
    
    # Model information
    st.header("Model Information")
    if st.session_state.llm_handler and st.session_state.llm_handler.adapter_loaded:
        st.write(f"Adapter loaded: {st.session_state.llm_handler.adapter_path_used}")
        st.success("LoRA adapter loaded successfully")
    else:
        st.error("No LoRA adapter loaded")
    
    # Training data distribution
    st.header("Training Data Distribution")
    if st.session_state.training_dist is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=st.session_state.training_dist.index, y=st.session_state.training_dist.values, ax=ax)
        ax.set_title("Class Distribution in Training Data")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.warning("Training data distribution not available")
    
    # Test bench
    st.header("Model Testing Bench")
    col1, col2 = st.columns(2)
    
    with col1:
        # Input parameters
        st.subheader("Test Vector Generator")
        
        sentiment = st.slider("Sentiment Score", 0.0, 1.0, 0.5, 0.1)
        
        st.write("Technical Indicators (Normalized)")
        open_val = st.slider("Open", -1.0, 1.0, 0.0, 0.1)
        high_val = st.slider("High", -1.0, 1.0, 0.1, 0.1)
        low_val = st.slider("Low", -1.0, 1.0, -0.1, 0.1)
        close_val = st.slider("Close", -1.0, 1.0, 0.05, 0.1)
        volume_val = st.slider("Volume", -1.0, 1.0, 0.0, 0.1)
        vwap_val = st.slider("VWAP", -1.0, 1.0, 0.0, 0.1)
        rsi_val = st.slider("RSI", -1.0, 1.0, 0.0, 0.1)
        vol_pct_val = st.slider("Volume %", -1.0, 1.0, 0.0, 0.1)
        returns_1h_val = st.slider("1-hour Returns", -1.0, 1.0, 0.0, 0.1)
        
        test_vector = [open_val, high_val, low_val, close_val, volume_val, vwap_val, rsi_val, vol_pct_val, returns_1h_val, sentiment]
        
        presets = st.selectbox(
            "Load Preset",
            ["Custom", "Bullish", "Bearish", "Neutral", "Strong Bullish", "Strong Bearish"]
        )
        
        if presets != "Custom":
            if presets == "Bullish":
                test_vector = [0.2, 0.3, 0.1, 0.25, 0.4, 0.3, 0.6, 0.2, 0.3, 0.7]
            elif presets == "Strong Bullish":
                test_vector = [0.5, 0.8, 0.4, 0.75, 1.0, 0.7, 0.8, 0.9, 0.6, 0.9]
            elif presets == "Bearish":
                test_vector = [-0.2, -0.1, -0.3, -0.25, -0.1, -0.2, 0.3, -0.2, -0.3, 0.3]
            elif presets == "Strong Bearish":
                test_vector = [-0.5, -0.3, -0.7, -0.6, -0.2, -0.4, 0.2, -0.4, -0.6, 0.1]
            elif presets == "Neutral":
                test_vector = [0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
    
    with col2:
        st.subheader("Model Prediction")
        if st.button("Predict"):
            if st.session_state.llm_handler:
                market_data = {
                    "context_vector": test_vector,
                    "sentiment_score": test_vector[-1],
                    "symbol": "TEST"
                }
                
                with st.spinner("Running prediction..."):
                    result = st.session_state.llm_handler.analyze_market(market_data)
                
                pred_class = result.get('predicted_class')
                decision = result.get('decision')
                confidence = result.get('confidence', 0)
                
                # Display results with appropriate styling
                decision_color = {
                    "BUY": "green",
                    "HOLD": "blue",
                    "SELL": "red"
                }.get(decision, "black")
                
                st.markdown(f"**Predicted Class:** {pred_class}")
                st.markdown(f"**Decision:** <span style='color:{decision_color};font-weight:bold'>{decision}</span>", unsafe_allow_html=True)
                
                # Format confidence as percentage with progress bar
                conf_pct = confidence * 100
                st.markdown(f"**Confidence:** {conf_pct:.1f}%")
                st.progress(confidence)
                
                # Show token probabilities
                st.subheader("Token Probabilities")
                if st.session_state.llm_handler:
                    with torch.no_grad():
                        tokenized = st.session_state.llm_handler.tokenizer(
                            st.session_state.llm_handler._build_prompt(market_data), 
                            return_tensors="pt"
                        )
                        tokenized = {k: v.to(st.session_state.llm_handler.device) for k, v in tokenized.items()}
                        outputs = st.session_state.llm_handler.model(**tokenized)
                        logits = outputs.logits[0, -1, :]
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        
                        # Get probabilities for digits 0-4
                        digit_token_ids = [st.session_state.llm_handler.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(5)]
                        class_probs = [probs[token_id].item() for token_id in digit_token_ids]
                        
                        prob_df = pd.DataFrame({
                            "Class": ["SELL (0)", "SELL (1)", "HOLD (2)", "BUY (3)", "BUY (4)"],
                            "Probability": class_probs
                        })
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x="Class", y="Probability", data=prob_df, ax=ax, palette=["red", "coral", "blue", "skyblue", "green"])
                        ax.set_title("Class Probabilities")
                        ax.set_ylabel("Probability")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
            else:
                st.error("Model not loaded correctly")
                
if __name__ == "__main__":
    main()
