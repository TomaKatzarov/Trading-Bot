import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import os
import json
import joblib
import mlflow
from datetime import datetime
import optuna
from optuna.samplers import TPESampler

# Assuming these are in core/models and core/nn_data_preprocessor
from core.models.nn_architectures import MLP, LSTMModel, GRUModel, CNNLSTMModel
from core.nn_data_preprocessor import NNDataPreprocessor

# Configuration
CONFIG = {
    "data_path": "data/processed_nn_data/ALL_SYMBOLS_combined_data.npz",
    "scaler_path": "models/scalers/GLOBAL_scaler.joblib",
    "model_save_dir": "models/trained_supervised_nn/",
    "features": [
        "RSI", "MACD", "MACD_Signal", "MACD_Hist", "SMA_10", "EMA_10", "Bollinger_Bands_Upper",
        "Bollinger_Bands_Lower", "ATR", "ADX", "CCI", "Stochastic_K", "Stochastic_D",
        "Volume_Change", "VWAP", "OBV", "CMF", "FI", "EOM", "Sentiment_Score"
    ],
    "target_column": "BUY_SIGNAL",
    "sequence_length": 30,
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 0.001,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout_rate": 0.2,
    "model_type": "LSTM", # Options: "MLP", "LSTM", "GRU", "CNN-LSTM"
    "early_stopping_patience": 10,
    "focal_loss_alpha": 0.25,
    "focal_loss_gamma": 2.0,
    "weighted_loss_pos_weight": 1.0 # Adjust based on class imbalance
}

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    best_val_f1_macro = -1
    epochs_no_improve = 0
    
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            preds = (outputs.squeeze() > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        train_accuracy = accuracy_score(all_targets, all_preds)
        train_precision = precision_score(all_targets, all_preds, zero_division=0)
        train_recall = recall_score(all_targets, all_preds, zero_division=0)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                val_running_loss += loss.item() * inputs.size(0)

                preds = (outputs.squeeze() > 0.5).long()
                val_all_preds.extend(preds.cpu().numpy())
                val_all_targets.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_f1_macro = f1_score(val_all_targets, val_all_preds, average='macro', zero_division=0)
        val_accuracy = accuracy_score(val_all_targets, val_all_preds)
        val_precision = precision_score(val_all_targets, val_all_preds, zero_division=0)
        val_recall = recall_score(val_all_targets, val_all_preds, zero_division=0)

        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {epoch_loss:.4f}, Train F1 (Macro): {train_f1_macro:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1 (Macro): {val_f1_macro:.4f}")

        mlflow.log_metrics({
            "train_loss": epoch_loss,
            "train_f1_macro": train_f1_macro,
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "val_loss": val_loss,
            "val_f1_macro": val_f1_macro,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall
        }, step=epoch)

        # Early stopping based on validation F1-macro score
        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            epochs_no_improve = 0
            # Save the best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"best_model_{timestamp}_epoch{epoch+1}_valf1macro{best_val_f1_macro:.4f}.pt"
            save_path = os.path.join(config["model_save_dir"], config["model_type"].lower(), model_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation F1-macro.")
                break

def objective(trial):
    with mlflow.start_run(run_name=f"Optuna_Trial_{trial.number}"):
        # Log Optuna trial parameters to MLflow
        mlflow.log_params(trial.params)

        # Hyperparameters to tune
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
        
        # Update CONFIG with trial parameters
        trial_config = CONFIG.copy()
        trial_config["learning_rate"] = learning_rate
        trial_config["batch_size"] = batch_size
        trial_config["hidden_size"] = hidden_size
        trial_config["num_layers"] = num_layers
        trial_config["dropout_rate"] = dropout_rate

        # Load data
        data = NNDataPreprocessor.load_npz(trial_config["data_path"])
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

        # Convert to PyTorch tensors
        if trial_config["model_type"] == "MLP":
            # For MLP, flatten the sequence length into the feature dimension
            input_size = X_train.shape[1] * X_train.shape[2]
            X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)
        else:
            input_size = X_train.shape[2] # Number of features
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_test_tensor, y_test_tensor) # Using test set as validation for Optuna

        train_loader = DataLoader(train_dataset, batch_size=trial_config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=trial_config["batch_size"], shuffle=False)

        # Initialize model
        output_size = 1 # Binary classification
        if trial_config["model_type"] == "MLP":
            model = MLP(input_size, trial_config["hidden_size"], output_size).to(DEVICE)
        elif trial_config["model_type"] == "LSTM":
            model = LSTMModel(input_size, trial_config["hidden_size"], trial_config["num_layers"], output_size, trial_config["dropout_rate"]).to(DEVICE)
        elif trial_config["model_type"] == "GRU":
            model = GRUModel(input_size, trial_config["hidden_size"], trial_config["num_layers"], output_size, trial_config["dropout_rate"]).to(DEVICE)
        elif trial_config["model_type"] == "CNN-LSTM":
            model = CNNLSTMModel(input_size, trial_config["hidden_size"], trial_config["num_layers"], output_size, dropout_rate=trial_config["dropout_rate"]).to(DEVICE)
        else:
            raise ValueError(f"Unknown model type: {trial_config['model_type']}")

        # Define loss function and optimizer
        pos_weight = torch.tensor([trial_config["weighted_loss_pos_weight"]]).to(DEVICE)
        criterion = FocalLoss(alpha=trial_config["focal_loss_alpha"], gamma=trial_config["focal_loss_gamma"], reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=trial_config["learning_rate"])

        # Train model
        best_val_f1_macro = -1
        epochs_no_improve = 0
        
        for epoch in range(trial_config["epochs"]):
            model.train()
            running_loss = 0.0
            all_preds = []
            all_targets = []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                preds = (outputs.squeeze() > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

            # Validation phase
            model.eval()
            val_all_preds = []
            val_all_targets = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    preds = (outputs.squeeze() > 0.5).long()
                    val_all_preds.extend(preds.cpu().numpy())
                    val_all_targets.extend(labels.cpu().numpy())

            val_f1_macro = f1_score(val_all_targets, val_all_preds, average='macro', zero_division=0)
            
            mlflow.log_metric("val_f1_macro", val_f1_macro, step=epoch)

            if val_f1_macro > best_val_f1_macro:
                best_val_f1_macro = val_f1_macro
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == trial_config["early_stopping_patience"]:
                    break
        
        mlflow.log_metric("best_val_f1_macro", best_val_f1_macro)
        return best_val_f1_macro

def main():
    # Ensure MLflow tracking URI is set
    if "MLFLOW_TRACKING_URI" not in os.environ:
        # Default to local MLflow if not set
        mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    
    mlflow.set_experiment("Supervised_NN_Training")

    # Load preprocessed data
    print(f"Loading data from {CONFIG['data_path']}")
    data = NNDataPreprocessor.load_npz(CONFIG["data_path"])
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    # Convert to PyTorch tensors
    if CONFIG["model_type"] == "MLP":
        input_size = X_train.shape[1] * X_train.shape[2]
        X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)
    else:
        input_size = X_train.shape[2] # Number of features
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Initialize model
    output_size = 1 # Binary classification
    if CONFIG["model_type"] == "MLP":
        model = MLP(input_size, CONFIG["hidden_size"], output_size).to(DEVICE)
    elif CONFIG["model_type"] == "LSTM":
        model = LSTMModel(input_size, CONFIG["hidden_size"], CONFIG["num_layers"], output_size, CONFIG["dropout_rate"]).to(DEVICE)
    elif CONFIG["model_type"] == "GRU":
        model = GRUModel(input_size, CONFIG["hidden_size"], CONFIG["num_layers"], output_size, CONFIG["dropout_rate"]).to(DEVICE)
    elif CONFIG["model_type"] == "CNN-LSTM":
        model = CNNLSTMModel(input_size, CONFIG["hidden_size"], CONFIG["num_layers"], output_size, dropout_rate=CONFIG["dropout_rate"]).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {CONFIG['model_type']}")

    # Define loss function and optimizer
    pos_weight = torch.tensor([CONFIG["weighted_loss_pos_weight"]]).to(DEVICE)
    criterion = FocalLoss(alpha=CONFIG["focal_loss_alpha"], gamma=CONFIG["focal_loss_gamma"], reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Start MLflow run for main training
    with mlflow.start_run(run_name=f"Main_Training_{CONFIG['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(CONFIG)
        train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG)

    # Optuna study for hyperparameter tuning
    print("\nStarting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=CONFIG["random_state"]))
    study.optimize(objective, n_trials=50) # Run 50 trials

    print("\nOptuna study finished.")
    print("Best trial:")
    print(f"  Value: {study.best_value:.4f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()