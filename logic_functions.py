import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

def prosumer_dr_decision_rolling_cove(
    lcoe: float,
    window_size: int,
    historical_prices_actual: list[float],
    historical_agent_generation: list[float],
    p_dr_offer: float):
    
    if len(historical_prices_actual) < window_size or len(historical_agent_generation) < window_size:
        return False, 0
    
    if window_size <= 1:
        return False, 0

    window_prices_actual = np.array(historical_prices_actual[-window_size:])
    window_agent_generation = np.array(historical_agent_generation[-window_size:])

    total_generation_in_window = np.sum(window_agent_generation)
    g_agent_avg_rw = np.mean(window_agent_generation)

    if total_generation_in_window <= 1e-2: # Using a small threshold to avoid division by zero
        return False, 0

    p_avg_rw = np.mean(window_prices_actual)
    if p_avg_rw <= 1e-6:

        if p_dr_offer > lcoe:
             return True, p_dr_offer
        else:
             return False, p_dr_offer
    p_values_rw = window_prices_actual / p_avg_rw
    
    if window_size < 2 : # Should have been caught earlier, but defensive
        return False, "Window too small for covariance."

    if np.any(np.isnan(p_values_rw)) or np.any(np.isinf(p_values_rw)):
        return False, "Normalized prices contain NaN/Inf. Check input prices."
    covariance_matrix = np.cov(p_values_rw, window_agent_generation, ddof=0)
    cov_p_g_rw = covariance_matrix[0, 1]
    denominator_cove_rolling = cov_p_g_rw + g_agent_avg_rw
    if abs(denominator_cove_rolling) < 1e-9:
         return False, "R_cov denominator near zero, implies extreme (de)valuation. Benchmark unstable."

    r_cov_rw = g_agent_avg_rw / denominator_cove_rolling
    cove = lcoe * r_cov_rw
    participate_in_dr = p_dr_offer > cove

    return participate_in_dr, cove

WINDOW_SIZE = 20  # Number of previous time steps to use for prediction
TARGET_VARIABLE_INDEX = 0  # Index of the variable to predict (0 for 'X' in this example)
TRAIN_SPLIT = 0.8 # Percentage of data to use for training
N_EPOCHS = 50
BATCH_SIZE = 32

def create_dataset(X_data, y_data, window_size):
    """
    Creates sequences of data for LSTM.
    X_data: Input features (numpy array)
    y_data: Target variable (numpy array)
    window_size: Number of previous time steps to use as input features
    """
    X, y = [], []
    for i in range(len(X_data) - window_size):
        X.append(X_data[i:(i + window_size), :]) # All features for the window
        y.append(y_data[i + window_size])       # Target variable at the next time step
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, n_outputs=1):
    """
    Builds the LSTM model.
    input_shape: Shape of the input data (window_size, n_features)
    n_outputs: Number of output values to predict (usually 1 for the next step)
    """
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape, activation='relu'),
        Dropout(0.2),
        LSTM(units=50, activation='relu'),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=n_outputs) # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(actual, predicted, title='LSTM Time Series Prediction'):
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Values', color='blue')
    plt.plot(predicted, label='Predicted Values', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def predict_price(data_len: int, features_to_use: list[str], target_variable_name: str):

    data_len = 200
    X_var = np.sin(np.linspace(0, 20, data_len)) + np.random.normal(0, 0.2, data_len)
    Y_var = np.cos(np.linspace(0, 15, data_len)) + np.random.normal(0, 0.15, data_len)
    

    # Combine into a DataFrame (optional, but good practice)
    df = pd.DataFrame({
        'X': X_var,
        'Y': Y_var
    })



    # Select features to use for prediction
    # For this example, let's use 'X' and 'Y' to predict 'X'
    features_to_use = ['X', 'Y'] # Add or remove variable names here
    target_variable_name = 'X'

    data_to_process = df[features_to_use].values
    target_data_to_process = df[[target_variable_name]].values # Keep it as 2D for scaler

    # --- 2. Scale Data ---
    # It's crucial to scale data for LSTMs
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(data_to_process)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target_data_to_process)

    # --- 3. Create Sequences ---
    X_sequences, y_sequences = create_dataset(scaled_features, scaled_target.flatten(), WINDOW_SIZE)

    if X_sequences.size == 0:
        raise ValueError(f"Not enough data to create sequences with window size {WINDOW_SIZE}. "
                         f"Need at least {WINDOW_SIZE + 1} data points.")

    print(f"\n--- Shape of sequence data ---")
    print(f"X_sequences shape: {X_sequences.shape}") # (samples, window_size, n_features)
    print(f"y_sequences shape: {y_sequences.shape}")   # (samples,)

    # --- 4. Split Data into Training and Testing ---
    split_index = int(len(X_sequences) * TRAIN_SPLIT)
    X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
    y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]

    print(f"\n--- Shape of training and testing data ---")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # --- 5. Build and Train LSTM Model ---
    n_features = X_train.shape[2] # Number of input features
    lstm_model = build_lstm_model(input_shape=(WINDOW_SIZE, n_features))
    print("\n--- LSTM Model Summary ---")
    lstm_model.summary()

    print("\n--- Training LSTM Model ---")
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = lstm_model.fit(
        X_train, y_train,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1, # Use 10% of training data for validation
        callbacks=[early_stopping],
        verbose=1
    )

    # --- 6. Make Predictions ---
    print("\n--- Making Predictions ---")
    y_pred_scaled = lstm_model.predict(X_test)

    # --- 7. Inverse Transform Predictions ---
    # Reshape y_pred_scaled to 2D if it's not, for inverse_transform
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)) # Reshape y_test for inverse transform

    # --- 8. Evaluate Model (Example: Mean Squared Error) ---
    mse = np.mean((y_pred - y_test_actual)**2)
    print(f"\n--- Model Evaluation ---")
    print(f"Mean Squared Error on Test Set: {mse:.4f}")

    # --- 9. Plot Results ---
    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plot actual vs predicted values for the test set
    plot_predictions(y_test_actual.flatten(), y_pred.flatten(), title=f'LSTM Prediction (Target: {target_variable_name})')

    # --- Example: Predict the very next step after the test data ---
    # Take the last window from the original scaled feature data
    last_window_scaled = scaled_features[-WINDOW_SIZE:]
    last_window_scaled = np.expand_dims(last_window_scaled, axis=0) # Reshape for model prediction (1, window_size, n_features)

    next_step_pred_scaled = lstm_model.predict(last_window_scaled)
    next_step_pred = target_scaler.inverse_transform(next_step_pred_scaled)

    print(f"\n--- Predicting Next Step ---")
    print(f"Last window of actual scaled data (shape): {last_window_scaled.shape}")
    print(f"Predicted next value of '{target_variable_name}' (scaled): {next_step_pred_scaled[0][0]:.4f}")
    print(f"Predicted next value of '{target_variable_name}' (actual): {next_step_pred[0][0]:.4f}")














if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Example 1: Favorable DR Offer ---")
    window = 20
    # Prices that are somewhat volatile, agent generates more when prices are higher recently
    # Let's simulate a scenario where the agent has been "lucky" or smart recently
    # So cov(p,G) should be positive, making R_cov < 1, lowering cove
    example_prices = [0.05, 0.06, 0.04, 0.07, 0.05, 0.08, 0.06, 0.09, 0.07, 0.10,
                      0.04, 0.05, 0.06, 0.08, 0.12, 0.10, 0.07, 0.09, 0.11, 0.13] # Last 20 prices
    example_generation = [1.0, 1.1, 0.9, 1.2, 1.0, 1.3, 1.1, 1.4, 1.2, 1.5,
                          0.8, 1.0, 1.1, 1.3, 1.7, 1.5, 1.2, 1.4, 1.6, 1.8]  # Last 20 generation (kWh)

    annual_cost = 500  # $/year
    # Assuming these are daily values for a specific hour (e.g. noon)
    periods_per_year = 365
    dr_offer = 0.10  # $/kWh

    decision, benchmark = prosumer_dr_decision_rolling_cove(
        window, example_prices, example_generation, annual_cost, periods_per_year, dr_offer
    )
    print(f"Window Size: {window}")
    print(f"P_DR_Offer: ${dr_offer:.4f}/kWh")
    print(f"Calculated cove: ${benchmark:.4f}/kWh" if isinstance(benchmark, float) else benchmark)
    print(f"Decision to participate in DR: {decision}")
    print("-" * 30)

    print("--- Example 2: Unfavorable DR Offer or Poor Recent Performance ---")
    # Agent generates more when prices are lower recently (negative covariance)
    # R_cov > 1, increasing cove
    example_prices_2 = [0.10, 0.09, 0.11, 0.08, 0.12, 0.07, 0.10, 0.06, 0.09, 0.05,
                        0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.08, 0.07, 0.06, 0.04]
    example_generation_2 = [1.0, 1.1, 0.9, 1.2, 1.0, 1.3, 1.1, 1.4, 1.2, 1.5,
                            1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.5, 1.6, 1.7, 1.9] # Still generating a lot
    dr_offer_2 = 0.08 # $/kWh

    decision_2, benchmark_2 = prosumer_dr_decision_rolling_cove(
        window, example_prices_2, example_generation_2, annual_cost, periods_per_year, dr_offer_2
    )
    print(f"Window Size: {window}")
    print(f"P_DR_Offer: ${dr_offer_2:.4f}/kWh")
    print(f"Calculated cove: ${benchmark_2:.4f}/kWh" if isinstance(benchmark_2, float) else benchmark_2)
    print(f"Decision to participate in DR: {decision_2}")
    print("-" * 30)

    print("--- Example 3: Zero Generation in Window ---")
    example_prices_3 = [0.05] * window
    example_generation_3 = [0.0] * window
    dr_offer_3 = 0.10

    decision_3, benchmark_3 = prosumer_dr_decision_rolling_cove(
        window, example_prices_3, example_generation_3, annual_cost, periods_per_year, dr_offer_3
    )
    print(f"Window Size: {window}")
    print(f"P_DR_Offer: ${dr_offer_3:.4f}/kWh")
    print(f"Calculated cove: {benchmark_3}")
    print(f"Decision to participate in DR: {decision_3}")
    print("-" * 30)

    print("--- Example 4: Very Low Average Price in Window ---")
    example_prices_4 = [0.000001, 0.000002, 0.000001, 0.000003, 0.000001] * (window // 5)
    example_generation_4 = [1.0, 1.1, 0.9, 1.2, 1.0] * (window // 5)
    dr_offer_4 = 0.05

    decision_4, benchmark_4 = prosumer_dr_decision_rolling_cove(
        window, example_prices_4, example_generation_4, annual_cost, periods_per_year, dr_offer_4
    )
    print(f"Window Size: {window}")
    print(f"P_DR_Offer: ${dr_offer_4:.4f}/kWh")
    print(f"Calculated cove: {benchmark_4}") # Will be a string message
    print(f"Decision to participate in DR: {decision_4}")
    print("-" * 30)
