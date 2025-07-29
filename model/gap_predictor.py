import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, LSTM, Dense, Dropout, TimeDistributed, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import joblib
from pathlib import Path # Import Path

# Suppress TensorFlow warnings for cleaner output
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

class GapPredictor:
    def __init__(self, sequence_length=192, output_length=96, base_dir=None):
        self.sequence_length = sequence_length  # 7 days * 96 quarters
        self.output_length = output_length      # 1 day * 96 quarters
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.base_dir = base_dir if base_dir is not None else os.getcwd() # Use provided base_dir or current working directory
        
        # Try to load pre-trained model
        self.load_pretrained_model()
        
    def load_pretrained_model(self):
        """Load the pre-trained model if it exists."""
        # Use the base_dir attribute to construct paths
        model_path = os.path.join(self.base_dir, 'pretrained_models', 'gap_predictor_model.h5')
        scaler_path = os.path.join(self.base_dir, 'pretrained_models', 'gap_predictor_scaler.save')
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print("Loading pre-trained gap predictor model...")
                # We need to provide custom_objects when loading if using Bidirectional layer
                # This is often necessary for custom layers or complex models
                # For standard Bidirectional(LSTM), sometimes it loads correctly, but explicitly defining
                # helps avoid issues. Let's add a try-except for this.
                try:
                     self.model = load_model(model_path)
                except Exception as load_error:
                     print(f"Attempting load with custom_objects due to error: {load_error}")
                     # Define custom objects if needed - for standard layers this might not be required
                     # but it's a common practice if you have custom activations/layers
                     # For Bidirectional, the underlying LSTM needs to be recognized
                     
                     custom_objects = {'LSTM': LSTM, 'Bidirectional': Bidirectional}
                     self.model = load_model(model_path, custom_objects=custom_objects)


                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                print("Pre-trained model loaded successfully!")
            else:
                print("No pre-trained model found. Please train the model first.")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            # If loading fails, ensure model is None and not trained
            self.model = None
            self.is_trained = False


    def build_model(self):
        """Build the LSTM model architecture."""
        model = Sequential([
            Input(shape=(self.sequence_length, 2)),
            Bidirectional(LSTM(units=128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(units=64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(units=32)),
            Dropout(0.2),
            Dense(units=64, activation='relu'),
            Dense(units=self.output_length * 2),  # Flattened output for 96*2
            # Reshape to (96, 2) for each quarter-hour of the next day
            tf.keras.layers.Reshape((self.output_length, 2))
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
        self.model = model
        return model
    
    def augment_data(self, X, y, num_augmentations=2, noise_factor=0.01):
        """Augment training data with noise and scaling."""
        X_augmented = []
        y_augmented = []
        
        for i in range(len(X)):
            # Original sample
            X_augmented.append(X[i])
            y_augmented.append(y[i])
            
            # Augmented samples
            for _ in range(num_augmentations):
                # Add noise
                noise_X = np.random.normal(0, np.std(X[i]) * noise_factor, X[i].shape)
                X_noisy = X[i] + noise_X
                noise_y = np.random.normal(0, np.std(y[i]) * noise_factor, y[i].shape)
                y_noisy = y[i] + noise_y
                
                X_augmented.append(X_noisy)
                y_augmented.append(y_noisy)
                
        return np.array(X_augmented), np.array(y_augmented)
    
    def train(self, data, epochs=300, batch_size=8, verbose=1):
        """Train the model on historical gap data."""
        if self.is_trained:
            print("Model is already trained. Skipping training.")
            return None
            
        if len(data) < self.sequence_length + self.output_length:
            raise ValueError(f"Need at least {self.sequence_length + self.output_length} intervals of data for training")
            
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.output_length + 1):
            X.append(scaled_data[i:(i + self.sequence_length), :])
            y.append(scaled_data[(i + self.sequence_length):(i + self.sequence_length + self.output_length), :])
            
        X = np.array(X)
        y = np.array(y)
        
        # Augment the data
        X_aug, y_aug = self.augment_data(X, y)
        
        # Callbacks
        # Use the base_dir attribute
        save_dir = os.path.join(self.base_dir, 'pretrained_models')
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, 'best_gap_predictor_model.h5')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
        ]
        
        # Build and train the model
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            X_aug, y_aug,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=verbose,
            callbacks=callbacks
        )
        
        self.is_trained = True
        
        # Save the best model as the main model
        # Use the base_dir attribute
        self.model.save(os.path.join(save_dir, 'gap_predictor_model.h5'))
        self.save_model()
        
        return history
    
    def save_model(self):
        """Save the trained model and scaler."""
        try:
            # Create directory if it doesn't exist using base_dir
            save_dir = os.path.join(self.base_dir, 'pretrained_models')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(save_dir, 'gap_predictor_model.h5')
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(save_dir, 'gap_predictor_scaler.save')
            joblib.dump(self.scaler, scaler_path)
            
            print(f"Model and scaler saved to {save_dir}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def predict(self, data):
        """Predict the next day's gap."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if data.shape != (self.sequence_length, 2):
            raise ValueError(f"Need data of shape ({self.sequence_length}, 2) for prediction")
            
        # Scale the input data
        scaled_data = self.scaler.transform(data)
        
        # Reshape for prediction
        X = np.reshape(scaled_data, (1, self.sequence_length, 2))
        
        # Make prediction
        prediction_scaled = self.model.predict(X)
        prediction = self.scaler.inverse_transform(prediction_scaled[0])
        
        return prediction  # shape (96, 2)