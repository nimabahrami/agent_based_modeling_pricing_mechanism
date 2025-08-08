"""
This script trains the gap predictor model once and saves it for future use.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.gap_predictor import GapPredictor





# Assuming GapPredictor class is defined in the same notebook or imported

def train_and_save_model():
    # Load training data
    data_path = "data/training/daily_gap_predictor_training_data.csv"

    df = pd.read_csv(data_path)
    # Ensure index is datetime
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Use only generation and demand columns
    data = df[['generation', 'demand']].values

    # Define the base directory for saving/loading models.
    # In a notebook, os.getcwd() is a reasonable default.
    # If you want a specific location, specify it here.
    current_dir = os.getcwd()

    # Initialize and train the model, passing the base_dir
    print("Initializing gap predictor...")
    predictor = GapPredictor(sequence_length=96, output_length=96, base_dir=current_dir)
    print("Training model...")
    history = predictor.train(
        data=data,
        epochs=50,
        batch_size=8,
        verbose=0
    )
    print("Training complete!")
    # Use the base_dir from the predictor instance for the print statement
    print(f"Model and scaler saved to {os.path.join(predictor.base_dir, 'pretrained_models')}")

if __name__ == "__main__":
    train_and_save_model()