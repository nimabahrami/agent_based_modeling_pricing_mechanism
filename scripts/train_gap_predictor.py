import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from model.gap_predictor import GapPredictor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


def train_and_save_model():
    data_path = "data/training/daily_gap_predictor_training_data.csv"
    df = pd.read_csv(data_path)
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
   
    data = df[['generation', 'demand']].values
    current_dir = os.getcwd()

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

    print(f"Model and scaler saved to {os.path.join(predictor.base_dir, 'pretrained_models')}")

if __name__ == "__main__":
    train_and_save_model()
