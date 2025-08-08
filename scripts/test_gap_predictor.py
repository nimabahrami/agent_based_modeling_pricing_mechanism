import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model.gap_predictor import GapPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



def load_quarter_hourly_data():
    print("Loading quarter-hourly test data...")
    df = pd.read_csv('data/training/daily_gap_predictor_training_data.csv')
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'Local' in df.columns:
        df['Local'] = pd.to_datetime(df['Local'])
        df = df.set_index('Local')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    return df

def test_gap_predictor():
    df = load_quarter_hourly_data()
    
    print("\nInitializing gap predictor...")
    predictor = GapPredictor(sequence_length=96, output_length=96)  # 1 day * 96 quarters = 96
    
    if not predictor.is_trained:
        print("No trained model found. Please train the model first.")
        return
    
    test_date = '2021-06-15'  # A date from the training set
    
    print("\nTesting predictions...")
    try:
        end_date = pd.to_datetime(test_date)
        start_date = end_date - pd.Timedelta(days=1)
        
        input_sequence = df.loc[start_date:end_date - pd.Timedelta(minutes=15)][['generation', 'demand']].values
        
        if len(input_sequence) != 96:
            print(f"Skipping {test_date}: Need exactly 96 intervals, got {len(input_sequence)}")
            return
        
        predicted = predictor.predict(input_sequence)  # Shape: (96, 2)
        print('generation', predicted[:][0])
        print('demand', predicted[:][1])
        actual = df.loc[test_date][['generation', 'demand']].values  # Shape: (96, 2)
        
        print(f"\nDate: {test_date}")
        print(f"Predicted shape: {predicted.shape}, Actual shape: {actual.shape}")
        
        generation_mae = np.mean(np.abs(predicted[:, 0] - actual[:, 0]))
        demand_mae = np.mean(np.abs(predicted[:, 1] - actual[:, 1]))
        generation_rmse = np.sqrt(np.mean((predicted[:, 0] - actual[:, 0])**2))
        demand_rmse = np.sqrt(np.mean((predicted[:, 1] - actual[:, 1])**2))
        
        print(f"Generation MAE: {generation_mae:,.2f}, RMSE: {generation_rmse:,.2f}")
        print(f"Demand MAE: {demand_mae:,.2f}, RMSE: {demand_rmse:,.2f}")
        
        print("Sample predictions (first 5 intervals):")
        for i in range(min(5, len(predicted))):
            print(f"  Interval {i+1}: Predicted G={predicted[i,0]:,.1f}, D={predicted[i,1]:,.1f} | "
                  f"Actual G={actual[i,0]:,.1f}, D={actual[i,1]:,.1f}")
        
    except Exception as e:
        print(f"\nError predicting for {test_date}: {str(e)}")
    
    print("\nTesting edge cases...")
    
    zeros_sequence = np.zeros((96, 2))
    try:
        zero_prediction = predictor.predict(zeros_sequence)
        print(f"Prediction for all zeros: shape {zero_prediction.shape}")
        
    except Exception as e:
        return np.zeros((96, 2))
    
    constant_sequence = np.ones((96, 2)) * 1000
    try:
        constant_prediction = predictor.predict(constant_sequence)
        print(f"Prediction for constant value: shape {constant_prediction.shape}")
    except Exception as e:
        return np.zeros((96, 2))

def plot_predictions():
    df = load_quarter_hourly_data()
    predictor = GapPredictor(sequence_length=96, output_length=96)
    
    if not predictor.is_trained:
        print("No trained model found for plotting.")
        return
    
    test_date = '2021-08-15'
    
    try:
        end_date = pd.to_datetime(test_date)
        start_date = end_date - pd.Timedelta(days=1)
        
        input_sequence = df.loc[start_date:end_date - pd.Timedelta(minutes=15)][['generation', 'demand']].values
        
        if len(input_sequence) != 96:
            print(f"Cannot plot: Need exactly 96 intervals, got {len(input_sequence)}")
            return
        
        predicted = predictor.predict(input_sequence)
        actual = df.loc[test_date][['generation', 'demand']].values
        
        time_labels = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(96)]
        
        actual_color = '#000080'   # Navy blue
        predicted_color = '#E9967A' # Dark salmon
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(time_labels, actual[:, 0], color=actual_color, label='Actual Generation', linewidth=2)
        ax1.plot(time_labels, predicted[:, 0], color=predicted_color, linestyle='--', label='Predicted Generation', linewidth=2)
        ax1.set_title(f'Generation Prediction vs Actual - {test_date}')
        ax1.set_ylabel('Generation (kWh)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time_labels, actual[:, 1], color=actual_color, label='Actual Demand', linewidth=2)
        ax2.plot(time_labels, predicted[:, 1], color=predicted_color, linestyle='--', label='Predicted Demand', linewidth=2)
        ax2.set_title(f'Demand Prediction vs Actual - {test_date}')
        ax2.set_xlabel('Time of Day')
        ax2.set_ylabel('Demand (kWh)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
            for i, label in enumerate(ax.xaxis.get_ticklabels()):
                if i % 4 != 0:
                    label.set_visible(False)
        
        plt.tight_layout()
        plot_path = 'test_predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPrediction plot saved as {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {str(e)}")

if __name__ == "__main__":
    test_gap_predictor()
    print("\n" + "="*50)
    print("Creating prediction visualization...")
    plot_predictions() 
