import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings and debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.gap_predictor import GapPredictor

def debug_model_predictions():
    """Debug the model predictions to understand the poor performance."""
    print("Debugging gap predictor model performance...")
    
    # Load data
    try:
        df = pd.read_csv('data/training/daily_gap_predictor_training_data.csv')
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 'Local' in df.columns:
            df['Local'] = pd.to_datetime(df['Local'])
            df = df.set_index('Local')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        print(f"Data loaded: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize predictor
    predictor = GapPredictor(sequence_length=96, output_length=96)
    
    if not predictor.is_trained:
        print("Model is not trained. Cannot debug.")
        return
    
    # Test with a specific date
    test_date = '2021-08-15'
    print(f"\nTesting with date: {test_date}")
    
    try:
        # Get input sequence
        end_date = pd.to_datetime(test_date)
        start_date = end_date - pd.Timedelta(days=1)
        
        input_sequence = df.loc[start_date:end_date - pd.Timedelta(minutes=15)][['generation', 'demand']].values
        
        print(f"Input sequence shape: {input_sequence.shape}")
        print(f"Input sequence range - Generation: [{input_sequence[:, 0].min():.2f}, {input_sequence[:, 0].max():.2f}]")
        print(f"Input sequence range - Demand: [{input_sequence[:, 1].min():.2f}, {input_sequence[:, 1].max():.2f}]")
        
        if len(input_sequence) != 96:
            print(f"ERROR: Need exactly 96 intervals, got {len(input_sequence)}")
            return
        
        # Make prediction
        predicted = predictor.predict(input_sequence)
        actual = df.loc[test_date][['generation', 'demand']].values
        
        print(f"Predicted shape: {predicted.shape}")
        print(f"Actual shape: {actual.shape}")
        
        # Analyze predictions vs actual
        print("\n=== PREDICTION ANALYSIS ===")
        
        for i, feature in enumerate(['generation', 'demand']):
            print(f"\n{feature.upper()}:")
            print(f"  Actual range: [{actual[:, i].min():.2f}, {actual[:, i].max():.2f}]")
            print(f"  Predicted range: [{predicted[:, i].min():.2f}, {predicted[:, i].max():.2f}]")
            print(f"  Actual mean: {actual[:, i].mean():.2f}")
            print(f"  Predicted mean: {predicted[:, i].mean():.2f}")
            print(f"  Actual std: {actual[:, i].std():.2f}")
            print(f"  Predicted std: {predicted[:, i].std():.2f}")
            
            # Calculate metrics
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            mae = mean_absolute_error(actual[:, i], predicted[:, i])
            r2 = r2_score(actual[:, i], predicted[:, i])
            
            print(f"  MSE: {mse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.3f}")
            
            # Check for issues
            if r2 < 0:
                print(f"  ⚠️  NEGATIVE R²! Model performs worse than predicting the mean")
                print(f"  Mean prediction would be: {actual[:, i].mean():.2f}")
            
            if np.any(predicted[:, i] < 0):
                print(f"  ⚠️  Negative predictions detected: {np.sum(predicted[:, i] < 0)} values")
            
            if np.any(np.isnan(predicted[:, i])):
                print(f"  ⚠️  NaN predictions detected: {np.sum(np.isnan(predicted[:, i]))} values")
            
            if np.any(np.isinf(predicted[:, i])):
                print(f"  ⚠️  Infinite predictions detected: {np.sum(np.isinf(predicted[:, i]))} values")
        
        # Create detailed visualization
        create_debug_plots(actual, predicted, test_date)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def create_debug_plots(actual, predicted, test_date):
    """Create detailed debug plots."""
    print("\nCreating debug plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time series comparison
    time_steps = range(len(actual))
    
    # Generation
    axes[0, 0].plot(time_steps, actual[:, 0], 'b-', label='Actual Generation', linewidth=2)
    axes[0, 0].plot(time_steps, predicted[:, 0], 'r--', label='Predicted Generation', linewidth=2)
    axes[0, 0].set_title('Generation: Actual vs Predicted')
    axes[0, 0].set_xlabel('Time Steps (15-min intervals)')
    axes[0, 0].set_ylabel('Generation (kWh)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Demand
    axes[0, 1].plot(time_steps, actual[:, 1], 'g-', label='Actual Demand', linewidth=2)
    axes[0, 1].plot(time_steps, predicted[:, 1], 'orange', linestyle='--', label='Predicted Demand', linewidth=2)
    axes[0, 1].set_title('Demand: Actual vs Predicted')
    axes[0, 1].set_xlabel('Time Steps (15-min intervals)')
    axes[0, 1].set_ylabel('Demand (kWh)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plots
    axes[1, 0].scatter(actual[:, 0], predicted[:, 0], alpha=0.6, color='blue')
    axes[1, 0].plot([actual[:, 0].min(), actual[:, 0].max()], [actual[:, 0].min(), actual[:, 0].max()], 'r--', linewidth=2)
    axes[1, 0].set_title('Generation: Predicted vs Actual')
    axes[1, 0].set_xlabel('Actual Generation')
    axes[1, 0].set_ylabel('Predicted Generation')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(actual[:, 1], predicted[:, 1], alpha=0.6, color='green')
    axes[1, 1].plot([actual[:, 1].min(), actual[:, 1].max()], [actual[:, 1].min(), actual[:, 1].max()], 'r--', linewidth=2)
    axes[1, 1].set_title('Demand: Predicted vs Actual')
    axes[1, 1].set_xlabel('Actual Demand')
    axes[1, 1].set_ylabel('Predicted Demand')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gap_predictor_debug_analysis.png', dpi=300, bbox_inches='tight')
    print("Debug analysis plot saved as gap_predictor_debug_analysis.png")
    plt.show()

def analyze_data_quality():
    """Analyze the quality of the training data."""
    print("\n=== DATA QUALITY ANALYSIS ===")
    
    try:
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
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for missing values
        print(f"\nMissing values:")
        for col in df.columns:
            missing = df[col].isnull().sum()
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
        
        # Check for negative values
        print(f"\nNegative values:")
        for col in df.columns:
            negative = (df[col] < 0).sum()
            print(f"  {col}: {negative} ({negative/len(df)*100:.2f}%)")
        
        # Check for zero values
        print(f"\nZero values:")
        for col in df.columns:
            zero = (df[col] == 0).sum()
            print(f"  {col}: {zero} ({zero/len(df)*100:.2f}%)")
        
        # Statistical summary
        print(f"\nStatistical summary:")
        print(df.describe())
        
        # Check for outliers
        print(f"\nOutlier analysis (values beyond 3 standard deviations):")
        for col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            outliers = df[(df[col] < mean_val - 3*std_val) | (df[col] > mean_val + 3*std_val)]
            print(f"  {col}: {len(outliers)} outliers")
        
        # Check data consistency
        print(f"\nData consistency:")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Total days: {(df.index.max() - df.index.min()).days}")
        print(f"  Expected records per day: {len(df) / ((df.index.max() - df.index.min()).days):.1f}")
        
    except Exception as e:
        print(f"Error analyzing data quality: {e}")

def check_model_architecture():
    """Check the model architecture for potential issues."""
    print("\n=== MODEL ARCHITECTURE ANALYSIS ===")
    
    predictor = GapPredictor(sequence_length=96, output_length=96)
    
    if not predictor.is_trained:
        print("Model is not trained.")
        return
    
    model = predictor.model
    
    print(f"Model summary:")
    model.summary()
    
    print(f"\nModel configuration:")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Total parameters: {model.count_params():,}")
    
    # Check layer configurations
    print(f"\nLayer analysis:")
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i}: {layer.name} - {layer.__class__.__name__}")
        if hasattr(layer, 'units'):
            print(f"    Units: {layer.units}")
        if hasattr(layer, 'rate'):
            print(f"    Dropout rate: {layer.rate}")
        print(f"    Output shape: {layer.output_shape}")
        print(f"    Parameters: {layer.count_params():,}")

if __name__ == "__main__":
    print("Debugging Gap Predictor Performance Issues...")
    
    # Analyze data quality
    analyze_data_quality()
    
    # Check model architecture
    check_model_architecture()
    
    # Debug predictions
    debug_model_predictions()
    
    print("\nDebug analysis completed!")
