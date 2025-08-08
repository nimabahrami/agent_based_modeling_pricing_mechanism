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

def recalculate_kpis_correctly():
    """Recalculate KPIs with proper error handling and validation."""
    print("Recalculating KPIs with proper validation...")
    
    # Load data
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
    
    # Initialize predictor
    predictor = GapPredictor(sequence_length=96, output_length=96)
    
    if not predictor.is_trained:
        print("Model is not trained. Cannot calculate KPIs.")
        return
    
    # Test dates for evaluation
    test_dates = [
        '2021-06-15', '2021-07-20', '2021-08-15', 
        '2021-09-10', '2021-10-05', '2021-11-20'
    ]
    
    all_metrics = {
        'generation': {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'r2': []},
        'demand': {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'r2': []}
    }
    
    detailed_results = []
    
    for test_date in test_dates:
        try:
            # Get input sequence
            end_date = pd.to_datetime(test_date)
            start_date = end_date - pd.Timedelta(days=1)
            
            input_sequence = df.loc[start_date:end_date - pd.Timedelta(minutes=15)][['generation', 'demand']].values
            
            if len(input_sequence) != 96:
                print(f"Skipping {test_date}: Need exactly 96 intervals, got {len(input_sequence)}")
                continue
            
            # Make prediction
            predicted = predictor.predict(input_sequence)
            actual = df.loc[test_date][['generation', 'demand']].values
            
            # Validate predictions
            if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
                print(f"⚠️  Invalid predictions for {test_date}: NaN or Inf values detected")
                continue
            
            # Calculate metrics for each feature
            for i, feature in enumerate(['generation', 'demand']):
                actual_values = actual[:, i]
                predicted_values = predicted[:, i]
                
                # Additional validation
                if np.all(actual_values == actual_values[0]):  # All values are the same
                    print(f"⚠️  {feature} has constant values for {test_date}, R² cannot be calculated")
                    continue
                
                # Calculate metrics
                mse = mean_squared_error(actual_values, predicted_values)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual_values, predicted_values)
                
                # Calculate MAPE safely (avoid division by zero)
                mape = np.mean(np.abs((actual_values - predicted_values) / np.where(actual_values != 0, actual_values, 1))) * 100
                
                # Calculate R²
                r2 = r2_score(actual_values, predicted_values)
                
                # Store metrics
                all_metrics[feature]['mse'].append(mse)
                all_metrics[feature]['rmse'].append(rmse)
                all_metrics[feature]['mae'].append(mae)
                all_metrics[feature]['mape'].append(mape)
                all_metrics[feature]['r2'].append(r2)
                
                # Store detailed results
                detailed_results.append({
                    'date': test_date,
                    'feature': feature,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'r2': r2,
                    'actual_mean': actual_values.mean(),
                    'predicted_mean': predicted_values.mean(),
                    'actual_std': actual_values.std(),
                    'predicted_std': predicted_values.std(),
                    'negative_predictions': np.sum(predicted_values < 0)
                })
            
            print(f"✓ Evaluated {test_date}")
            
        except Exception as e:
            print(f"✗ Error evaluating {test_date}: {e}")
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("CORRECTED KPI ANALYSIS RESULTS")
    print("="*80)
    
    for feature in ['generation', 'demand']:
        if len(all_metrics[feature]['r2']) > 0:
            print(f"\n{feature.upper()} PREDICTION METRICS:")
            print(f"  RMSE: {np.mean(all_metrics[feature]['rmse']):.2f} ± {np.std(all_metrics[feature]['rmse']):.2f}")
            print(f"  MAE: {np.mean(all_metrics[feature]['mae']):.2f} ± {np.std(all_metrics[feature]['mae']):.2f}")
            print(f"  MAPE: {np.mean(all_metrics[feature]['mape']):.2f}% ± {np.std(all_metrics[feature]['mape']):.2f}%")
            print(f"  R²: {np.mean(all_metrics[feature]['r2']):.3f} ± {np.std(all_metrics[feature]['r2']):.3f}")
            
            # Performance assessment
            avg_r2 = np.mean(all_metrics[feature]['r2'])
            avg_mape = np.mean(all_metrics[feature]['mape'])
            
            if avg_r2 > 0.8:
                performance = "Excellent"
            elif avg_r2 > 0.6:
                performance = "Good"
            elif avg_r2 > 0.4:
                performance = "Fair"
            else:
                performance = "Poor"
            
            print(f"  Performance: {performance}")
        else:
            print(f"\n{feature.upper()}: No valid metrics calculated")
    
    # Create detailed results DataFrame
    if detailed_results:
        results_df = pd.DataFrame(detailed_results)
        print(f"\nDetailed Results Summary:")
        print(results_df.groupby('feature')[['mse', 'rmse', 'mae', 'mape', 'r2']].describe())
        
        # Save detailed results
        results_df.to_csv('gap_predictor_detailed_results.csv', index=False)
        print(f"\nDetailed results saved to gap_predictor_detailed_results.csv")
    
    return all_metrics, detailed_results

def analyze_prediction_patterns():
    """Analyze prediction patterns to understand model behavior."""
    print("\n" + "="*80)
    print("PREDICTION PATTERN ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv('data/training/daily_gap_predictor_training_data.csv')
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Initialize predictor
    predictor = GapPredictor(sequence_length=96, output_length=96)
    
    if not predictor.is_trained:
        print("Model is not trained.")
        return
    
    # Test with multiple dates
    test_dates = ['2021-06-15', '2021-08-15', '2021-10-15']
    
    for test_date in test_dates:
        print(f"\nAnalyzing {test_date}:")
        
        try:
            end_date = pd.to_datetime(test_date)
            start_date = end_date - pd.Timedelta(days=1)
            
            input_sequence = df.loc[start_date:end_date - pd.Timedelta(minutes=15)][['generation', 'demand']].values
            predicted = predictor.predict(input_sequence)
            actual = df.loc[test_date][['generation', 'demand']].values
            
            for i, feature in enumerate(['generation', 'demand']):
                actual_values = actual[:, i]
                predicted_values = predicted[:, i]
                
                print(f"  {feature.capitalize()}:")
                print(f"    Actual range: [{actual_values.min():.2f}, {actual_values.max():.2f}]")
                print(f"    Predicted range: [{predicted_values.min():.2f}, {predicted_values.max():.2f}]")
                print(f"    Correlation: {np.corrcoef(actual_values, predicted_values)[0,1]:.3f}")
                print(f"    R²: {r2_score(actual_values, predicted_values):.3f}")
                
                # Check for systematic bias
                bias = np.mean(predicted_values - actual_values)
                print(f"    Bias: {bias:.2f}")
                
                # Check for variance preservation
                variance_ratio = predicted_values.var() / actual_values.var()
                print(f"    Variance ratio: {variance_ratio:.3f}")
        
        except Exception as e:
            print(f"  Error analyzing {test_date}: {e}")

def create_corrected_kpi_dashboard():
    """Create a corrected KPI dashboard."""
    print("\nCreating corrected KPI dashboard...")
    
    # Recalculate metrics
    all_metrics, detailed_results = recalculate_kpis_correctly()
    
    if not detailed_results:
        print("No valid results to display.")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert to DataFrame for easier plotting
    results_df = pd.DataFrame(detailed_results)
    
    # Plot 1: R² scores by date and feature
    for feature in ['generation', 'demand']:
        feature_data = results_df[results_df['feature'] == feature]
        axes[0, 0].plot(range(len(feature_data)), feature_data['r2'], 
                        marker='o', label=feature.capitalize(), linewidth=2)
    
    axes[0, 0].set_title('R² Scores Over Time')
    axes[0, 0].set_xlabel('Test Date Index')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MAPE scores by date and feature
    for feature in ['generation', 'demand']:
        feature_data = results_df[results_df['feature'] == feature]
        axes[0, 1].plot(range(len(feature_data)), feature_data['mape'], 
                        marker='s', label=feature.capitalize(), linewidth=2)
    
    axes[0, 1].set_title('MAPE Scores Over Time')
    axes[0, 1].set_xlabel('Test Date Index')
    axes[0, 1].set_ylabel('MAPE (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: RMSE comparison
    features = ['generation', 'demand']
    avg_rmse = [results_df[results_df['feature'] == f]['rmse'].mean() for f in features]
    std_rmse = [results_df[results_df['feature'] == f]['rmse'].std() for f in features]
    
    bars = axes[1, 0].bar(features, avg_rmse, yerr=std_rmse, capsize=5, alpha=0.7)
    axes[1, 0].set_title('Average RMSE by Feature')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rmse in zip(bars, avg_rmse):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(avg_rmse)*0.01,
                        f'{rmse:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: R² comparison
    avg_r2 = [results_df[results_df['feature'] == f]['r2'].mean() for f in features]
    std_r2 = [results_df[results_df['feature'] == f]['r2'].std() for f in features]
    
    bars = axes[1, 1].bar(features, avg_r2, yerr=std_r2, capsize=5, alpha=0.7)
    axes[1, 1].set_title('Average R² by Feature')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, r2 in zip(bars, avg_r2):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(avg_r2)*0.01,
                        f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gap_predictor_corrected_kpi_dashboard.png', dpi=300, bbox_inches='tight')
    print("Corrected KPI dashboard saved as gap_predictor_corrected_kpi_dashboard.png")
    plt.show()

if __name__ == "__main__":
    print("Fixing KPI Calculation Issues...")
    
    # Analyze prediction patterns
    analyze_prediction_patterns()
    
    # Recalculate KPIs correctly
    recalculate_kpis_correctly()
    
    # Create corrected dashboard
    create_corrected_kpi_dashboard()
    
    print("\nKPI calculation fixes completed!")
