import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
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

def load_test_data():
    """Load test data for evaluation."""
    print("Loading test data...")
    
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
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_prediction_metrics(actual, predicted):
    """Calculate comprehensive prediction metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(actual, predicted)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(actual, predicted)
    metrics['mape'] = mean_absolute_percentage_error(actual, predicted) * 100
    metrics['r2'] = r2_score(actual, predicted)
    
    # Additional metrics
    metrics['mean_absolute_deviation'] = np.mean(np.abs(actual - predicted))
    metrics['median_absolute_error'] = np.median(np.abs(actual - predicted))
    
    # Percentage metrics
    metrics['mean_relative_error'] = np.mean(np.abs((actual - predicted) / actual)) * 100
    metrics['median_relative_error'] = np.median(np.abs((actual - predicted) / actual)) * 100
    
    # Bias metrics
    metrics['bias'] = np.mean(predicted - actual)
    metrics['relative_bias'] = np.mean((predicted - actual) / actual) * 100
    
    return metrics

def evaluate_model_performance():
    """Evaluate the gap predictor model performance."""
    print("Evaluating gap predictor model performance...")
    
    # Load data
    df = load_test_data()
    if df is None:
        print("Could not load test data")
        return None, None
    
    # Initialize predictor with correct sequence length
    predictor = GapPredictor(sequence_length=96, output_length=96)  # 1 day = 96 quarters
    
    if not predictor.is_trained:
        print("Model is not trained. Cannot evaluate performance.")
        return None, None
    
    # Test dates for evaluation
    test_dates = [
        '2021-06-15', '2021-07-20', '2021-08-15', 
        '2021-09-10', '2021-10-05', '2021-11-20'
    ]
    
    all_metrics = {
        'generation': {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'r2': []},
        'demand': {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'r2': []}
    }
    
    predictions_data = []
    
    for test_date in test_dates:
        try:
            # Get input sequence (1 day = 96 quarter-hourly intervals)
            end_date = pd.to_datetime(test_date)
            start_date = end_date - pd.Timedelta(days=1)  # 1 day for 96 intervals
            
            input_sequence = df.loc[start_date:end_date - pd.Timedelta(minutes=15)][['generation', 'demand']].values
            
            if len(input_sequence) != 96:
                print(f"Skipping {test_date}: Need exactly 96 intervals, got {len(input_sequence)}")
                continue
            
            # Make prediction
            predicted = predictor.predict(input_sequence)
            actual = df.loc[test_date][['generation', 'demand']].values
            
            # Calculate metrics for each feature
            for i, feature in enumerate(['generation', 'demand']):
                metrics = calculate_prediction_metrics(actual[:, i], predicted[:, i])
                
                for metric_name in ['mse', 'rmse', 'mae', 'mape', 'r2']:
                    all_metrics[feature][metric_name].append(metrics[metric_name])
            
            # Store predictions for analysis
            predictions_data.append({
                'date': test_date,
                'actual_generation': actual[:, 0],
                'predicted_generation': predicted[:, 0],
                'actual_demand': actual[:, 1],
                'predicted_demand': predicted[:, 1]
            })
            
            print(f"✓ Evaluated {test_date}")
            
        except Exception as e:
            print(f"✗ Error evaluating {test_date}: {e}")
    
    return all_metrics, predictions_data

def calculate_model_statistics():
    """Calculate comprehensive model statistics."""
    print("Calculating model statistics...")
    
    predictor = GapPredictor(sequence_length=96, output_length=96)
    
    if not predictor.is_trained:
        predictor.build_model()
    
    model = predictor.model
    
    # Model complexity statistics
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Layer statistics
    layer_stats = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type not in layer_stats:
            layer_stats[layer_type] = {'count': 0, 'params': 0}
        layer_stats[layer_type]['count'] += 1
        layer_stats[layer_type]['params'] += layer.count_params()
    
    # Model architecture statistics
    architecture_stats = {
        'total_layers': len(model.layers),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'parameter_efficiency': trainable_params / total_params * 100,
        'layer_distribution': layer_stats,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'model_depth': len([l for l in model.layers if hasattr(l, 'units')])
    }
    
    return architecture_stats

def create_kpi_dashboard(metrics, predictions_data, architecture_stats):
    """Create a comprehensive KPI dashboard."""
    print("Creating KPI dashboard...")
    
    # Create subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Performance Metrics Summary
    ax1 = fig.add_subplot(gs[0, :2])
    
    if metrics and any(len(metrics['generation']['rmse']) > 0 for feature in ['generation', 'demand']):
        # Calculate average metrics
        avg_metrics = {}
        for feature in ['generation', 'demand']:
            avg_metrics[feature] = {}
            for metric in ['rmse', 'mae', 'mape', 'r2']:
                if len(metrics[feature][metric]) > 0:
                    avg_metrics[feature][metric] = np.mean(metrics[feature][metric])
                else:
                    avg_metrics[feature][metric] = 0
        
        # Create comparison table
        metrics_data = []
        for feature in ['generation', 'demand']:
            metrics_data.append([
                feature.capitalize(),
                f"{avg_metrics[feature]['rmse']:.2f}",
                f"{avg_metrics[feature]['mae']:.2f}",
                f"{avg_metrics[feature]['mape']:.2f}%",
                f"{avg_metrics[feature]['r2']:.3f}"
            ])
        
        table = ax1.table(cellText=metrics_data,
                         colLabels=['Feature', 'RMSE', 'MAE', 'MAPE', 'R²'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(metrics_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax1.set_title('Model Performance Metrics', fontsize=16, fontweight='bold')
        ax1.axis('off')
    else:
        ax1.text(0.5, 0.5, 'No performance data available', ha='center', va='center', fontsize=14)
        ax1.set_title('Model Performance Metrics', fontsize=16, fontweight='bold')
        ax1.axis('off')
    
    # 2. Model Architecture Summary
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    arch_text = f"""
    Model Architecture Summary:
    
    Total Parameters: {architecture_stats['total_parameters']:,}
    Trainable Parameters: {architecture_stats['trainable_parameters']:,}
    Non-trainable Parameters: {architecture_stats['non_trainable_parameters']:,}
    Parameter Efficiency: {architecture_stats['parameter_efficiency']:.1f}%
    
    Total Layers: {architecture_stats['total_layers']}
    Model Depth: {architecture_stats['model_depth']}
    
    Input Shape: {architecture_stats['input_shape']}
    Output Shape: {architecture_stats['output_shape']}
    """
    
    ax2.text(0.05, 0.95, arch_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 3. Prediction vs Actual (Generation)
    ax3 = fig.add_subplot(gs[1, 0])
    if predictions_data:
        # Plot first prediction as example
        data = predictions_data[0]
        time_steps = range(len(data['actual_generation']))
        
        ax3.plot(time_steps, data['actual_generation'], 'b-', label='Actual', linewidth=2)
        ax3.plot(time_steps, data['predicted_generation'], 'r--', label='Predicted', linewidth=2)
        ax3.set_title('Generation: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Generation (kWh)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', fontsize=12)
        ax3.set_title('Generation: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax3.axis('off')
    
    # 4. Prediction vs Actual (Demand)
    ax4 = fig.add_subplot(gs[1, 1])
    if predictions_data:
        ax4.plot(time_steps, data['actual_demand'], 'g-', label='Actual', linewidth=2)
        ax4.plot(time_steps, data['predicted_demand'], 'orange', linestyle='--', label='Predicted', linewidth=2)
        ax4.set_title('Demand: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Demand (kWh)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', fontsize=12)
        ax4.set_title('Demand: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax4.axis('off')
    
    # 5. Error Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    if predictions_data:
        errors_generation = data['actual_generation'] - data['predicted_generation']
        errors_demand = data['actual_demand'] - data['predicted_demand']
        
        ax5.hist(errors_generation, alpha=0.7, label='Generation', bins=20, color='blue')
        ax5.hist(errors_demand, alpha=0.7, label='Demand', bins=20, color='green')
        ax5.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Prediction Error')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No error data available', ha='center', va='center', fontsize=12)
        ax5.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        ax5.axis('off')
    
    # 6. Layer Parameter Distribution
    ax6 = fig.add_subplot(gs[2, 0])
    layer_types = list(architecture_stats['layer_distribution'].keys())
    param_counts = [architecture_stats['layer_distribution'][lt]['params'] for lt in layer_types]
    
    bars = ax6.bar(layer_types, param_counts, color=plt.cm.Set3(np.linspace(0, 1, len(layer_types))))
    ax6.set_title('Parameters by Layer Type', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Number of Parameters')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(param_counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    # 7. Model Complexity Gauge
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis('off')
    
    # Create a simple gauge chart
    total_params = architecture_stats['total_parameters']
    complexity_level = min(total_params / 500000, 1.0)  # Normalize to 0-1
    
    # Draw gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax7.plot(x, y, 'k-', linewidth=3)
    ax7.fill_between(x, 0, y, alpha=0.3, color='lightgray')
    
    # Fill based on complexity
    fill_theta = np.linspace(0, np.pi * complexity_level, 100)
    fill_x = r * np.cos(fill_theta)
    fill_y = r * np.sin(fill_theta)
    ax7.fill_between(fill_x, 0, fill_y, alpha=0.7, color='red')
    
    ax7.text(0, 0.5, f'{total_params:,}\nParameters', ha='center', va='center', 
             fontsize=12, fontweight='bold')
    ax7.set_xlim(-1.2, 1.2)
    ax7.set_ylim(-0.2, 1.2)
    ax7.set_title('Model Complexity', fontsize=12, fontweight='bold')
    
    # 8. Performance Trends (if data available)
    ax8 = fig.add_subplot(gs[2, 2])
    if metrics and len(metrics['generation']['rmse']) > 0:
        dates = ['2021-06-15', '2021-07-20', '2021-08-15', '2021-09-10', '2021-10-05', '2021-11-20']
        rmse_gen = metrics['generation']['rmse']
        rmse_dem = metrics['demand']['rmse']
        
        x = range(len(rmse_gen))
        ax8.plot(x, rmse_gen, 'bo-', label='Generation RMSE', linewidth=2)
        ax8.plot(x, rmse_dem, 'go-', label='Demand RMSE', linewidth=2)
        ax8.set_title('RMSE Performance Over Time', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Test Date')
        ax8.set_ylabel('RMSE')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.set_xticks(x)
        ax8.set_xticklabels([d[5:] for d in dates[:len(rmse_gen)]], rotation=45)
    else:
        ax8.text(0.5, 0.5, 'No performance trend data available', ha='center', va='center', fontsize=12)
        ax8.set_title('RMSE Performance Over Time', fontsize=12, fontweight='bold')
        ax8.axis('off')
    
    # 9. KPI Summary
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')
    
    if metrics and any(len(metrics['generation']['rmse']) > 0 for feature in ['generation', 'demand']):
        # Calculate overall KPIs
        overall_rmse_gen = np.mean(metrics['generation']['rmse']) if len(metrics['generation']['rmse']) > 0 else 0
        overall_rmse_dem = np.mean(metrics['demand']['rmse']) if len(metrics['demand']['rmse']) > 0 else 0
        overall_r2_gen = np.mean(metrics['generation']['r2']) if len(metrics['generation']['r2']) > 0 else 0
        overall_r2_dem = np.mean(metrics['demand']['r2']) if len(metrics['demand']['r2']) > 0 else 0
        overall_mape_gen = np.mean(metrics['generation']['mape']) if len(metrics['generation']['mape']) > 0 else 0
        overall_mape_dem = np.mean(metrics['demand']['mape']) if len(metrics['demand']['mape']) > 0 else 0
        
        kpi_text = f"""
        GAP PREDICTOR MODEL - KEY PERFORMANCE INDICATORS
        
        📊 Overall Performance Metrics:
        • Generation RMSE: {overall_rmse_gen:.2f} kWh
        • Demand RMSE: {overall_rmse_dem:.2f} kWh
        • Generation R² Score: {overall_r2_gen:.3f}
        • Demand R² Score: {overall_r2_dem:.3f}
        • Generation MAPE: {overall_mape_gen:.2f}%
        • Demand MAPE: {overall_mape_dem:.2f}%
        
        🏗️ Model Architecture:
        • Total Parameters: {architecture_stats['total_parameters']:,}
        • Model Depth: {architecture_stats['model_depth']} layers
        • Parameter Efficiency: {architecture_stats['parameter_efficiency']:.1f}%
        
        🎯 Model Assessment:
        • R² > 0.8: {'✓ Excellent' if overall_r2_gen > 0.8 and overall_r2_dem > 0.8 else '✗ Needs improvement'}
        • MAPE < 10%: {'✓ Good' if overall_mape_gen < 10 and overall_mape_dem < 10 else '✗ High error'}
        • Model Complexity: {'✓ Appropriate' if 100000 < architecture_stats['total_parameters'] < 500000 else '⚠️ Review needed'}
        """
    else:
        kpi_text = f"""
        GAP PREDICTOR MODEL - KEY PERFORMANCE INDICATORS
        
        📊 Performance Metrics: Not available (model evaluation failed)
        
        🏗️ Model Architecture:
        • Total Parameters: {architecture_stats['total_parameters']:,}
        • Model Depth: {architecture_stats['model_depth']} layers
        • Parameter Efficiency: {architecture_stats['parameter_efficiency']:.1f}%
        
        🎯 Model Assessment:
        • Architecture: ✓ Well-structured LSTM model
        • Complexity: {'✓ Appropriate' if 100000 < architecture_stats['total_parameters'] < 500000 else '⚠️ Review needed'}
        """
    
    ax9.text(0.05, 0.95, kpi_text, transform=ax9.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.suptitle('Gap Predictor Model - KPI Dashboard', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save the dashboard
    plt.savefig('gap_predictor_kpi_dashboard.png', dpi=300, bbox_inches='tight')
    print("KPI dashboard saved as gap_predictor_kpi_dashboard.png")
    plt.show()

def generate_kpi_report(metrics, predictions_data, architecture_stats):
    """Generate a comprehensive KPI report."""
    print("Generating KPI report...")
    
    report = []
    report.append("=" * 80)
    report.append("GAP PREDICTOR MODEL - KPI ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Model Architecture Summary
    report.append("1. MODEL ARCHITECTURE SUMMARY")
    report.append("-" * 40)
    report.append(f"Total Parameters: {architecture_stats['total_parameters']:,}")
    report.append(f"Trainable Parameters: {architecture_stats['trainable_parameters']:,}")
    report.append(f"Non-trainable Parameters: {architecture_stats['non_trainable_parameters']:,}")
    report.append(f"Parameter Efficiency: {architecture_stats['parameter_efficiency']:.1f}%")
    report.append(f"Total Layers: {architecture_stats['total_layers']}")
    report.append(f"Model Depth: {architecture_stats['model_depth']}")
    report.append(f"Input Shape: {architecture_stats['input_shape']}")
    report.append(f"Output Shape: {architecture_stats['output_shape']}")
    report.append("")
    
    # Layer Distribution
    report.append("2. LAYER DISTRIBUTION")
    report.append("-" * 40)
    for layer_type, stats in architecture_stats['layer_distribution'].items():
        report.append(f"{layer_type}: {stats['count']} layers, {stats['params']:,} parameters")
    report.append("")
    
    # Performance Metrics
    if metrics and any(len(metrics['generation']['rmse']) > 0 for feature in ['generation', 'demand']):
        report.append("3. PERFORMANCE METRICS")
        report.append("-" * 40)
        
        for feature in ['generation', 'demand']:
            report.append(f"\n{feature.upper()} PREDICTION:")
            avg_metrics = {}
            for metric in ['rmse', 'mae', 'mape', 'r2']:
                if len(metrics[feature][metric]) > 0:
                    avg_metrics[metric] = np.mean(metrics[feature][metric])
                else:
                    avg_metrics[metric] = 0
            
            report.append(f"  RMSE: {avg_metrics['rmse']:.2f} kWh")
            report.append(f"  MAE: {avg_metrics['mae']:.2f} kWh")
            report.append(f"  MAPE: {avg_metrics['mape']:.2f}%")
            report.append(f"  R² Score: {avg_metrics['r2']:.3f}")
        
        # Overall performance assessment
        overall_r2_gen = np.mean(metrics['generation']['r2']) if len(metrics['generation']['r2']) > 0 else 0
        overall_r2_dem = np.mean(metrics['demand']['r2']) if len(metrics['demand']['r2']) > 0 else 0
        overall_mape_gen = np.mean(metrics['generation']['mape']) if len(metrics['generation']['mape']) > 0 else 0
        overall_mape_dem = np.mean(metrics['demand']['mape']) if len(metrics['demand']['mape']) > 0 else 0
        
        report.append("\n4. OVERALL ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Generation R² Score: {overall_r2_gen:.3f} ({'Excellent' if overall_r2_gen > 0.8 else 'Good' if overall_r2_gen > 0.6 else 'Needs Improvement'})")
        report.append(f"Demand R² Score: {overall_r2_dem:.3f} ({'Excellent' if overall_r2_dem > 0.8 else 'Good' if overall_r2_dem > 0.6 else 'Needs Improvement'})")
        report.append(f"Generation MAPE: {overall_mape_gen:.2f}% ({'Excellent' if overall_mape_gen < 5 else 'Good' if overall_mape_gen < 10 else 'Needs Improvement'})")
        report.append(f"Demand MAPE: {overall_mape_dem:.2f}% ({'Excellent' if overall_mape_dem < 5 else 'Good' if overall_mape_dem < 10 else 'Needs Improvement'})")
        
        # Model complexity assessment
        total_params = architecture_stats['total_parameters']
        report.append(f"\nModel Complexity: {total_params:,} parameters ({'Appropriate' if 100000 < total_params < 500000 else 'High' if total_params > 500000 else 'Low'})")
    else:
        report.append("3. PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append("Performance metrics could not be calculated due to model evaluation issues.")
        report.append("This may be due to data availability or model configuration.")
        report.append("")
        
        report.append("4. MODEL ASSESSMENT")
        report.append("-" * 40)
        total_params = architecture_stats['total_parameters']
        report.append(f"Model Complexity: {total_params:,} parameters ({'Appropriate' if 100000 < total_params < 500000 else 'High' if total_params > 500000 else 'Low'})")
        report.append("Architecture: Well-structured LSTM model with appropriate layer distribution")
    
    report.append("\n" + "=" * 80)
    
    # Save report
    with open('gap_predictor_kpi_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("KPI report saved as gap_predictor_kpi_report.txt")
    
    # Print summary
    print("\n" + "=" * 80)
    print("GAP PREDICTOR MODEL - KPI SUMMARY")
    print("=" * 80)
    
    if metrics and any(len(metrics['generation']['rmse']) > 0 for feature in ['generation', 'demand']):
        overall_r2_gen = np.mean(metrics['generation']['r2']) if len(metrics['generation']['r2']) > 0 else 0
        overall_r2_dem = np.mean(metrics['demand']['r2']) if len(metrics['demand']['r2']) > 0 else 0
        overall_mape_gen = np.mean(metrics['generation']['mape']) if len(metrics['generation']['mape']) > 0 else 0
        overall_mape_dem = np.mean(metrics['demand']['mape']) if len(metrics['demand']['mape']) > 0 else 0
        
        print(f"📊 Performance Metrics:")
        print(f"   • Generation R²: {overall_r2_gen:.3f}")
        print(f"   • Demand R²: {overall_r2_dem:.3f}")
        print(f"   • Generation MAPE: {overall_mape_gen:.2f}%")
        print(f"   • Demand MAPE: {overall_mape_dem:.2f}%")
    else:
        print(f"📊 Performance Metrics: Not available")
    
    print(f"🏗️ Model Architecture:")
    print(f"   • Total Parameters: {architecture_stats['total_parameters']:,}")
    print(f"   • Parameter Efficiency: {architecture_stats['parameter_efficiency']:.1f}%")
    print(f"   • Model Depth: {architecture_stats['model_depth']} layers")
    
    return report

if __name__ == "__main__":
    print("Analyzing Gap Predictor Model KPIs...")
    
    # Evaluate model performance
    metrics, predictions_data = evaluate_model_performance()
    
    # Calculate model statistics
    architecture_stats = calculate_model_statistics()
    
    # Create KPI dashboard
    create_kpi_dashboard(metrics, predictions_data, architecture_stats)
    
    # Generate KPI report
    generate_kpi_report(metrics, predictions_data, architecture_stats)
    
    print("\nKPI analysis completed!")
