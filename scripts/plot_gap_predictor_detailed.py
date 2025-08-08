import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Suppress TensorFlow warnings and debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.gap_predictor import GapPredictor

def plot_detailed_model_architecture():
    """Create a detailed visualization of the LSTM model architecture."""
    print("Creating detailed model architecture visualization...")
    
    # Initialize the predictor
    predictor = GapPredictor(sequence_length=192, output_length=96)
    
    if not predictor.is_trained:
        print("Building new model for visualization...")
        predictor.build_model()
    
    model = predictor.model
    
    # Create a comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Model summary as text
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    
    # Get model summary as string
    from io import StringIO
    summary_io = StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
    summary_str = summary_io.getvalue()
    
    ax1.text(0.05, 0.95, 'Model Architecture Summary', 
             transform=ax1.transAxes, fontsize=16, fontweight='bold')
    ax1.text(0.05, 0.85, summary_str, transform=ax1.transAxes, 
             fontsize=10, fontfamily='monospace', verticalalignment='top')
    
    # Plot 2: Layer connectivity diagram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title('Layer Connectivity', fontsize=14, fontweight='bold')
    
    # Create a simple connectivity diagram
    layers = [layer.name for layer in model.layers]
    y_pos = np.linspace(0, 1, len(layers))
    
    for i, (layer, y) in enumerate(zip(layers, y_pos)):
        ax2.text(0.5, y, layer, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        if i < len(layers) - 1:
            ax2.arrow(0.5, y - 0.05, 0, y_pos[i+1] - y - 0.1, 
                     head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axis('off')
    
    # Plot 3: Parameter distribution by layer
    ax3 = fig.add_subplot(gs[1, 0])
    
    layer_params = []
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
        layer_params.append(layer.count_params())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    bars = ax3.barh(layer_names, layer_params, color=colors)
    ax3.set_title('Parameters per Layer', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Parameters')
    
    # Add value labels
    for bar, param_count in zip(bars, layer_params):
        width = bar.get_width()
        ax3.text(width + max(layer_params) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{param_count:,}', ha='left', va='center', fontsize=8)
    
    # Plot 4: Layer types distribution
    ax4 = fig.add_subplot(gs[1, 1])
    
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    unique_types = list(set(layer_types))
    type_counts = [layer_types.count(t) for t in unique_types]
    
    wedges, texts, autotexts = ax4.pie(type_counts, labels=unique_types, autopct='%1.1f%%',
                                       colors=plt.cm.Set3(np.linspace(0, 1, len(unique_types))))
    ax4.set_title('Layer Types Distribution', fontsize=12, fontweight='bold')
    
    # Plot 5: Model complexity metrics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    metrics_text = f"""
    Model Complexity Metrics:
    
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    Non-trainable Parameters: {non_trainable_params:,}
    
    Model Depth: {len(model.layers)} layers
    Input Shape: (192, 2)
    Output Shape: (96, 2)
    
    Architecture Type: Bidirectional LSTM
    with Dropout Regularization
    """
    
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, 
             fontsize=11, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
    
    # Plot 6: LSTM cell visualization
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create a simplified LSTM cell diagram
    lstm_components = ['Input Gate', 'Forget Gate', 'Cell State', 'Output Gate', 'Hidden State']
    component_positions = np.linspace(0, 1, len(lstm_components))
    
    for i, (component, pos) in enumerate(zip(lstm_components, component_positions)):
        circle = plt.Circle((pos, 0.5), 0.08, color='lightcoral', alpha=0.7)
        ax6.add_patch(circle)
        ax6.text(pos, 0.5, component, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add arrows between components
        if i < len(lstm_components) - 1:
            ax6.arrow(pos + 0.08, 0.5, component_positions[i+1] - pos - 0.16, 0,
                     head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    ax6.set_xlim(-0.1, 1.1)
    ax6.set_ylim(0, 1)
    ax6.set_title('LSTM Cell Components (Simplified)', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # Add overall title
    fig.suptitle('Gap Predictor LSTM Model Architecture - Detailed Analysis', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = 'gap_predictor_detailed_architecture.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Detailed model architecture plot saved as {plot_path}")
    
    plt.show()
    
    return model

def create_model_info_table():
    """Create a comprehensive model information table."""
    predictor = GapPredictor(sequence_length=192, output_length=96)
    
    if not predictor.is_trained:
        predictor.build_model()
    
    model = predictor.model
    
    # Create a table with model information
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.axis('off')
    
    # Model information
    model_info = [
        ['Model Type', 'Bidirectional LSTM with Dropout'],
        ['Input Shape', '(192, 2) - 2 days of quarter-hourly data'],
        ['Output Shape', '(96, 2) - 1 day of quarter-hourly predictions'],
        ['Total Parameters', f'{model.count_params():,}'],
        ['Trainable Parameters', f'{sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}'],
        ['Non-trainable Parameters', '0'],
        ['Number of Layers', str(len(model.layers))],
        ['LSTM Units (Layer 1)', '128 (Bidirectional)'],
        ['LSTM Units (Layer 2)', '64 (Bidirectional)'],
        ['LSTM Units (Layer 3)', '32 (Bidirectional)'],
        ['Dropout Rate (Layer 1)', '0.3'],
        ['Dropout Rate (Layer 2)', '0.3'],
        ['Dropout Rate (Layer 3)', '0.2'],
        ['Dense Layer 1', '64 units (ReLU)'],
        ['Dense Layer 2', '192 units (Linear)'],
        ['Optimizer', 'Adam (learning_rate=0.0005)'],
        ['Loss Function', 'Mean Squared Error'],
        ['Training Data', 'Historical generation and demand data'],
        ['Prediction Task', 'Next day generation and demand forecasting']
    ]
    
    # Create table
    table = ax.table(cellText=model_info, colLabels=['Property', 'Value'],
                    cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(model_info) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax.set_title('Gap Predictor Model Specifications', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = 'gap_predictor_model_specifications.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Model specifications table saved as {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating detailed gap predictor model architecture visualizations...")
    
    # Create detailed architecture visualization
    model = plot_detailed_model_architecture()
    
    # Create model specifications table
    create_model_info_table()
    
    print("\nDetailed visualizations completed!")
