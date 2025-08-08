import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import seaborn as sns

# Suppress TensorFlow warnings and debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.gap_predictor import GapPredictor

def create_professional_lstm_architecture():
    """Create a professional LSTM architecture visualization."""
    print("Creating professional LSTM architecture visualization...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define layers with professional styling
    layers = [
        {'name': 'Input\n(192, 2)', 'pos': (1, 8), 'color': '#2E86AB', 'size': 1.2, 'type': 'input'},
        {'name': 'Bidirectional\nLSTM 128', 'pos': (3, 8), 'color': '#A23B72', 'size': 1.5, 'type': 'lstm'},
        {'name': 'Dropout\n0.3', 'pos': (5, 8), 'color': '#F18F01', 'size': 0.8, 'type': 'dropout'},
        {'name': 'Bidirectional\nLSTM 64', 'pos': (7, 8), 'color': '#C73E1D', 'size': 1.3, 'type': 'lstm'},
        {'name': 'Dropout\n0.3', 'pos': (9, 8), 'color': '#F18F01', 'size': 0.8, 'type': 'dropout'},
        {'name': 'Bidirectional\nLSTM 32', 'pos': (2, 5), 'color': '#A23B72', 'size': 1.1, 'type': 'lstm'},
        {'name': 'Dropout\n0.2', 'pos': (4, 5), 'color': '#F18F01', 'size': 0.8, 'type': 'dropout'},
        {'name': 'Dense 64\n(ReLU)', 'pos': (6, 5), 'color': '#2E86AB', 'size': 1.0, 'type': 'dense'},
        {'name': 'Dense 192', 'pos': (8, 5), 'color': '#2E86AB', 'size': 1.0, 'type': 'dense'},
        {'name': 'Reshape\n(96, 2)', 'pos': (5, 2), 'color': '#C73E1D', 'size': 1.0, 'type': 'reshape'},
        {'name': 'Output\n(96, 2)', 'pos': (5, 0.5), 'color': '#2E86AB', 'size': 1.0, 'type': 'output'}
    ]
    
    # Draw layers with gradient effects
    for layer in layers:
        x, y = layer['pos']
        size = layer['size']
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x - size/2, y - size/2), size, size,
            boxstyle="round,pad=0.1",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add text with shadow effect
        ax.text(x, y, layer['name'], ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.3))
    
    # Add connections with arrows
    connections = [
        ((1, 8), (3, 8)),
        ((3, 8), (5, 8)),
        ((5, 8), (7, 8)),
        ((7, 8), (9, 8)),
        ((9, 8), (2, 5)),
        ((2, 5), (4, 5)),
        ((4, 5), (6, 5)),
        ((6, 5), (8, 5)),
        ((8, 5), (5, 2)),
        ((5, 2), (5, 0.5))
    ]
    
    for start, end in connections:
        # Draw arrow with gradient
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color='black',
                                 connectionstyle="arc3,rad=0.1"))
    
    # Add title with professional styling
    ax.text(6, 9.5, 'Gap Predictor LSTM Architecture', 
           ha='center', va='center', fontsize=20, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Add legend
    legend_elements = [
        patches.Patch(color='#2E86AB', label='Input/Output/Dense'),
        patches.Patch(color='#A23B72', label='LSTM Layers'),
        patches.Patch(color='#F18F01', label='Dropout'),
        patches.Patch(color='#C73E1D', label='Reshape')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.9),
             fontsize=12, framealpha=0.9)
    
    # Add model info
    info_text = """
    Model Specifications:
    • Input: 192 time steps × 2 features
    • Output: 96 time steps × 2 features  
    • Total Parameters: 356,352
    • Architecture: Bidirectional LSTM
    • Optimizer: Adam (lr=0.0005)
    • Loss: Mean Squared Error
    """
    
    ax.text(0.5, 1, info_text, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('gap_predictor_professional_architecture.png', dpi=300, bbox_inches='tight')
    print("Professional LSTM architecture saved as gap_predictor_professional_architecture.png")
    plt.show()

def create_lstm_cell_advanced():
    """Create an advanced LSTM cell visualization."""
    print("Creating advanced LSTM cell visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # LSTM cell components with advanced styling
    components = [
        {'name': 'Input Gate', 'pos': (2, 6), 'color': '#2E86AB', 'symbol': 'σ', 'formula': 'i_t = σ(W_i · [h_{t-1}, x_t] + b_i)'},
        {'name': 'Forget Gate', 'pos': (4, 6), 'color': '#A23B72', 'symbol': 'σ', 'formula': 'f_t = σ(W_f · [h_{t-1}, x_t] + b_f)'},
        {'name': 'Cell State', 'pos': (6, 6), 'color': '#C73E1D', 'symbol': 'tanh', 'formula': 'C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)'},
        {'name': 'Output Gate', 'pos': (8, 6), 'color': '#F18F01', 'symbol': 'σ', 'formula': 'o_t = σ(W_o · [h_{t-1}, x_t] + b_o)'},
        {'name': 'Hidden State', 'pos': (10, 6), 'color': '#2E86AB', 'symbol': 'h', 'formula': 'h_t = o_t ⊙ tanh(C_t)'}
    ]
    
    # Draw components with gradient effects
    for comp in components:
        x, y = comp['pos']
        
        # Create circle with gradient effect
        circle = Circle((x, y), 0.8, facecolor=comp['color'], edgecolor='black', linewidth=3, alpha=0.8)
        ax.add_patch(circle)
        
        # Add component name with shadow
        ax.text(x, y + 0.4, comp['name'], ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.3))
        
        # Add mathematical symbol
        ax.text(x, y, comp['symbol'], ha='center', va='center', 
               fontsize=16, fontweight='bold', color='white')
        
        # Add formula
        ax.text(x, y - 0.4, comp['formula'], ha='center', va='center', 
               fontsize=9, color='darkblue', fontweight='bold')
    
    # Add connections with curved arrows
    for i in range(len(components) - 1):
        x1, y1 = components[i]['pos']
        x2, y2 = components[i+1]['pos']
        ax.annotate('', xy=(x2-0.8, y2), xytext=(x1+0.8, y1),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black',
                                 connectionstyle="arc3,rad=0.2"))
    
    # Add cell state flow with special styling
    ax.plot([6, 6], [5.2, 2], '--', color='red', linewidth=4, alpha=0.7)
    ax.text(6.5, 3.5, 'Cell State\nFlow', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='red',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add title with professional styling
    ax.text(7, 7.5, 'LSTM Cell Architecture', 
           ha='center', va='center', fontsize=18, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Add detailed description
    description = """
    LSTM (Long Short-Term Memory) cells contain specialized gates that control information flow:
    
    • Input Gate (i_t): Controls what new information to store in the cell state
    • Forget Gate (f_t): Controls what information to discard from the cell state  
    • Cell State (C_t): The memory line that runs through the entire sequence
    • Output Gate (o_t): Controls what information to output from the cell state
    • Hidden State (h_t): The output of the cell that is passed to the next time step
    
    The cell state acts as a "conveyor belt" that runs through the entire sequence,
    with gates controlling what information flows along it.
    """
    
    ax.text(0.5, 1, description, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('lstm_cell_advanced.png', dpi=300, bbox_inches='tight')
    print("Advanced LSTM cell visualization saved as lstm_cell_advanced.png")
    plt.show()

def create_parameter_analysis():
    """Create a comprehensive parameter analysis visualization."""
    print("Creating parameter analysis visualization...")
    
    # Get model data
    predictor = GapPredictor(sequence_length=192, output_length=96)
    if not predictor.is_trained:
        predictor.build_model()
    
    model = predictor.model
    
    # Extract layer information
    layer_names = []
    layer_params = []
    layer_types = []
    
    for layer in model.layers:
        layer_names.append(layer.name)
        layer_params.append(layer.count_params())
        layer_types.append(layer.__class__.__name__)
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Parameter distribution by layer
    colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))
    bars = ax1.barh(layer_names, layer_params, color=colors)
    ax1.set_title('Parameters per Layer', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Parameters')
    
    # Add value labels
    for bar, param_count in zip(bars, layer_params):
        width = bar.get_width()
        ax1.text(width + max(layer_params) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{param_count:,}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Parameter distribution pie chart
    wedges, texts, autotexts = ax2.pie(layer_params, labels=layer_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax2.set_title('Parameter Distribution', fontsize=14, fontweight='bold')
    
    # Plot 3: Layer types
    unique_types = list(set(layer_types))
    type_counts = [layer_types.count(t) for t in unique_types]
    type_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    
    bars = ax3.bar(unique_types, type_counts, color=type_colors)
    ax3.set_title('Layer Types Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Layers')
    
    # Add value labels
    for bar, count in zip(bars, type_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Model complexity metrics
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    metrics_data = ['Trainable', 'Non-trainable']
    metrics_values = [trainable_params, non_trainable_params]
    colors_metrics = ['#2E86AB', '#A23B72']
    
    wedges, texts, autotexts = ax4.pie(metrics_values, labels=metrics_data, autopct='%1.1f%%',
                                       colors=colors_metrics, startangle=90)
    ax4.set_title('Trainable vs Non-trainable Parameters', fontsize=14, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Gap Predictor Model Parameter Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gap_predictor_parameter_analysis.png', dpi=300, bbox_inches='tight')
    print("Parameter analysis saved as gap_predictor_parameter_analysis.png")
    plt.show()

if __name__ == "__main__":
    print("Creating advanced LSTM architecture visualizations...")
    
    # Create all advanced visualizations
    create_professional_lstm_architecture()
    create_lstm_cell_advanced()
    create_parameter_analysis()
    
    print("\nAll advanced visualizations completed!")
    print("These are much more professional and visually appealing!")
