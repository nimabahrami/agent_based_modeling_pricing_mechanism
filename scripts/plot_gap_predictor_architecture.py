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

def plot_model_architecture():
    """Plot the LSTM model architecture."""
    print("Loading gap predictor model...")
    
    # Initialize the predictor
    predictor = GapPredictor(sequence_length=192, output_length=96)
    
    if not predictor.is_trained:
        print("No trained model found. Building a new model for visualization...")
        predictor.build_model()
    
    model = predictor.model
    
    if model is None:
        print("Error: Could not load or build model")
        return
    
    print(f"Model architecture:")
    print(model.summary())
    
    # Create a visual representation of the model architecture
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Layer structure visualization
    layers = []
    layer_types = []
    layer_units = []
    
    for layer in model.layers:
        layer_types.append(layer.__class__.__name__)
        if hasattr(layer, 'units'):
            layer_units.append(layer.units)
        elif hasattr(layer, 'output_shape'):
            layer_units.append(str(layer.output_shape))
        else:
            layer_units.append('N/A')
        layers.append(layer.name)
    
    # Create bar chart of layer types
    unique_types = list(set(layer_types))
    type_counts = [layer_types.count(t) for t in unique_types]
    
    bars1 = ax1.bar(unique_types, type_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax1.set_title('Gap Predictor Model Architecture - Layer Types', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Layers')
    ax1.set_xlabel('Layer Type')
    
    # Add value labels on bars
    for bar, count in zip(bars1, type_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Detailed layer information
    y_pos = np.arange(len(layers))
    colors = plt.cm.Set3(np.linspace(0, 1, len(layers)))
    
    bars2 = ax2.barh(y_pos, [1]*len(layers), color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(layers, fontsize=8)
    ax2.set_xlabel('Layer')
    ax2.set_title('Model Layers Detail', fontsize=14, fontweight='bold')
    
    # Add layer information as text
    for i, (layer, units) in enumerate(zip(layers, layer_units)):
        ax2.text(0.5, i, f'{layer_types[i]}: {units}', 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Remove x-axis ticks for cleaner look
    ax2.set_xticks([])
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = 'gap_predictor_architecture.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Model architecture plot saved as {plot_path}")
    
    plt.show()
    
    return model

def plot_model_flow():
    """Create a flow diagram of the model architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Define the model flow
    layers_info = [
        ("Input", "Shape: (192, 2)", "#FF6B6B"),
        ("Bidirectional LSTM", "128 units", "#4ECDC4"),
        ("Dropout", "0.3", "#45B7D1"),
        ("Bidirectional LSTM", "64 units", "#96CEB4"),
        ("Dropout", "0.3", "#FFEAA7"),
        ("Bidirectional LSTM", "32 units", "#DDA0DD"),
        ("Dropout", "0.2", "#98D8C8"),
        ("Dense", "64 units (ReLU)", "#F7DC6F"),
        ("Dense", "192 units", "#BB8FCE"),
        ("Reshape", "(96, 2)", "#85C1E9"),
        ("Output", "Shape: (96, 2)", "#F8C471")
    ]
    
    # Create flow diagram
    y_positions = np.linspace(0, 1, len(layers_info))
    
    for i, (layer_name, layer_desc, color) in enumerate(layers_info):
        # Draw rectangle for each layer
        rect = plt.Rectangle((0.1, y_positions[i] - 0.03), 0.8, 0.06, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add layer name
        ax.text(0.5, y_positions[i], layer_name, ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        # Add layer description
        ax.text(0.5, y_positions[i] - 0.015, layer_desc, ha='center', va='center', 
               fontsize=10)
        
        # Draw arrows between layers
        if i < len(layers_info) - 1:
            ax.arrow(0.5, y_positions[i] - 0.03, 0, 
                    y_positions[i+1] - y_positions[i] - 0.06, 
                    head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Gap Predictor LSTM Model Flow', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add model description
    description = """
    Model Architecture:
    • Input: 192 time steps × 2 features (generation, demand)
    • Bidirectional LSTM layers for temporal pattern recognition
    • Dropout layers for regularization
    • Dense layers for feature transformation
    • Output: 96 time steps × 2 features (next day prediction)
    """
    
    ax.text(0.02, -0.1, description, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = 'gap_predictor_flow_diagram.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Model flow diagram saved as {plot_path}")
    
    plt.show()

def plot_model_parameters():
    """Plot model parameters and complexity."""
    predictor = GapPredictor(sequence_length=192, output_length=96)
    
    if not predictor.is_trained:
        predictor.build_model()
    
    model = predictor.model
    
    # Get model parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Create parameter breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Parameter distribution
    labels = ['Trainable', 'Non-trainable']
    sizes = [trainable_params, non_trainable_params]
    colors = ['#FF6B6B', '#4ECDC4']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90)
    ax1.set_title('Model Parameters Distribution', fontsize=14, fontweight='bold')
    
    # Plot 2: Parameter counts by layer type
    layer_params = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type not in layer_params:
            layer_params[layer_type] = 0
        layer_params[layer_type] += layer.count_params()
    
    layer_names = list(layer_params.keys())
    param_counts = list(layer_params.values())
    
    bars = ax2.bar(layer_names, param_counts, color=plt.cm.Set3(np.linspace(0, 1, len(layer_names))))
    ax2.set_title('Parameters by Layer Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Parameters')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(param_counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = 'gap_predictor_parameters.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Model parameters plot saved as {plot_path}")
    
    print(f"\nModel Complexity:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating gap predictor model architecture visualizations...")
    
    # Plot model architecture
    model = plot_model_architecture()
    
    # Plot model flow diagram
    plot_model_flow()
    
    # Plot model parameters
    plot_model_parameters()
    
    print("\nAll visualizations completed!")
