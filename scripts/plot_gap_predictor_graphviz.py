import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings and debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.gap_predictor import GapPredictor

def create_graphviz_lstm_diagram():
    """Create a professional LSTM diagram using graphviz."""
    try:
        import graphviz
    except ImportError:
        print("Graphviz not available. Installing...")
        os.system("pip3 install graphviz")
        try:
            import graphviz
        except ImportError:
            print("Could not install graphviz. Creating alternative visualization...")
            return create_alternative_diagram()
    
    print("Creating Graphviz LSTM diagram...")
    
    # Create the graph
    dot = graphviz.Digraph(comment='Gap Predictor LSTM Architecture')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    # Set graph attributes
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='12')
    
    # Define colors for different layer types
    colors = {
        'input': '#FF6B6B',
        'lstm': '#4ECDC4', 
        'dropout': '#FFEAA7',
        'dense': '#96CEB4',
        'reshape': '#DDA0DD',
        'output': '#F8C471'
    }
    
    # Add nodes
    nodes = [
        ('input', 'Input\n(192, 2)', 'input'),
        ('lstm1', 'Bidirectional LSTM\n128 units', 'lstm'),
        ('dropout1', 'Dropout\n0.3', 'dropout'),
        ('lstm2', 'Bidirectional LSTM\n64 units', 'lstm'),
        ('dropout2', 'Dropout\n0.3', 'dropout'),
        ('lstm3', 'Bidirectional LSTM\n32 units', 'lstm'),
        ('dropout3', 'Dropout\n0.2', 'dropout'),
        ('dense1', 'Dense\n64 units (ReLU)', 'dense'),
        ('dense2', 'Dense\n192 units', 'dense'),
        ('reshape', 'Reshape\n(96, 2)', 'reshape'),
        ('output', 'Output\n(96, 2)', 'output')
    ]
    
    # Add nodes to graph
    for node_id, label, node_type in nodes:
        dot.node(node_id, label, fillcolor=colors[node_type], 
                fontcolor='white' if node_type != 'dropout' else 'black')
    
    # Add edges
    edges = [
        ('input', 'lstm1'),
        ('lstm1', 'dropout1'),
        ('dropout1', 'lstm2'),
        ('lstm2', 'dropout2'),
        ('dropout2', 'lstm3'),
        ('lstm3', 'dropout3'),
        ('dropout3', 'dense1'),
        ('dense1', 'dense2'),
        ('dense2', 'reshape'),
        ('reshape', 'output')
    ]
    
    for edge in edges:
        dot.edge(edge[0], edge[1])
    
    # Save the diagram
    dot.render('gap_predictor_lstm_diagram', format='png', cleanup=True)
    print("Graphviz LSTM diagram saved as gap_predictor_lstm_diagram.png")
    
    return dot

def create_alternative_diagram():
    """Create an alternative diagram using matplotlib."""
    print("Creating alternative LSTM diagram using matplotlib...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define layers
    layers = [
        {'name': 'Input', 'pos': (1, 10), 'color': '#FF6B6B', 'size': 0.8},
        {'name': 'Bidirectional\nLSTM 128', 'pos': (3, 10), 'color': '#4ECDC4', 'size': 1.2},
        {'name': 'Dropout 0.3', 'pos': (5, 10), 'color': '#FFEAA7', 'size': 0.6},
        {'name': 'Bidirectional\nLSTM 64', 'pos': (7, 10), 'color': '#4ECDC4', 'size': 1.0},
        {'name': 'Dropout 0.3', 'pos': (9, 10), 'color': '#FFEAA7', 'size': 0.6},
        {'name': 'Bidirectional\nLSTM 32', 'pos': (2, 7), 'color': '#4ECDC4', 'size': 0.8},
        {'name': 'Dropout 0.2', 'pos': (4, 7), 'color': '#FFEAA7', 'size': 0.6},
        {'name': 'Dense 64\n(ReLU)', 'pos': (6, 7), 'color': '#96CEB4', 'size': 0.8},
        {'name': 'Dense 192', 'pos': (8, 7), 'color': '#96CEB4', 'size': 0.8},
        {'name': 'Reshape\n(96, 2)', 'pos': (5, 4), 'color': '#DDA0DD', 'size': 0.8},
        {'name': 'Output\n(96, 2)', 'pos': (5, 1), 'color': '#F8C471', 'size': 0.8}
    ]
    
    # Draw layers
    for layer in layers:
        x, y = layer['pos']
        size = layer['size']
        
        # Draw rectangle
        rect = plt.Rectangle((x - size/2, y - size/2), size, size, 
                           facecolor=layer['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y, layer['name'], ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white' if layer['color'] != '#FFEAA7' else 'black')
    
    # Add connections
    connections = [
        ((1, 10), (3, 10)),
        ((3, 10), (5, 10)),
        ((5, 10), (7, 10)),
        ((7, 10), (9, 10)),
        ((9, 10), (2, 7)),
        ((2, 7), (4, 7)),
        ((4, 7), (6, 7)),
        ((6, 7), (8, 7)),
        ((8, 7), (5, 4)),
        ((5, 4), (5, 1))
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add title
    ax.text(5, 11.5, 'Gap Predictor LSTM Architecture', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FF6B6B', label='Input/Output'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#4ECDC4', label='LSTM Layers'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFEAA7', label='Dropout'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#96CEB4', label='Dense Layers'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#DDA0DD', label='Reshape')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.9))
    
    plt.tight_layout()
    plt.savefig('gap_predictor_alternative_diagram.png', dpi=300, bbox_inches='tight')
    print("Alternative diagram saved as gap_predictor_alternative_diagram.png")
    plt.show()

def create_lstm_cell_detailed():
    """Create a detailed LSTM cell visualization."""
    print("Creating detailed LSTM cell visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # LSTM cell components
    components = [
        {'name': 'Input Gate', 'pos': (2, 6), 'color': '#FF6B6B', 'formula': 'i_t = σ(W_i · [h_{t-1}, x_t] + b_i)'},
        {'name': 'Forget Gate', 'pos': (4, 6), 'color': '#4ECDC4', 'formula': 'f_t = σ(W_f · [h_{t-1}, x_t] + b_f)'},
        {'name': 'Cell State', 'pos': (6, 6), 'color': '#96CEB4', 'formula': 'C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)'},
        {'name': 'Output Gate', 'pos': (8, 6), 'color': '#DDA0DD', 'formula': 'o_t = σ(W_o · [h_{t-1}, x_t] + b_o)'},
        {'name': 'Hidden State', 'pos': (10, 6), 'color': '#F7DC6F', 'formula': 'h_t = o_t ⊙ tanh(C_t)'}
    ]
    
    # Draw components
    for comp in components:
        x, y = comp['pos']
        
        # Draw circle
        circle = plt.Circle((x, y), 0.8, facecolor=comp['color'], edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Add component name
        ax.text(x, y + 0.3, comp['name'], ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Add formula
        ax.text(x, y - 0.3, comp['formula'], ha='center', va='center', 
               fontsize=8, color='darkblue')
    
    # Add connections
    for i in range(len(components) - 1):
        x1, y1 = components[i]['pos']
        x2, y2 = components[i+1]['pos']
        ax.annotate('', xy=(x2-0.8, y2), xytext=(x1+0.8, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add cell state flow
    ax.plot([6, 6], [5.2, 2], '--', color='red', linewidth=3, alpha=0.7)
    ax.text(6.5, 3.5, 'Cell State\nFlow', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='red')
    
    # Add title
    ax.text(6, 7.5, 'LSTM Cell Architecture', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add description
    description = """
    LSTM (Long Short-Term Memory) cells contain gates that control information flow:
    • Input Gate: Controls what new information to store
    • Forget Gate: Controls what information to discard
    • Cell State: The memory line that runs through the cell
    • Output Gate: Controls what information to output
    • Hidden State: The output of the cell
    """
    
    ax.text(0.5, 1, description, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('lstm_cell_detailed.png', dpi=300, bbox_inches='tight')
    print("Detailed LSTM cell visualization saved as lstm_cell_detailed.png")
    plt.show()

if __name__ == "__main__":
    print("Creating professional LSTM visualizations with Graphviz...")
    
    # Try to create Graphviz diagram
    try:
        create_graphviz_lstm_diagram()
    except Exception as e:
        print(f"Graphviz failed: {e}")
        create_alternative_diagram()
    
    # Create detailed LSTM cell visualization
    create_lstm_cell_detailed()
    
    print("\nAll professional visualizations completed!")
