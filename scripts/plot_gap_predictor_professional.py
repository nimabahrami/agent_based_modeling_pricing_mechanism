import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Suppress TensorFlow warnings and debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model.gap_predictor import GapPredictor

def create_interactive_lstm_architecture():
    """Create an interactive LSTM architecture visualization using Plotly."""
    print("Creating interactive LSTM architecture visualization...")
    
    # Define the LSTM architecture layers
    layers = [
        {'name': 'Input', 'type': 'Input', 'units': '(192, 2)', 'color': '#FF6B6B'},
        {'name': 'Bidirectional LSTM 1', 'type': 'Bidirectional LSTM', 'units': '128 units', 'color': '#4ECDC4'},
        {'name': 'Dropout 1', 'type': 'Dropout', 'units': '0.3', 'color': '#45B7D1'},
        {'name': 'Bidirectional LSTM 2', 'type': 'Bidirectional LSTM', 'units': '64 units', 'color': '#96CEB4'},
        {'name': 'Dropout 2', 'type': 'Dropout', 'units': '0.3', 'color': '#FFEAA7'},
        {'name': 'Bidirectional LSTM 3', 'type': 'Bidirectional LSTM', 'units': '32 units', 'color': '#DDA0DD'},
        {'name': 'Dropout 3', 'type': 'Dropout', 'units': '0.2', 'color': '#98D8C8'},
        {'name': 'Dense 1', 'type': 'Dense', 'units': '64 units (ReLU)', 'color': '#F7DC6F'},
        {'name': 'Dense 2', 'type': 'Dense', 'units': '192 units', 'color': '#BB8FCE'},
        {'name': 'Reshape', 'type': 'Reshape', 'units': '(96, 2)', 'color': '#85C1E9'},
        {'name': 'Output', 'type': 'Output', 'units': '(96, 2)', 'color': '#F8C471'}
    ]
    
    # Create interactive network diagram
    fig = go.Figure()
    
    # Node positions
    y_positions = np.linspace(0, 1, len(layers))
    
    # Add nodes
    for i, layer in enumerate(layers):
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[y_positions[i]],
            mode='markers+text',
            marker=dict(
                size=30,
                color=layer['color'],
                line=dict(width=2, color='black')
            ),
            text=layer['name'],
            textposition="middle center",
            textfont=dict(size=12, color='white', family='Arial Black'),
            name=layer['name'],
            hovertemplate=f"<b>{layer['name']}</b><br>" +
                         f"Type: {layer['type']}<br>" +
                         f"Units: {layer['units']}<br>" +
                         "<extra></extra>"
        ))
        
        # Add layer type labels
        fig.add_trace(go.Scatter(
            x=[0.3],
            y=[y_positions[i]],
            mode='text',
            text=layer['type'],
            textposition="middle right",
            textfont=dict(size=10, color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add units labels
        fig.add_trace(go.Scatter(
            x=[0.7],
            y=[y_positions[i]],
            mode='text',
            text=layer['units'],
            textposition="middle left",
            textfont=dict(size=10, color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add connections (arrows)
    for i in range(len(layers) - 1):
        fig.add_trace(go.Scatter(
            x=[0.5, 0.5],
            y=[y_positions[i] - 0.02, y_positions[i+1] + 0.02],
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Arrow head
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[y_positions[i+1] + 0.02],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='black'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Gap Predictor LSTM Architecture",
            x=0.5,
            font=dict(size=24, color='black')
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        plot_bgcolor='white',
        width=800,
        height=1000,
        showlegend=False
    )
    
    # Save as HTML for interactive viewing
    html_path = 'gap_predictor_interactive_architecture.html'
    fig.write_html(html_path)
    print(f"Interactive architecture saved as {html_path}")
    
    return fig

def create_lstm_cell_diagram():
    """Create a detailed LSTM cell diagram."""
    print("Creating detailed LSTM cell diagram...")
    
    # LSTM cell components
    components = [
        {'name': 'Input Gate', 'symbol': 'σ', 'color': '#FF6B6B'},
        {'name': 'Forget Gate', 'symbol': 'σ', 'color': '#4ECDC4'},
        {'name': 'Cell State', 'symbol': 'tanh', 'color': '#96CEB4'},
        {'name': 'Output Gate', 'symbol': 'σ', 'color': '#DDA0DD'},
        {'name': 'Hidden State', 'symbol': 'h', 'color': '#F7DC6F'}
    ]
    
    fig = go.Figure()
    
    # Create LSTM cell layout
    x_positions = np.linspace(0.1, 0.9, len(components))
    y_positions = [0.5] * len(components)
    
    # Add component nodes
    for i, component in enumerate(components):
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[y_positions[i]],
            mode='markers+text',
            marker=dict(
                size=40,
                color=component['color'],
                line=dict(width=3, color='black')
            ),
            text=component['symbol'],
            textposition="middle center",
            textfont=dict(size=16, color='white', family='Arial Black'),
            name=component['name'],
            hovertemplate=f"<b>{component['name']}</b><br>" +
                         f"Function: {component['symbol']}<br>" +
                         "<extra></extra>"
        ))
        
        # Add component labels
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[y_positions[i] - 0.15],
            mode='text',
            text=component['name'],
            textposition="middle center",
            textfont=dict(size=12, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add connections between components
    for i in range(len(components) - 1):
        fig.add_trace(go.Scatter(
            x=[x_positions[i] + 0.05, x_positions[i+1] - 0.05],
            y=[y_positions[i], y_positions[i+1]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add mathematical formulas
    formulas = [
        "i_t = σ(W_i · [h_{t-1}, x_t] + b_i)",
        "f_t = σ(W_f · [h_{t-1}, x_t] + b_f)",
        "C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)",
        "o_t = σ(W_o · [h_{t-1}, x_t] + b_o)",
        "h_t = o_t ⊙ tanh(C_t)"
    ]
    
    for i, formula in enumerate(formulas):
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[y_positions[i] + 0.15],
            mode='text',
            text=formula,
            textposition="middle center",
            textfont=dict(size=10, color='darkblue'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(
            text="LSTM Cell Architecture",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        plot_bgcolor='white',
        width=1000,
        height=600,
        showlegend=False
    )
    
    html_path = 'lstm_cell_diagram.html'
    fig.write_html(html_path)
    print(f"LSTM cell diagram saved as {html_path}")
    
    return fig

def create_3d_lstm_visualization():
    """Create a 3D visualization of the LSTM architecture."""
    print("Creating 3D LSTM visualization...")
    
    # Create 3D network structure
    fig = go.Figure()
    
    # Define layer positions in 3D space
    layers_3d = [
        {'name': 'Input', 'x': 0, 'y': 0, 'z': 0, 'color': '#FF6B6B'},
        {'name': 'LSTM 1', 'x': 1, 'y': 0, 'z': 0, 'color': '#4ECDC4'},
        {'name': 'LSTM 2', 'x': 2, 'y': 0, 'z': 0, 'color': '#96CEB4'},
        {'name': 'LSTM 3', 'x': 3, 'y': 0, 'z': 0, 'color': '#DDA0DD'},
        {'name': 'Dense 1', 'x': 4, 'y': 0, 'z': 0, 'color': '#F7DC6F'},
        {'name': 'Dense 2', 'x': 5, 'y': 0, 'z': 0, 'color': '#BB8FCE'},
        {'name': 'Output', 'x': 6, 'y': 0, 'z': 0, 'color': '#F8C471'}
    ]
    
    # Add 3D nodes
    for layer in layers_3d:
        fig.add_trace(go.Scatter3d(
            x=[layer['x']],
            y=[layer['y']],
            z=[layer['z']],
            mode='markers+text',
            marker=dict(
                size=15,
                color=layer['color'],
                line=dict(width=2, color='black')
            ),
            text=layer['name'],
            textposition="middle center",
            textfont=dict(size=12, color='white'),
            name=layer['name']
        ))
    
    # Add connections
    for i in range(len(layers_3d) - 1):
        fig.add_trace(go.Scatter3d(
            x=[layers_3d[i]['x'], layers_3d[i+1]['x']],
            y=[layers_3d[i]['y'], layers_3d[i+1]['y']],
            z=[layers_3d[i]['z'], layers_3d[i+1]['z']],
            mode='lines',
            line=dict(color='black', width=5),
            showlegend=False
        ))
    
    # Add dropout layers as smaller nodes
    dropout_positions = [
        {'x': 1.5, 'y': 0.5, 'z': 0, 'name': 'Dropout 0.3'},
        {'x': 2.5, 'y': 0.5, 'z': 0, 'name': 'Dropout 0.3'},
        {'x': 3.5, 'y': 0.5, 'z': 0, 'name': 'Dropout 0.2'}
    ]
    
    for dropout in dropout_positions:
        fig.add_trace(go.Scatter3d(
            x=[dropout['x']],
            y=[dropout['y']],
            z=[dropout['z']],
            mode='markers+text',
            marker=dict(
                size=8,
                color='orange',
                line=dict(width=1, color='black')
            ),
            text=dropout['name'],
            textposition="middle center",
            textfont=dict(size=8, color='black'),
            name=dropout['name']
        ))
    
    fig.update_layout(
        title=dict(
            text="3D LSTM Architecture Visualization",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='white'
        ),
        width=1000,
        height=800
    )
    
    html_path = 'lstm_3d_visualization.html'
    fig.write_html(html_path)
    print(f"3D LSTM visualization saved as {html_path}")
    
    return fig

def create_parameter_heatmap():
    """Create a parameter heatmap visualization."""
    print("Creating parameter heatmap...")
    
    # Get model parameters
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
    
    # Create heatmap data
    heatmap_data = np.array(layer_params).reshape(1, -1)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=layer_names,
        y=['Parameters'],
        colorscale='Viridis',
        text=[[f'{p:,}' for p in layer_params]],
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(
            text="Model Parameters Heatmap",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        xaxis=dict(title="Layers"),
        yaxis=dict(title=""),
        width=1000,
        height=400
    )
    
    html_path = 'parameter_heatmap.html'
    fig.write_html(html_path)
    print(f"Parameter heatmap saved as {html_path}")
    
    return fig

def create_comprehensive_dashboard():
    """Create a comprehensive dashboard with all visualizations."""
    print("Creating comprehensive dashboard...")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('LSTM Architecture Flow', 'Parameter Distribution', 
                       'Layer Types', 'Model Complexity'),
        specs=[[{"type": "scatter"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Get model data
    predictor = GapPredictor(sequence_length=192, output_length=96)
    if not predictor.is_trained:
        predictor.build_model()
    
    model = predictor.model
    
    # 1. Architecture flow
    layers = [layer.name for layer in model.layers]
    y_pos = np.linspace(0, 1, len(layers))
    
    fig.add_trace(
        go.Scatter(
            x=[0.5] * len(layers),
            y=y_pos,
            mode='markers+text',
            marker=dict(size=20, color='lightblue'),
            text=layers,
            textposition="middle center",
            name="Architecture"
        ),
        row=1, col=1
    )
    
    # 2. Parameter distribution
    layer_params = [layer.count_params() for layer in model.layers]
    layer_names = [layer.name for layer in model.layers]
    
    fig.add_trace(
        go.Pie(
            labels=layer_names,
            values=layer_params,
            name="Parameters"
        ),
        row=1, col=2
    )
    
    # 3. Layer types
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    unique_types = list(set(layer_types))
    type_counts = [layer_types.count(t) for t in unique_types]
    
    fig.add_trace(
        go.Bar(
            x=unique_types,
            y=type_counts,
            name="Layer Types"
        ),
        row=2, col=1
    )
    
    # 4. Model complexity indicator
    total_params = model.count_params()
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=total_params,
            title={'text': "Total Parameters"},
            gauge={'axis': {'range': [None, 400000]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 100000], 'color': "lightgray"},
                            {'range': [100000, 200000], 'color': "gray"},
                            {'range': [200000, 400000], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75,
                               'value': 300000}},
            name="Complexity"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(
            text="Gap Predictor LSTM Model Dashboard",
            x=0.5,
            font=dict(size=24, color='black')
        ),
        height=1000,
        showlegend=False
    )
    
    html_path = 'lstm_model_dashboard.html'
    fig.write_html(html_path)
    print(f"Comprehensive dashboard saved as {html_path}")
    
    return fig

if __name__ == "__main__":
    print("Creating professional LSTM architecture visualizations...")
    
    # Create all visualizations
    create_interactive_lstm_architecture()
    create_lstm_cell_diagram()
    create_3d_lstm_visualization()
    create_parameter_heatmap()
    create_comprehensive_dashboard()
    
    print("\nAll professional visualizations completed!")
    print("Check the generated HTML files for interactive visualizations!")
