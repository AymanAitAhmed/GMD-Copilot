import plotly.graph_objects as go
import json

def create_grafana_plotly_chart():
    """
    Creates a simple Plotly bar chart and formats its data, layout,
    and configuration for use in a Grafana Plotly panel.

    Returns:
        dict: A dictionary containing 'data', 'layout', and 'config'
              fields, suitable for direct use in Grafana's Plotly panel.
    """

    # 1. Define the data for the bar chart
    categories = ['Apples', 'Oranges', 'Bananas', 'Grapes']
    values = [10, 15, 7, 12]

    # Create a bar trace
    bar_trace = go.Bar(
        x=categories,
        y=values,
        marker_color='skyblue', # Set a color for the bars
        name='Fruit Sales'
    )

    # 2. Define the layout for the chart
    chart_layout = go.Layout(
        title={
            'text': 'Monthly Fruit Sales',
            'font': {'size': 24, 'color': '#333'}
        },
        xaxis=dict(
            title='Fruit Type',
            tickangle=-45,
            automargin=True,
            showgrid=False # Hide x-axis grid lines
        ),
        yaxis=dict(
            title='Units Sold',
            rangemode='tozero', # Ensure y-axis starts from zero
            gridcolor='#e0e0e0' # Light grey grid lines for y-axis
        ),
        margin=dict(l=60, r=60, t=80, b=100), # Adjust margins for better spacing
        plot_bgcolor='white', # Set plot background to white
        paper_bgcolor='#f8f8f8', # Set paper background to light grey
        hovermode='closest', # Show hover info for the closest point
        font=dict(family="Arial, sans-serif", size=12, color="#444")
    )

    # 3. Define the configuration options (optional, but good for interactivity)
    # This controls things like modebar buttons, static plots, etc.
    chart_config = {
        'displayModeBar': True,  # Show the modebar (zoom, pan, etc.)
        'responsive': True,      # Make the chart responsive to container size
        'scrollZoom': True,      # Enable scroll to zoom
        'displaylogo': False     # Hide the Plotly logo
    }

    # Combine into the Grafana-compatible structure
    grafana_plotly_json = {
        'data': [bar_trace.to_plotly_json()], # Convert trace to Plotly JSON format
        'layout': chart_layout.to_plotly_json(), # Convert layout to Plotly JSON format
        'config': chart_config
    }

    # You can print this JSON to copy-paste into Grafana
    print("--- Copy this JSON into your Grafana Plotly panel ---")
    print(json.dumps(grafana_plotly_json, indent=2))
    print("-----------------------------------------------------")

    return grafana_plotly_json

# Call the function to generate and print the output
if __name__ == "__main__":
    create_grafana_plotly_chart()