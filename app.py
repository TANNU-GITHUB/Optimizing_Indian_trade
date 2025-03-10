import os
import pandas as pd
import networkx as nx
from flask import Flask, render_template, request
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
import numpy as np

app = Flask(__name__)

# Load the dataset
try:
    df = pd.read_csv('Trade_network_data.csv')
    print("DataFrame loaded successfully. Head:", df.head())
except FileNotFoundError:
    print("Error: 'Trade_network_data.csv' not found in", os.getcwd())
    df = pd.DataFrame()

def correlation_analysis(data):
    """Scatter plot for Impact of Distance on Trade Volume."""
    if 'Distance (km)' not in data.columns or 'Trade Volume (tons)' not in data.columns:
        return None, None, None, ""
    corr, p_value = pearsonr(data['Distance (km)'], data['Trade Volume (tons)'])
    fig = px.scatter(
        data, x="Distance (km)", y="Trade Volume (tons)",
        color="Mode_of_Transport", size="Transportation Cost (USD)",
        hover_data=["Origin", "Destination", "Toll Charges (USD)"],
        title="Impact of Distance on Trade Volume",
        color_continuous_scale="Plasma", opacity=0.8
    )
    fig.update_traces(marker=dict(line=dict(width=2, color='#FFD700')))
    fig.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a", font_color="#FFD700",
        font=dict(family="Arial", size=16),
        title=dict(x=0.5, font=dict(size=24, color="#39ff14")),
        xaxis_title="Distance (km)", yaxis_title="Trade Volume (tons)",
        xaxis=dict(gridcolor="#444", zerolinecolor="#444"),
        yaxis=dict(gridcolor="#444", zerolinecolor="#444"),
        legend_title="Transport Mode", margin=dict(l=50, r=50, t=100, b=50),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=14, font_color="#FFD700")
    )
    explanation = (
        "<b>Goal:</b> To understand whether proximity influences trade.<br>"
        "<b>Solution:</b> Correlation Analysis<br>"
        f"This scatter plot shows the relationship between distance and trade volume. "
        f"Bubble size reflects transportation cost, and color indicates transport mode. "
        f"The correlation coefficient ({corr:.3f}) suggests {'a strong' if abs(corr) > 0.5 else 'a weak'} "
        f"relationship, with a p-value of {p_value:.5f} indicating "
        f"{'statistical significance' if p_value < 0.05 else 'no significant relationship'}."
    )
    return corr, p_value, fig.to_html(full_html=False, include_plotlyjs='cdn'), explanation

def average_cost_per_km(data):
    """Bar chart for Average Trade Cost Per Unit Distance."""
    if 'Transportation Cost (USD)' not in data.columns or 'Distance (km)' not in data.columns:
        return None, None
    data = data.copy()
    data['Cost_Per_Km'] = data['Transportation Cost (USD)'] / data['Distance (km)'].replace(0, 1)
    avg_cost = data.groupby('Mode_of_Transport')['Cost_Per_Km'].agg(['mean', 'std']).reset_index()
    fig = go.Figure()
    colors = ['#FF6347', '#32CD32', '#1E90FF']  # Tomato Red, Lime Green, Dodger Blue
    error_colors = ['#FF6347', '#32CD32', '#1E90FF']  # Match error bars to bar colors
    for i, mode in enumerate(avg_cost['Mode_of_Transport']):
        subset = avg_cost[avg_cost['Mode_of_Transport'] == mode]
        fig.add_trace(go.Bar(
            x=[mode], y=[subset['mean'].values[0]], name=mode,
            marker_color=colors[i % len(colors)],
            error_y=dict(type='data', array=[subset['std'].values[0]], visible=True, color=error_colors[i % len(error_colors)]),
            width=0.6, opacity=0.9
        ))
    fig.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a", font_color="#FFD700",
        font=dict(family="Arial", size=16),
        title=dict(text="Average Trade Cost Per Unit Distance", x=0.5, font=dict(size=24, color="#39ff14")),
        xaxis_title="Mode of Transport", yaxis_title="Cost Per Km (USD/km)",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#444"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="#FFD700", borderwidth=1),
        margin=dict(l=50, r=50, t=100, b=50),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=14, font_color="#FFD700")
    )
    explanation = (
        "<b>Goal:</b> To find efficiency of trade per km.<br>"
        "<b>Solution:</b> Simple Graph Weight Analysis<br>"
        "This bar chart displays the average cost per kilometer for each transport mode. "
        "Error bars show variability (standard deviation), helping identify the most cost-efficient options."
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn'), explanation

def optimize_trade_routes(data):
    """Network graph and data for Optimizing Trade Routes for Maximum Profit."""
    if not all(col in data.columns for col in ['Origin', 'Destination', 'Trade Volume (tons)', 'Origin_Latitude', 'Origin_Longitude']):
        return None, None, None, pd.DataFrame()
    
    # Create the graph
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['Origin'], row['Destination'], weight=row['Trade Volume (tons)'])
    
    # Compute the maximum spanning tree (MST) for optimized routes
    mst = nx.maximum_spanning_tree(G, algorithm="prim")
    edges = list(mst.edges(data=True))
    optimized_routes = pd.DataFrame(edges, columns=['Origin', 'Destination', 'Trade_Volume'])

    # Get unique countries and their coordinates
    countries = list(G.nodes())
    coords = {}
    for country in countries:
        # Find the first occurrence of the country in Origin to get its coordinates
        country_data = data[data['Origin'] == country]
        if not country_data.empty:
            coords[country] = (country_data.iloc[0]['Origin_Latitude'], country_data.iloc[0]['Origin_Longitude'])
        else:
            # Fallback if the country is only in Destination
            country_data = data[data['Destination'] == country]
            if not country_data.empty:
                coords[country] = (country_data.iloc[0]['Destination_Latitude'], country_data.iloc[0]['Destination_Longitude'])
            else:
                # Default to (0,0) if coordinates are missing
                coords[country] = (0, 0)

    # Debug: Print raw coordinates for India and Sri Lanka
    print("Raw coordinates:")
    for country in ['India', 'Sri Lanka']:
        if country in coords:
            print(f"{country}: Latitude = {coords[country][0]}, Longitude = {coords[country][1]}")

    # Scale coordinates to fit within the Plotly graph
    lat = [coords[country][0] for country in countries]
    lon = [coords[country][1] for country in countries]
    
    # Normalize latitude and longitude to fit within a reasonable range
    lat_range = max(lat) - min(lat) if max(lat) != min(lat) else 1
    lon_range = max(lon) - min(lon) if max(lon) != min(lon) else 1
    lat_scaled = [(l - min(lat)) / lat_range * 2 - 1 for l in lat]  # Scale to [-1, 1]
    lon_scaled = [(l - min(lon)) / lon_range * 2 - 1 for l in lon]  # Scale to [-1, 1]
    
    # Adjust scaling to fit better in the viewport
    lat_scaled = [l * 1.5 for l in lat_scaled]  # Increase latitude spread
    lon_scaled = [l * 2.0 for l in lon_scaled]  # Increase longitude spread
    
    pos = {country: (lon_scaled[i], lat_scaled[i]) for i, country in enumerate(countries)}

    # Debug: Print scaled positions for India and Sri Lanka
    print("Scaled positions:")
    for country in ['India', 'Sri Lanka']:
        if country in pos:
            print(f"{country}: x = {pos[country][0]}, y = {pos[country][1]}")

    # Prepare edges for plotting
    edge_x, edge_y, edge_text = [], [], []
    for edge in mst.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[0]} - {edge[1]}: {mst[edge[0]][edge[1]]['weight']} tons")

    # Prepare nodes
    node_x, node_y = zip(*pos.values())
    node_labels = list(G.nodes())

    # Dynamic label offsetting based on node degree and position
    label_positions = {}
    for node in G.nodes():
        x, y = pos[node]
        degree = G.degree(node)
        # Offset labels based on degree
        if degree > 5:  # High-degree nodes (e.g., India)
            offset = 0.2
        elif degree > 2:
            offset = 0.15
        else:
            offset = 0.1
        
        # Adjust offset direction based on latitude (y)
        if y > 0:  # Northern hemisphere
            if node == 'Sri Lanka':
                label_positions[node] = (x, y - offset - 0.2)  # Force Sri Lanka's label below
            else:
                label_positions[node] = (x, y + offset)  # Label above
        else:  # Southern hemisphere
            label_positions[node] = (x, y - offset)  # Label below

        # Additional adjustments for crowded areas
        if node == 'India':
            label_positions[node] = (x + 0.1, y + 0.2)
        elif node in ['Nepal', 'Bhutan', 'Bangladesh']:
            if x > pos['India'][0]:  # East of India
                label_positions[node] = (x + offset, y)
            else:  # West of India
                label_positions[node] = (x - offset, y)

    label_x, label_y = zip(*label_positions.values())

    # Create the figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line=dict(color="#FFD700", width=2),
        hoverinfo="text", text=edge_text, opacity=0.8
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=25, color="#DAA520", line=dict(width=2, color="#FFD700")),
        hoverinfo="text", text=node_labels, opacity=0.9
    ))
    
    # Add labels with dynamic offset and new color
    fig.add_trace(go.Scatter(
        x=label_x, y=label_y, mode="text", text=node_labels,
        textposition="middle center", hoverinfo="none",
        textfont=dict(size=12, color="#FF69B4", family="Arial")  # Changed to Hot Pink
    ))
    
    # Update layout with zoom and pan support
    fig.update_layout(
        title=dict(text="Optimizing Trade Routes for Maximum Profit", x=0.5, font=dict(size=24, color="#39ff14")),
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a", font_color="#FFD700",
        font=dict(family="Arial", size=16),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=100, b=20),
        height=800,
        width=1200,
        dragmode="zoom",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=14, font_color="#FFD700"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{"xaxis.range": None, "yaxis.range": None}], label="Reset Zoom", method="relayout")
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01, xanchor="left", y=1.1, yanchor="top",
                bgcolor="#FFD700", font=dict(color="#1a1a1a")
            )
        ]
    )
    
    explanation = (
        "<b>Goal:</b> To identify the most profitable trade paths.<br>"
        "<b>Solution:</b> Greedy Approach (Prim’s Algorithm for Maximum Spanning Tree)<br>"
        "This network graph highlights the most profitable trade routes using Prim's algorithm. "
        "Nodes are positioned based on geographical coordinates, with labels adjusted for clarity. "
        "Zoom and pan to explore the graph in detail."
    )
    
    return optimized_routes.to_dict('records'), fig.to_html(full_html=False, include_plotlyjs='cdn'), explanation, optimized_routes

def tariff_impact(data, optimized_routes, selected_route=None):
    """Impressive world map showing optimized routes with tariff impact."""
    if not all(col in data.columns for col in ['Origin_Latitude', 'Origin_Longitude', 'Destination_Latitude', 'Destination_Longitude', 'Trade Volume (tons)', 'Toll Charges (USD)']):
        return None, None, None

    # Filter to optimized routes
    optimized_data = data.merge(
        optimized_routes[['Origin', 'Destination']],
        on=['Origin', 'Destination'],
        how='inner'
    )

    # Aggregate trade statistics for countries
    country_stats = optimized_data.groupby('Destination').agg({
        'Trade Volume (tons)': 'sum',
        'Toll Charges (USD)': 'sum',
        'Transportation Cost (USD)': 'sum',
        'Destination': 'count'
    }).rename(columns={'Destination': 'Route_Count'}).reset_index()
    country_stats['Total_Cost'] = country_stats['Transportation Cost (USD)'] + country_stats['Toll Charges (USD)']

    fig = go.Figure()

    # Base map with starry night aesthetic
    fig.update_geos(
        bgcolor="rgba(10, 10, 20, 0.9)",
        landcolor="#1F2A44",
        showcountries=True, countrycolor="#FFD700",
        showocean=True, oceancolor="#0A1A2F",
        projection_type="natural earth",
        resolution=110
    )

    # Add optimized trade routes with tariff impact and distinct colors
    for _, row in optimized_data.iterrows():
        is_selected = selected_route == f"{row['Origin']}-{row['Destination']}"
        width = min(1 + row['Trade Volume (tons)'] / optimized_data['Trade Volume (tons)'].max() * 6, 8) if not is_selected else 10
        toll_normalized = row['Toll Charges (USD)'] / optimized_data['Toll Charges (USD)'].max()
        color = '#00FF00' if toll_normalized < 0.33 else '#FFA500' if toll_normalized < 0.66 else '#FF4500'  # Lime Green, Orange, Red-Orange
        color = '#39FF14' if is_selected else color  # Neon Green for selected
        opacity = 0.7 if not is_selected else 1.0
        fig.add_trace(go.Scattergeo(
            lon=[row['Origin_Longitude'], row['Destination_Longitude']],
            lat=[row['Origin_Latitude'], row['Destination_Latitude']],
            mode='lines',
            line=dict(width=width, color=color),
            opacity=opacity,
            hoverinfo='text',
            text=f"<b>Route:</b> {row['Origin']} to {row['Destination']}<br>"
                 f"<b>Volume:</b> {row['Trade Volume (tons)']} tons<br>"
                 f"<b>Cost:</b> {row['Transportation Cost (USD)']} USD<br>"
                 f"<b>Tariff:</b> {row['Toll Charges (USD)']} USD",
            hovertemplate="%{text}<extra></extra>",
            customdata=[is_selected]
        ))

    # Add pulsating markers for destinations with hover information
    for _, row in country_stats.iterrows():
        country = row['Destination']
        # Find corresponding coordinates (using Destination_Latitude/Longitude from optimized_data)
        coord_data = optimized_data[optimized_data['Destination'] == country].iloc[0]
        fig.add_trace(go.Scattergeo(
            lon=[coord_data['Destination_Longitude']],
            lat=[coord_data['Destination_Latitude']],
            mode='markers+text',
            marker=dict(
                size=20, color="#FFD700",
                line=dict(width=3, color="#FFC107"),
                symbol="circle", opacity=0.9
            ),
            text=[country],
            hoverinfo='text',
            hovertext=f"<b>Country:</b> {country}<br>"
                     f"<b>Total Trade Volume:</b> {row['Trade Volume (tons)']} tons<br>"
                     f"<b>Total Cost:</b> {row['Total_Cost']} USD<br>"
                     f"<b>Number of Routes:</b> {row['Route_Count']}",
            textposition="top center",
            textfont=dict(size=16, color="#FFD700", family="Arial")
        ))

    # Highlight selected route with neon green
    if selected_route:
        route_parts = selected_route.split('-')
        if len(route_parts) == 2:
            origin, destination = route_parts
            selected_data = optimized_data[(optimized_data['Origin'] == origin) & (optimized_data['Destination'] == destination)]
            if not selected_data.empty:
                row = selected_data.iloc[0]
                fig.add_trace(go.Scattergeo(
                    lon=[row['Origin_Longitude'], row['Destination_Longitude']],
                    lat=[row['Origin_Latitude'], row['Destination_Latitude']],
                    mode='lines',
                    line=dict(width=10, color='#39FF14'),
                    opacity=1.0,
                    hoverinfo='text',
                    text=f"<b>Selected Route:</b> {row['Origin']} to {row['Destination']}<br>"
                         f"<b>Volume:</b> {row['Trade Volume (tons)']} tons<br>"
                         f"<b>Cost:</b> {row['Transportation Cost (USD)']} USD<br>"
                         f"<b>Tariff:</b> {row['Toll Charges (USD)']} USD",
                    hovertemplate="%{text}<extra></extra>",
                    customdata=[True]
                ))

    fig.update_layout(
        title=dict(text="Effect of Tariffs on Optimized Trade Routes", x=0.5, font=dict(size=28, color="#39ff14", family="Arial Black")),
        plot_bgcolor="rgba(10, 10, 20, 0.9)", paper_bgcolor="rgba(10, 10, 20, 0.9)",
        font_color="#FFD700", font=dict(family="Arial", size=16),
        geo=dict(scope='world', projection_type="natural earth"),
        margin=dict(l=20, r=20, t=120, b=20), height=900,
        showlegend=False,
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.9)", font_size=16, font_color="#FFD700", bordercolor="#FFD700", align="left", namelength=-1),
        dragmode="zoom",
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=[dict(args=[{"visible": [True] * len(fig.data)}], label="Reset Zoom", method="update")],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.01, xanchor="left", y=1.1, yanchor="top",
            bgcolor="#FFD700", font=dict(color="#1a1a1a")
        )],
        annotations=[
            dict(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text="<b>Legend</b><br>"
                     "Line Thickness: Trade Volume<br>"
                     "Colors:<br>"
                     "  <span style='color:#00FF00'>Lime Green</span>: Low Tariffs<br>"
                     "  <span style='color:#FFA500'>Orange</span>: Medium Tariffs<br>"
                     "  <span style='color:#FF4500'>Red-Orange</span>: High Tariffs<br>"
                     "  <span style='color:#39FF14'>Neon Green</span>: Selected Route",
                showarrow=False, align="left",
                bgcolor="rgba(0,0,0,0.7)", bordercolor="#FFD700", borderwidth=2,
                font=dict(size=14, color="#FFD700")
            )
        ]
    )

    explanation = (
        "<b>Goal:</b> To analyze whether higher tariffs reduce trade between countries.<br>"
        "<b>Solution:</b> An awe-inspiring map with optimized routes and tariff impact<br>"
        "This luxurious map showcases optimized trade routes, where thickness reflects trade volume "
        "and colors indicate tariff levels. Hover over routes or countries to see detailed information, "
        "select a route to highlight it in neon green, and zoom into the cosmic backdrop for an immersive experience."
    )
    routes = [f"{row['Origin']}-{row['Destination']}" for _, row in optimized_data.iterrows()]
    return fig.to_html(full_html=False, include_plotlyjs='cdn'), explanation, routes

def find_shortest_path(origin, destination, criterion='cost'):
    if df.empty:
        return [], 0
    G = nx.DiGraph()
    for _, row in df.iterrows():
        if criterion == 'cost':
            weight = row['Transportation Cost (USD)'] + row['Toll Charges (USD)']
        elif criterion == 'time':
            weight = row['Estimated Travel Time (hours)']
        else:
            weight = row['Distance (km)']
        G.add_edge(row['Origin'], row['Destination'], weight=weight)
    try:
        path = nx.shortest_path(G, origin, destination, weight='weight')
        value = nx.shortest_path_length(G, origin, destination, weight='weight')
        return path, value
    except nx.NetworkXNoPath:
        return [], 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data-analysis', methods=['GET', 'POST'])
def data_analysis():
    if df.empty:
        stats = {'avg_cost': 0.0, 'avg_distance': 0.0, 'top_origins': {}, 'modes': {}}
        return render_template('data_analysis.html', stats=stats, filtered_routes=[], path=[], value=0,
                               corr=None, p_value=None, fig1=None, fig2=None, optimized_routes=[], fig3=None,
                               exp1="", exp2="", exp3="", unique_origins=[], unique_destinations=[])

    # Debug: Check dataset distribution
    print("Route Distribution by Origin:")
    print(df['Origin'].value_counts())
    print("Route Distribution by Destination:")
    print(df['Destination'].value_counts())
    print("Specific Route Check (India to Italy):")
    print(df[(df['Origin'] == 'India') & (df['Destination'] == 'Italy')])

    stats = {
        'avg_cost': df['Transportation Cost (USD)'].mean(),
        'avg_distance': df['Distance (km)'].mean(),
        'top_origins': df['Origin'].value_counts().head(3).to_dict(),
        'modes': df['Mode_of_Transport'].value_counts().to_dict()
    }

    data_clean = df.dropna(subset=['Distance (km)', 'Trade Volume (tons)', 'Transportation Cost (USD)', 
                                   'Mode_of_Transport', 'Toll Charges (USD)'])
    corr, p_value, fig1, exp1 = correlation_analysis(data_clean)
    fig2, exp2 = average_cost_per_km(data_clean)
    optimized_routes, fig3, exp3, optimized_df = optimize_trade_routes(data_clean)

    # Prepare unique origins and destinations for dropdowns
    unique_origins = sorted(df['Origin'].unique())
    unique_destinations = sorted(df['Destination'].unique())

    if request.method == 'POST':
        origin = request.form.get('origin', '').strip()
        destination = request.form.get('destination', '').strip()
        criterion = request.form.get('criterion', 'cost')
        # Updated filtering logic with case-insensitive matching and deduplication
        filtered_routes = df
        if origin and destination:
            filtered_routes = df[
                (df['Origin'].str.lower() == origin.lower()) & 
                (df['Destination'].str.lower() == destination.lower())
            ]
        elif origin:
            filtered_routes = df[df['Origin'].str.lower() == origin.lower()]
        elif destination:
            filtered_routes = df[df['Destination'].str.lower() == destination.lower()]
        
        # Remove duplicates based on Origin, Destination, and other key columns
        filtered_routes = filtered_routes.drop_duplicates(
            subset=['Origin', 'Destination', 'Distance (km)', 'Transportation Cost (USD)', 'Trade Volume (tons)']
        )
        
        print(f"Filtered routes for origin={origin}, destination={destination}:")
        print(filtered_routes)
        
        path, value = find_shortest_path('India', destination, criterion) if origin == 'India' and destination else ([], 0)
        return render_template('data_analysis.html', stats=stats, filtered_routes=filtered_routes.to_dict('records'),
                               path=path, value=value, criterion=criterion, corr=corr, p_value=p_value, fig1=fig1,
                               fig2=fig2, optimized_routes=optimized_routes, fig3=fig3, exp1=exp1, exp2=exp2, exp3=exp3,
                               unique_origins=unique_origins, unique_destinations=unique_destinations)

    return render_template('data_analysis.html', stats=stats, filtered_routes=df.to_dict('records'), path=[], value=0,
                           corr=corr, p_value=p_value, fig1=fig1, fig2=fig2, optimized_routes=optimized_routes, fig3=fig3,
                           exp1=exp1, exp2=exp2, exp3=exp3, unique_origins=unique_origins, unique_destinations=unique_destinations)

@app.route('/world-map', methods=['GET', 'POST'])
def world_map():
    if df.empty:
        return render_template('world_map.html', fig4=None, exp4="", routes=[])

    data_clean = df.dropna(subset=['Origin_Latitude', 'Origin_Longitude', 'Destination_Latitude', 'Destination_Longitude', 
                                   'Trade Volume (tons)', 'Transportation Cost (USD)', 'Toll Charges (USD)'])
    _, _, _, optimized_df = optimize_trade_routes(data_clean)
    selected_route = request.form.get('route') if request.method == 'POST' else None
    fig4, exp4, routes = tariff_impact(data_clean, optimized_df, selected_route)

    return render_template('world_map.html', fig4=fig4, exp4=exp4, routes=routes, selected_route=selected_route)

@app.route('/algo-rendering')
def algo_rendering():
    return render_template('algo_rendering.html')

if __name__ == '__main__':
    app.run(debug=True)
