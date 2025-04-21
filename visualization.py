import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_profit_distribution(df, profit_column):
    """
    Creates a histogram to visualize the distribution of profits.
    
    Args:
        df (pd.DataFrame): The analyzed trading data
        profit_column (str): The name of the column containing profit/return data
        
    Returns:
        plotly.graph_objects.Figure: A plotly figure with the profit distribution
    """
    # Create histogram with custom bins
    fig = px.histogram(
        df, 
        x=profit_column,
        title="Distribution of Trade Returns",
        labels={profit_column: "Return (%)"},
        color_discrete_sequence=['#636EFA'],
        nbins=50
    )
    
    # Add vertical line at 0 to mark profit/loss boundary
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
    
    # Add vertical line at 100% to mark 2x return
    fig.add_vline(x=100, line_width=2, line_dash="dash", line_color="green")
    
    # Add annotations
    fig.add_annotation(
        x=0, 
        y=1, 
        yref="paper",
        text="Break Even",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    fig.add_annotation(
        x=100, 
        y=1, 
        yref="paper",
        text="2X Return",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Return (%)",
        yaxis_title="Number of Trades",
        bargap=0.1
    )
    
    return fig

def plot_filter_effectiveness(df, filter_name, filter_value, profit_column):
    """
    Creates a box plot to compare returns with and without a specific filter.
    
    Args:
        df (pd.DataFrame): The analyzed trading data
        filter_name (str): The name of the filter column
        filter_value (str): The value of the filter
        profit_column (str): The name of the column containing profit/return data
        
    Returns:
        plotly.graph_objects.Figure: A plotly figure with the comparison
    """
    # Create a new column indicating if the filter is applied
    df['filter_applied'] = df[filter_name] == filter_value
    
    # Create box plot
    fig = px.box(
        df, 
        x='filter_applied', 
        y=profit_column,
        color='filter_applied',
        title=f"Return Distribution: {filter_name} = {filter_value}",
        labels={
            'filter_applied': 'Filter Applied',
            profit_column: 'Return (%)'
        },
        category_orders={'filter_applied': [True, False]},
        color_discrete_map={True: 'green', False: 'red'}
    )
    
    # Add horizontal line at 0 to mark profit/loss boundary
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    
    # Add horizontal line at 100% to mark 2x return
    fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="green")
    
    # Compute and display summary statistics
    with_filter = df[df['filter_applied']].copy()
    without_filter = df[~df['filter_applied']].copy()
    
    with_avg = with_filter[profit_column].mean()
    without_avg = without_filter[profit_column].mean()
    
    # Add annotations for means
    fig.add_annotation(
        x=0, 
        y=with_avg,
        text=f"Mean: {with_avg:.2f}%",
        showarrow=True,
        arrowhead=1,
        ax=30,
        ay=0
    )
    
    fig.add_annotation(
        x=1, 
        y=without_avg,
        text=f"Mean: {without_avg:.2f}%",
        showarrow=True,
        arrowhead=1,
        ax=-30,
        ay=0
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Return (%)",
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=[f'With Filter\n({len(with_filter)} trades)', f'Without Filter\n({len(without_filter)} trades)']
        )
    )
    
    return fig

def plot_win_rate_by_filter(df, filter_name, profit_column):
    """
    Creates a bar chart showing win rates for different values of a filter.
    
    Args:
        df (pd.DataFrame): The analyzed trading data
        filter_name (str): The name of the filter column
        profit_column (str): The name of the column containing profit/return data
        
    Returns:
        plotly.graph_objects.Figure: A plotly figure with the win rates
    """
    # Group by the filter and calculate win rate
    filter_stats = df.groupby(filter_name).agg(
        win_rate=('win', lambda x: 100 * x.mean()),
        avg_return=(profit_column, 'mean'),
        trade_count=(profit_column, 'count')
    ).reset_index()
    
    # Filter to include only values with enough trades
    min_trades = max(50, len(df) * 0.05)  # At least 50 trades or 5% of total
    filter_stats = filter_stats[filter_stats['trade_count'] >= min_trades]
    
    # Sort by win rate
    filter_stats = filter_stats.sort_values('win_rate', ascending=False)
    
    # Create the figure
    fig = px.bar(
        filter_stats,
        x=filter_name,
        y='win_rate',
        title=f"Win Rate by {filter_name}",
        labels={'win_rate': 'Win Rate (%)'},
        color='avg_return',
        color_continuous_scale='RdYlGn',
        hover_data=['trade_count', 'avg_return']
    )
    
    # Add horizontal line at overall win rate
    overall_win_rate = 100 * df['win'].mean()
    fig.add_hline(
        y=overall_win_rate, 
        line_width=2, 
        line_dash="dash", 
        line_color="black",
        annotation_text=f"Overall Win Rate: {overall_win_rate:.2f}%",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=filter_name,
        yaxis_title="Win Rate (%)",
        coloraxis_colorbar_title="Avg Return (%)"
    )
    
    return fig
