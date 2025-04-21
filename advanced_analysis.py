import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, time
import math
import itertools
from scipy import stats
import networkx as nx

def analyze_daily_performance(df, profit_column):
    """
    Analyze trading performance for each day based on the Timestamp column.
    
    Args:
        df (pd.DataFrame): Trading data
        profit_column (str): Column containing profit/return data
        
    Returns:
        tuple: (fig, daily_performance_df) Plotly figure and daily performance dataframe
    """
    # Check if we have a timestamp column with time information
    if 'Timestamp' not in df.columns:
        return None, pd.DataFrame()
    
    # Filter valid data
    filtered_df = df.copy()
    
    # Convert timestamp to datetime if it's not already
    try:
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['Timestamp']):
            filtered_df['datetime'] = pd.to_datetime(filtered_df['Timestamp'])
        else:
            filtered_df['datetime'] = filtered_df['Timestamp']
            
        # Extract date
        filtered_df['date'] = filtered_df['datetime'].dt.date
        
        # Group by date
        daily_performance = []
        
        # Get unique dates
        unique_dates = filtered_df['date'].unique()
        
        for date in unique_dates:
            date_data = filtered_df[filtered_df['date'] == date]
            
            if len(date_data) >= 3:  # Only include dates with at least 3 trades
                win_rate = date_data['win'].mean() * 100 if 'win' in date_data.columns else 0
                avg_return = date_data[profit_column].mean()
                high_return_rate = date_data['high_return'].mean() * 100 if 'high_return' in date_data.columns else 0
                
                daily_performance.append({
                    'date': date,
                    'trade_count': len(date_data),
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'high_return_rate': high_return_rate,
                    'total_profit': date_data[profit_column].sum(),
                    'weekday': date_data['datetime'].iloc[0].strftime('%A')  # Day of week name
                })
        
        daily_df = pd.DataFrame(daily_performance)
        
        if len(daily_df) < 2:  # Not enough days with data
            return None, daily_df
        
        # Calculate combined performance score for each day
        daily_df['performance_score'] = (daily_df['win_rate'] / 100) * daily_df['avg_return']
        
        # Sort by date for the chart
        daily_df = daily_df.sort_values('date')
        
        # Create figure showing performance over days
        fig = go.Figure()
        
        # Add win rate bars
        fig.add_trace(go.Bar(
            x=daily_df['date'],
            y=daily_df['win_rate'],
            name='Win Rate (%)',
            marker_color='green',
            opacity=0.7
        ))
        
        # Add average return line
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['avg_return'],
            name='Avg Return (%)',
            marker_color='blue',
            mode='lines+markers',
            yaxis='y2'
        ))
        
        # Add 2X+ return rate line
        if 'high_return_rate' in daily_df.columns:
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['high_return_rate'],
                name='2X+ Return Rate (%)',
                marker_color='orange',
                mode='lines+markers',
                line=dict(dash='dash'),
                yaxis='y'
            ))
        
        # Add trade count as text
        for i, row in daily_df.iterrows():
            fig.add_annotation(
                x=row['date'],
                y=row['win_rate'] + 5,
                text=f"{row['trade_count']} trades",
                showarrow=False,
                font=dict(size=10)
            )
        
        # Update layout with dual y-axis
        fig.update_layout(
            title='Trading Performance by Day (Based on Timestamp)',
            xaxis=dict(
                title='Date',
                tickangle=45
            ),
            yaxis=dict(
                title='Win Rate & 2X+ Rate (%)',
                range=[0, max(daily_df['win_rate']) * 1.3 if len(daily_df) > 0 else 100]
            ),
            yaxis2=dict(
                title='Avg Return (%)',
                overlaying='y',
                side='right',
                range=[0, max(daily_df['avg_return']) * 1.3 if len(daily_df) > 0 else 100]
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.7)'
            ),
            margin=dict(l=20, r=20, t=40, b=100),
            hovermode='x unified',
            height=600
        )
        
        return fig, daily_df
    
    except Exception as e:
        print(f"Error in daily performance analysis: {str(e)}")
        return None, pd.DataFrame()

def analyze_time_windows(df, profit_column, filter_column=None, filter_value=None):
    """
    Analyze how trading performance varies throughout the day.
    
    Args:
        df (pd.DataFrame): Trading data
        profit_column (str): Column containing profit/return data
        filter_column (str, optional): Column to filter by
        filter_value (str/float, optional): Value to filter for
        
    Returns:
        tuple: (fig, hourly_performance_df) Plotly figure and hourly performance dataframe
    """
    # Check if we have a timestamp column with time information
    if 'Timestamp' not in df.columns:
        return None, pd.DataFrame({
            'hour': [],
            'trade_count': [],
            'win_rate': [],
            'avg_return': []
        })
    
    # Apply filter if specified
    if filter_column is not None and filter_value is not None:
        filtered_df = df[df[filter_column] == filter_value].copy()
    else:
        filtered_df = df.copy()
    
    if len(filtered_df) < 10:  # Not enough data
        return None, pd.DataFrame({
            'hour': [],
            'trade_count': [],
            'win_rate': [],
            'avg_return': []
        })
    
    # Convert timestamp to datetime if it's not already
    try:
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['Timestamp']):
            filtered_df['datetime'] = pd.to_datetime(filtered_df['Timestamp'])
        else:
            filtered_df['datetime'] = filtered_df['Timestamp']
            
        # Extract hour
        filtered_df['hour'] = filtered_df['datetime'].dt.hour
        
        # Group by hour
        hourly_performance = []
        
        for hour in range(24):
            hour_data = filtered_df[filtered_df['hour'] == hour]
            
            if len(hour_data) >= 5:  # Only include hours with at least 5 trades
                win_rate = hour_data['win'].mean() * 100 if 'win' in hour_data.columns else 0
                avg_return = hour_data[profit_column].mean()
                high_return_rate = hour_data['high_return'].mean() * 100 if 'high_return' in hour_data.columns else 0
                
                hourly_performance.append({
                    'hour': hour,
                    'trade_count': len(hour_data),
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'high_return_rate': high_return_rate
                })
        
        hourly_df = pd.DataFrame(hourly_performance)
        
        if len(hourly_df) < 3:  # Not enough hours with data
            return None, hourly_df
        
        # Calculate combined performance score for each hour
        hourly_df['performance_score'] = (hourly_df['win_rate'] / 100) * hourly_df['avg_return']
        
        # Sort by performance score to find the best hours
        hourly_df = hourly_df.sort_values('performance_score', ascending=False).reset_index(drop=True)
        
        # Create plot with dual y-axis for win rate and average return
        fig = go.Figure()
        
        # Add win rate bars
        fig.add_trace(go.Bar(
            x=hourly_df['hour'],
            y=hourly_df['win_rate'],
            name='Win Rate (%)',
            marker_color='green',
            opacity=0.7
        ))
        
        # Add average return line
        fig.add_trace(go.Scatter(
            x=hourly_df['hour'],
            y=hourly_df['avg_return'],
            name='Avg Return (%)',
            marker_color='blue',
            mode='lines+markers',
            yaxis='y2'
        ))
        
        # Add 2X+ return rate line
        if 'high_return_rate' in hourly_df.columns:
            fig.add_trace(go.Scatter(
                x=hourly_df['hour'],
                y=hourly_df['high_return_rate'],
                name='2X+ Return Rate (%)',
                marker_color='orange',
                mode='lines+markers',
                line=dict(dash='dash'),
                yaxis='y'
            ))
        
        # Add trade count as text
        for i, row in hourly_df.iterrows():
            # Mark the best hour with special formatting
            if i == 0:  # Best hour by performance score
                fig.add_annotation(
                    x=row['hour'],
                    y=row['win_rate'] + 5,
                    text=f"BEST HOUR: {row['trade_count']} trades",
                    showarrow=True,
                    arrowhead=1,
                    font=dict(size=12, color="red", family="Arial Black"),
                    bordercolor="red",
                    bgcolor="yellow",
                    borderwidth=2
                )
            else:
                fig.add_annotation(
                    x=row['hour'],
                    y=row['win_rate'] + 5,
                    text=f"{row['trade_count']} trades",
                    showarrow=False,
                    font=dict(size=10)
                )
        
        # Highlight the best hours (top 3) with vertical rectangles
        best_hours = hourly_df.head(3)['hour'].tolist()
        colors = ['rgba(255,255,0,0.2)', 'rgba(255,200,0,0.15)', 'rgba(255,150,0,0.1)']  # Yellow to orange with decreasing opacity
        
        for i, hour in enumerate(best_hours):
            if i < len(colors):
                fig.add_shape(
                    type="rect",
                    x0=hour-0.4, y0=0,
                    x1=hour+0.4, y1=max(hourly_df['win_rate']) * 1.3,
                    fillcolor=colors[i],
                    layer="below",
                    line=dict(width=0)
                )
        
        # Update layout with dual y-axis
        fig.update_layout(
            title='Trading Performance by Hour of Day (Based on Timestamp)',
            xaxis=dict(
                title='Hour of Day',
                tickmode='array',
                tickvals=list(range(24)),
                ticktext=[f"{h}:00" for h in range(24)]
            ),
            yaxis=dict(
                title='Win Rate & 2X+ Rate (%)',
                range=[0, max(hourly_df['win_rate']) * 1.3 if len(hourly_df) > 0 else 100]
            ),
            yaxis2=dict(
                title='Avg Return (%)',
                overlaying='y',
                side='right',
                range=[0, max(hourly_df['avg_return']) * 1.3 if len(hourly_df) > 0 else 100]
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.7)'
            ),
            margin=dict(l=20, r=20, t=40, b=40),
            hovermode='x unified'
        )
        
        return fig, hourly_df
    
    except Exception as e:
        print(f"Error in time window analysis: {str(e)}")
        return None, pd.DataFrame()

def calculate_filter_correlation_matrix(df, filter_columns, profit_column, min_trades=20):
    """
    Calculate correlation matrix between different filters and their performance metrics.
    
    Args:
        df (pd.DataFrame): Analyzed trading data
        filter_columns (list): List of columns to analyze for correlation
        profit_column (str): Name of the column containing profit/return data
        min_trades (int): Minimum number of trades to include a filter
        
    Returns:
        tuple: (correlation_matrix, performance_df, fig) Correlation matrix, performance dataframe, and plotly figure
    """
    try:
        # Initialize list to store filter performance data
        filter_performance = []
        
        # Process each filter column
        for col in filter_columns:
            # Get unique values for this column
            values = df[col].dropna().unique()
            
            for val in values:
                # Filter data for this specific value
                filtered_data = df[df[col] == val]
                
                # Check if we have enough trades
                if len(filtered_data) >= min_trades:
                    # Calculate performance metrics
                    win_rate = filtered_data['win'].mean() * 100 if 'win' in filtered_data.columns else 0
                    avg_return = filtered_data[profit_column].mean()
                    trade_count = len(filtered_data)
                    
                    filter_performance.append({
                        'filter_name': f"{col}: {val}",
                        'column': col,
                        'value': val,
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'trade_count': trade_count,
                        'normalized_score': (win_rate / 100) * avg_return  # Combined metric
                    })
        
        # Convert to DataFrame
        performance_df = pd.DataFrame(filter_performance)
        
        if len(performance_df) < 3:  # Not enough data for correlation
            return None, performance_df, None
        
        # Create a co-occurrence matrix to see which filters often appear together
        correlation_matrix = pd.DataFrame(index=performance_df['filter_name'], columns=performance_df['filter_name'])
        
        # For each pair of filters, count how many trades match both
        for i, filter1 in performance_df.iterrows():
            filter1_data = df[df[filter1['column']] == filter1['value']]
            
            for j, filter2 in performance_df.iterrows():
                filter2_data = df[df[filter2['column']] == filter2['value']]
                
                # Find trades that match both filters
                common_trades = set(filter1_data.index).intersection(set(filter2_data.index))
                correlation_score = len(common_trades) / min(len(filter1_data), len(filter2_data))
                
                correlation_matrix.loc[filter1['filter_name'], filter2['filter_name']] = correlation_score
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            labels=dict(x="Filter", y="Filter", color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            color_continuous_scale="RdBu_r",
            title="Filter Correlation Matrix"
        )
        
        fig.update_layout(
            height=800,
            width=900,
            xaxis=dict(tickangle=45),
            margin=dict(l=20, r=20, t=40, b=150)
        )
        
        return correlation_matrix, performance_df, fig
    
    except Exception as e:
        print(f"Error in correlation matrix: {str(e)}")
        return None, pd.DataFrame(), None

def detect_conflicting_filters(df, filter_columns, profit_column, min_trades=20):
    """
    Detect when filters conflict with each other (one prefers high values, another prefers low values).
    
    Args:
        df (pd.DataFrame): Analyzed trading data
        filter_columns (list): List of columns to analyze for conflicts
        profit_column (str): Name of the column containing profit/return data
        min_trades (int): Minimum number of trades to include a filter
        
    Returns:
        list: List of conflicting filter pairs with explanation
    """
    try:
        conflicts = []
        
        # For each numeric column, determine if high or low values perform better
        column_preferences = {}
        
        for col in filter_columns:
            # Check if column is numeric
            try:
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                non_nan_count = numeric_values.notna().sum()
                
                if non_nan_count < min_trades:
                    continue
                    
                # Sort values and split into quartiles
                sorted_values = numeric_values.sort_values().dropna()
                
                if len(sorted_values) < min_trades:
                    continue
                    
                q1_cutoff = sorted_values.quantile(0.25)
                q4_cutoff = sorted_values.quantile(0.75)
                
                # Low values (Q1)
                low_values_mask = numeric_values <= q1_cutoff
                if low_values_mask.sum() >= min_trades:
                    low_values_data = df[low_values_mask]
                    low_values_win_rate = low_values_data['win'].mean() * 100 if 'win' in low_values_data.columns else 0
                    low_values_return = low_values_data[profit_column].mean()
                    low_values_score = (low_values_win_rate / 100) * low_values_return
                else:
                    low_values_score = 0
                
                # High values (Q4)
                high_values_mask = numeric_values >= q4_cutoff
                if high_values_mask.sum() >= min_trades:
                    high_values_data = df[high_values_mask]
                    high_values_win_rate = high_values_data['win'].mean() * 100 if 'win' in high_values_data.columns else 0
                    high_values_return = high_values_data[profit_column].mean()
                    high_values_score = (high_values_win_rate / 100) * high_values_return
                else:
                    high_values_score = 0
                
                # Determine preference with a threshold for significance
                preference = None
                if high_values_score > low_values_score * 1.5:
                    preference = 'high'
                elif low_values_score > high_values_score * 1.5:
                    preference = 'low'
                
                if preference:
                    column_preferences[col] = {
                        'preference': preference,
                        'high_score': high_values_score,
                        'low_score': low_values_score,
                        'difference': abs(high_values_score - low_values_score)
                    }
            
            except:
                continue
        
        # Check for conflicting relationships between columns
        for col1, pref1 in column_preferences.items():
            for col2, pref2 in column_preferences.items():
                if col1 >= col2:  # Avoid duplicates and self-comparisons
                    continue
                
                # Calculate correlation between the columns
                try:
                    numeric_col1 = pd.to_numeric(df[col1], errors='coerce')
                    numeric_col2 = pd.to_numeric(df[col2], errors='coerce')
                    
                    # Drop rows where either column has NaN
                    valid_mask = numeric_col1.notna() & numeric_col2.notna()
                    if valid_mask.sum() < min_trades:
                        continue
                        
                    correlation = numeric_col1[valid_mask].corr(numeric_col2[valid_mask])
                    
                    # Check if preferences conflict with correlation
                    conflict = False
                    explanation = ""
                    
                    if abs(correlation) > 0.4:  # Only consider meaningful correlations
                        if correlation > 0:  # Positive correlation
                            if pref1['preference'] != pref2['preference']:
                                conflict = True
                                explanation = f"Columns {col1} and {col2} have a positive correlation ({correlation:.2f}), but {col1} performs better with {pref1['preference']} values while {col2} performs better with {pref2['preference']} values."
                        else:  # Negative correlation
                            if pref1['preference'] == pref2['preference']:
                                conflict = True
                                explanation = f"Columns {col1} and {col2} have a negative correlation ({correlation:.2f}), but both perform better with {pref1['preference']} values."
                    
                    if conflict:
                        conflicts.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': correlation,
                            'column1_preference': pref1['preference'],
                            'column2_preference': pref2['preference'],
                            'explanation': explanation,
                            'severity': abs(correlation) * max(pref1['difference'], pref2['difference'])
                        })
                
                except:
                    continue
        
        # Sort conflicts by severity
        conflicts.sort(key=lambda x: x['severity'], reverse=True)
        
        return conflicts
    
    except Exception as e:
        print(f"Error in conflict detection: {str(e)}")
        return []

def calculate_synergy_scores(df, filter_combinations, profit_column):
    """
    Calculate if certain filter combinations produce better results than the sum of individual filters.
    
    Args:
        df (pd.DataFrame): Analyzed trading data
        filter_combinations (pd.DataFrame): DataFrame of filter combinations to analyze
        profit_column (str): Name of the column containing profit/return data
        
    Returns:
        pd.DataFrame: Filter combinations with synergy scores
    """
    try:
        # Initialize results list
        synergy_results = []
        
        # Process each filter combination
        for idx, combo in filter_combinations.iterrows():
            # Skip non-combined filters
            if combo['column_count'] < 2:
                continue
                
            # Parse the individual filters from the combination
            filter_parts = combo['filter_value'].split(" + ")
            individual_filters = []
            
            for part in filter_parts:
                if ": " in part:
                    col, val = part.split(": ", 1)
                    individual_filters.append({'column': col, 'value': val})
            
            # Calculate individual effects
            individual_effects = []
            
            for filter_info in individual_filters:
                col, val = filter_info['column'], filter_info['value']
                
                # Check if this is a range filter
                if " to " in val and col.lower() in ['made', 'liq sol', 'scans', 'hodls', 'age']:
                    # Extract range values
                    range_parts = val.split(" to ")
                    if len(range_parts) == 2:
                        try:
                            # Clean and convert range values
                            range_min = float(range_parts[0].replace("SOL", "").strip())
                            range_max = float(range_parts[1].replace("SOL", "").strip())
                            
                            # Apply range filter
                            numeric_col = pd.to_numeric(df[col], errors='coerce')
                            filtered_data = df[(numeric_col >= range_min) & (numeric_col <= range_max)]
                            
                            win_rate = filtered_data['win'].mean() * 100 if 'win' in filtered_data.columns else 0
                            avg_return = filtered_data[profit_column].mean()
                            
                            individual_effects.append({
                                'filter': f"{col}: {val}",
                                'win_rate': win_rate,
                                'avg_return': avg_return,
                                'score': (win_rate / 100) * avg_return,
                                'trade_count': len(filtered_data)
                            })
                        except:
                            continue
                else:
                    # Regular exact value filter
                    try:
                        filtered_data = df[df[col] == val]
                        
                        win_rate = filtered_data['win'].mean() * 100 if 'win' in filtered_data.columns else 0
                        avg_return = filtered_data[profit_column].mean()
                        
                        individual_effects.append({
                            'filter': f"{col}: {val}",
                            'win_rate': win_rate,
                            'avg_return': avg_return,
                            'score': (win_rate / 100) * avg_return,
                            'trade_count': len(filtered_data)
                        })
                    except:
                        continue
            
            # Skip if we don't have all individual filters
            if len(individual_effects) != len(individual_filters):
                continue
                
            # Calculate expected and actual scores
            if individual_effects:
                # Average of individual filter scores
                avg_individual_score = sum(item['score'] for item in individual_effects) / len(individual_effects)
                
                # Combined filter score from the original data
                combined_score = (combo['win_rate'] / 100) * combo['avg_return']
                
                # Calculate synergy score as ratio of actual/expected
                synergy_score = combined_score / avg_individual_score if avg_individual_score > 0 else 1.0
                
                synergy_results.append({
                    'combo_filter': combo['filter_value'],
                    'column_count': combo['column_count'],
                    'trade_count': combo['trade_count'],
                    'combined_win_rate': combo['win_rate'],
                    'combined_avg_return': combo['avg_return'],
                    'combined_score': combined_score,
                    'avg_individual_score': avg_individual_score,
                    'synergy_score': synergy_score,
                    'synergy_type': 'Positive' if synergy_score > 1.1 else ('Negative' if synergy_score < 0.9 else 'Neutral')
                })
        
        # Convert to DataFrame and sort by synergy score
        synergy_df = pd.DataFrame(synergy_results)
        
        if not synergy_df.empty:
            synergy_df = synergy_df.sort_values('synergy_score', ascending=False)
            
        return synergy_df
    
    except Exception as e:
        print(f"Error in synergy calculation: {str(e)}")
        return pd.DataFrame()

def find_similar_trades(df, target_trade_idx, columns_to_compare, profit_column, top_n=10):
    """
    Find trades similar to a target trade based on matching column values.
    
    Args:
        df (pd.DataFrame): Trading data
        target_trade_idx (int): Index of the target trade
        columns_to_compare (list): Columns to use for similarity matching
        profit_column (str): Name of the column containing profit/return data
        top_n (int): Number of similar trades to return
        
    Returns:
        pd.DataFrame: Similar trades sorted by similarity score
    """
    try:
        # Check if target_trade_idx exists in dataframe
        if target_trade_idx not in df.index:
            return pd.DataFrame()
        
        # Get the target trade
        target_trade = df.loc[target_trade_idx]
        
        # Initialize similarity scores
        similarity_scores = []
        
        # Calculate similarity for each trade
        for idx, trade in df.iterrows():
            if idx == target_trade_idx:
                continue  # Skip the target trade itself
                
            # Count matching values
            matching_values = 0
            total_values = 0
            
            for col in columns_to_compare:
                if col in df.columns:
                    # Skip NaN values
                    if pd.notna(target_trade[col]) and pd.notna(trade[col]):
                        total_values += 1
                        if target_trade[col] == trade[col]:
                            matching_values += 1
            
            # Calculate similarity as percentage of matching values
            if total_values > 0:
                similarity = (matching_values / total_values) * 100
                
                similarity_scores.append({
                    'index': idx,
                    'similarity': similarity,
                    'matching_columns': matching_values,
                    'total_columns': total_values,
                    'profit': trade[profit_column] if profit_column in trade else None
                })
        
        # Convert to DataFrame
        similar_trades_df = pd.DataFrame(similarity_scores)
        
        if similar_trades_df.empty:
            return pd.DataFrame()
            
        # Sort by similarity and get top N
        similar_trades_df = similar_trades_df.sort_values('similarity', ascending=False).head(top_n)
        
        # Merge with original data for all columns
        result = pd.merge(similar_trades_df, df, left_on='index', right_index=True)
        
        return result
    
    except Exception as e:
        print(f"Error finding similar trades: {str(e)}")
        return pd.DataFrame()

def calculate_pattern_consistency(df, filter_combinations, profit_column):
    """
    Score filters based on how consistently they identify the same types of setups.
    
    Args:
        df (pd.DataFrame): Trading data
        filter_combinations (pd.DataFrame): DataFrame of filter combinations
        profit_column (str): Name of the column containing profit/return data
        
    Returns:
        pd.DataFrame: Filters with consistency scores
    """
    try:
        # Initialize results
        consistency_results = []
        
        # Process each filter
        for idx, filter_combo in filter_combinations.iterrows():
            # Parse the filter
            filter_value = filter_combo['filter_value']
            filter_parts = filter_value.split(" + ")
            
            # Prepare a mask for this filter
            filter_mask = pd.Series(True, index=df.index)
            column_values = {}
            
            # Apply each filter part
            for part in filter_parts:
                if ": " in part:
                    col, val = part.split(": ", 1)
                    column_values[col] = val
                    
                    # Handle range filters
                    if " to " in val and col.lower() in ['made', 'liq sol', 'scans', 'hodls', 'age']:
                        try:
                            range_parts = val.split(" to ")
                            range_min = float(range_parts[0].replace("SOL", "").strip())
                            range_max = float(range_parts[1].replace("SOL", "").strip())
                            
                            numeric_col = pd.to_numeric(df[col], errors='coerce')
                            range_mask = (numeric_col >= range_min) & (numeric_col <= range_max)
                            filter_mask = filter_mask & range_mask
                        except:
                            continue
                    else:
                        # Regular exact value filter
                        try:
                            filter_mask = filter_mask & (df[col] == val)
                        except:
                            continue
            
            # Get filtered trades
            filtered_trades = df[filter_mask]
            
            if len(filtered_trades) < 5:
                continue
            
            # Calculate profit stability metrics
            profit_std = filtered_trades[profit_column].std()
            profit_mean = filtered_trades[profit_column].mean()
            coefficient_variation = profit_std / profit_mean if profit_mean > 0 else float('inf')
            
            # Calculate pattern consistency across additional columns
            pattern_columns = [col for col in df.columns if col not in column_values and col != profit_column]
            column_consistency_scores = {}
            
            for col in pattern_columns:
                try:
                    # Skip columns with too many unique values or all same values
                    value_counts = filtered_trades[col].value_counts()
                    if len(value_counts) <= 1 or len(value_counts) > len(filtered_trades) * 0.8:
                        continue
                        
                    # Calculate concentration - higher means more consistent
                    top_value_concentration = value_counts.iloc[0] / len(filtered_trades)
                    entropy = stats.entropy(value_counts / len(filtered_trades))
                    
                    column_consistency_scores[col] = {
                        'top_value': value_counts.index[0],
                        'top_value_pct': top_value_concentration * 100,
                        'entropy': entropy
                    }
                except:
                    continue
            
            # Calculate overall consistency score
            if column_consistency_scores:
                # Average top value concentration
                avg_concentration = sum(val['top_value_pct'] for val in column_consistency_scores.values()) / len(column_consistency_scores)
                
                # Average entropy (lower is better - more consistent)
                avg_entropy = sum(val['entropy'] for val in column_consistency_scores.values()) / len(column_consistency_scores)
                
                # Normalize coefficient of variation (lower is better - more consistent)
                cv_score = 100 / (1 + coefficient_variation) if coefficient_variation != float('inf') else 0
                
                # Combined score (higher is better)
                consistency_score = (avg_concentration + cv_score) / 2
            else:
                avg_concentration = 0
                avg_entropy = 0 
                cv_score = 0
                consistency_score = 0
            
            # Add pattern insights
            pattern_insights = []
            for col, metrics in column_consistency_scores.items():
                if metrics['top_value_pct'] > 70:  # Only include strong patterns
                    pattern_insights.append(f"{col} is usually '{metrics['top_value']}' ({metrics['top_value_pct']:.1f}%)")
            
            consistency_results.append({
                'filter_value': filter_value,
                'trade_count': len(filtered_trades),
                'profit_mean': profit_mean,
                'profit_std': profit_std,
                'coefficient_variation': coefficient_variation,
                'avg_value_consistency': avg_concentration,
                'pattern_entropy': avg_entropy,
                'consistency_score': consistency_score,
                'pattern_insights': pattern_insights
            })
        
        # Convert to DataFrame
        consistency_df = pd.DataFrame(consistency_results)
        
        if not consistency_df.empty:
            consistency_df = consistency_df.sort_values('consistency_score', ascending=False)
            
        return consistency_df
    
    except Exception as e:
        print(f"Error calculating pattern consistency: {str(e)}")
        return pd.DataFrame()

def detect_outlier_trades(df, profit_column, z_score_threshold=3.0):
    """
    Identify and highlight unusual trades that don't fit normal patterns.
    
    Args:
        df (pd.DataFrame): Trading data
        profit_column (str): Name of the column containing profit/return data
        z_score_threshold (float): Z-score threshold for outlier detection
        
    Returns:
        tuple: (outlier_df, fig) DataFrame of outliers and visualization
    """
    try:
        # Calculate z-scores for profit/return
        profit_mean = df[profit_column].mean()
        profit_std = df[profit_column].std()
        
        if profit_std == 0:  # Avoid division by zero
            return pd.DataFrame(), None
            
        df['profit_z_score'] = (df[profit_column] - profit_mean) / profit_std
        
        # Identify outliers
        outliers = df[abs(df['profit_z_score']) > z_score_threshold].copy()
        
        if len(outliers) == 0:
            return pd.DataFrame(), None
            
        # Add outlier type
        outliers['outlier_type'] = outliers['profit_z_score'].apply(
            lambda x: 'Extremely High Profit' if x > z_score_threshold else 'Extremely Low Profit'
        )
        
        # Create visualization
        fig = px.scatter(
            df, x=df.index, y=profit_column,
            color=abs(df['profit_z_score']) > z_score_threshold,
            color_discrete_map={True: 'red', False: 'blue'},
            labels={profit_column: 'Profit/Return', 'index': 'Trade Index'},
            title='Outlier Trades Detection'
        )
        
        # Add horizontal line for mean
        fig.add_hline(
            y=profit_mean, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Mean: {profit_mean:.2f}%"
        )
        
        # Add horizontal lines for outlier thresholds
        fig.add_hline(
            y=profit_mean + z_score_threshold * profit_std, 
            line_dash="dot", 
            line_color="red",
            annotation_text=f"Upper threshold"
        )
        
        fig.add_hline(
            y=profit_mean - z_score_threshold * profit_std, 
            line_dash="dot", 
            line_color="red",
            annotation_text=f"Lower threshold"
        )
        
        fig.update_layout(height=600)
        
        return outliers, fig
    
    except Exception as e:
        print(f"Error detecting outliers: {str(e)}")
        return pd.DataFrame(), None

def create_decision_tree_visualization(filter_combinations, max_depth=3):
    """
    Create a visual flowchart showing how to apply multiple filters in sequence.
    
    Args:
        filter_combinations (pd.DataFrame): DataFrame of filter combinations with performance metrics
        max_depth (int): Maximum depth of the decision tree
        
    Returns:
        plotly.graph_objects.Figure: Decision tree visualization
    """
    try:
        if filter_combinations.empty or 'filter_value' not in filter_combinations.columns:
            return None
            
        # Extract filter parts from combined filters
        all_filter_parts = []
        for idx, row in filter_combinations.iterrows():
            if 'filter_value' in row and ' + ' in row['filter_value']:
                parts = row['filter_value'].split(' + ')
                all_filter_parts.extend([(part, row['win_rate'], row['avg_return']) for part in parts])
        
        # Count frequency and success of each filter part
        filter_stats = {}
        for part, win_rate, avg_return in all_filter_parts:
            if part not in filter_stats:
                filter_stats[part] = {
                    'count': 0,
                    'win_rate_sum': 0,
                    'avg_return_sum': 0
                }
            filter_stats[part]['count'] += 1
            filter_stats[part]['win_rate_sum'] += win_rate
            filter_stats[part]['avg_return_sum'] += avg_return
        
        # Calculate average metrics
        for part in filter_stats:
            count = filter_stats[part]['count']
            filter_stats[part]['avg_win_rate'] = filter_stats[part]['win_rate_sum'] / count if count > 0 else 0
            filter_stats[part]['avg_return'] = filter_stats[part]['avg_return_sum'] / count if count > 0 else 0
            filter_stats[part]['score'] = filter_stats[part]['avg_win_rate'] * filter_stats[part]['avg_return'] / 100
        
        # Sort filter parts by score
        sorted_parts = sorted(filter_stats.keys(), key=lambda x: filter_stats[x]['score'], reverse=True)
        
        # Limit to top parts for decision tree
        top_parts = sorted_parts[:8]  # Limit to 8 nodes for clarity
        
        # Create a graph
        G = nx.DiGraph()
        
        # Add start node
        G.add_node("Start")
        
        # Build decision tree
        for i, part in enumerate(top_parts[:max_depth]):
            if i == 0:
                # First level connects from start
                G.add_edge("Start", part)
                G.add_edge(part, "High Win Rate", weight=filter_stats[part]['avg_win_rate'])
                G.add_edge(part, "Low Win Rate", weight=100-filter_stats[part]['avg_win_rate'])
            else:
                # Connect from previous high win rate node
                prev_part = top_parts[i-1]
                G.add_edge("High Win Rate", part)
                
                # Add result nodes for this level
                success_node = f"Success ({filter_stats[part]['avg_win_rate']:.1f}%)"
                fail_node = f"Limited Success ({100-filter_stats[part]['avg_win_rate']:.1f}%)"
                
                G.add_edge(part, success_node, weight=filter_stats[part]['avg_win_rate'])
                G.add_edge(part, fail_node, weight=100-filter_stats[part]['avg_win_rate'])
        
        # Create positions for nodes
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Add edge weight as text if available
            if 'weight' in G.edges[edge]:
                weight = G.edges[edge]['weight']
                edge_text.append(f"{weight:.1f}%")
            else:
                edge_text.append("")
        
        # Create a trace for edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Format node text
            if node == "Start":
                node_text.append("Start Trading Analysis")
                node_size.append(30)
                node_color.append('blue')
            elif node in filter_stats:
                # For filter nodes, show filter and metrics
                metrics = filter_stats[node]
                node_text.append(f"{node}<br>Win Rate: {metrics['avg_win_rate']:.1f}%<br>Avg Return: {metrics['avg_return']:.1f}%")
                node_size.append(20 + metrics['count'])
                node_color.append('green')
            elif "Success" in node:
                # Success nodes
                node_text.append(f"{node}")
                node_size.append(15)
                node_color.append('gold')
            else:
                # Other nodes
                node_text.append(f"{node}")
                node_size.append(15)
                node_color.append('red')
        
        # Create a trace for nodes
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=2, color='#000')
            )
        )
        
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          title="Decision Tree for Filter Application",
                          titlefont=dict(size=16),
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          height=800,
                          width=1000,
                          annotations=[
                              dict(
                                  text="Follow the paths with higher percentages for better trading outcomes",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.01, y=0.01
                              )
                          ]
                      ))
        
        return fig
    
    except Exception as e:
        print(f"Error creating decision tree: {str(e)}")
        return None

def find_optimal_filter_combinations(df, filter_columns, profit_column, min_trades=20, max_combinations=5000):
    """
    Automatically test thousands of filter combinations to find optimal sets.
    
    Args:
        df (pd.DataFrame): Trading data
        filter_columns (list): List of columns to include in filter combinations
        profit_column (str): Name of the column containing profit/return data
        min_trades (int): Minimum trades for a combination to be valid
        max_combinations (int): Maximum number of combinations to test
        
    Returns:
        pd.DataFrame: Top performing filter combinations
    """
    try:
        # Initialize results
        optimal_results = []
        
        # Track number of combinations tested
        combinations_tested = 0
        
        # Create dictionaries to store filter options by column
        column_filters = {}
        
        # Process each column
        for col in filter_columns:
            # For numeric columns, create quantile ranges
            if pd.api.types.is_numeric_dtype(df[col]) or col.lower() in ['made', 'liq sol', 'scans', 'hodls', 'age']:
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    
                    # Skip columns with too many NaN values
                    if numeric_values.isna().mean() > 0.5:
                        continue
                        
                    # Create ranges based on quantiles (0-25%, 25-50%, 50-75%, 75-100%)
                    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
                    quantile_values = [numeric_values.quantile(q) for q in quantiles if q < 1.0] + [numeric_values.max()]
                    
                    ranges = []
                    for i in range(len(quantile_values) - 1):
                        lower = quantile_values[i]
                        upper = quantile_values[i + 1]
                        
                        # Format range based on column
                        if col.lower() == 'liq sol':
                            range_str = f"{lower:.1f} SOL to {upper:.1f} SOL"
                        elif col.lower() in ['made', 'scans', 'hodls']:
                            range_str = f"{int(lower)} to {int(upper)}"
                        else:
                            range_str = f"{lower:.2f} to {upper:.2f}"
                            
                        # Calculate performance for this range
                        range_mask = (numeric_values >= lower) & (numeric_values <= upper)
                        
                        if range_mask.sum() >= min_trades:
                            filtered_data = df[range_mask]
                            win_rate = filtered_data['win'].mean() * 100 if 'win' in filtered_data.columns else 0
                            avg_return = filtered_data[profit_column].mean()
                            
                            # Score this range
                            range_score = (win_rate / 100) * avg_return
                            
                            ranges.append({
                                'range_str': range_str,
                                'lower': lower,
                                'upper': upper,
                                'score': range_score,
                                'trade_count': range_mask.sum()
                            })
                    
                    # Store ranges for this column
                    if ranges:
                        column_filters[col] = {
                            'type': 'range',
                            'ranges': ranges
                        }
                        
                except Exception as e:
                    print(f"Error processing numeric column {col}: {str(e)}")
                    continue
                    
            else:
                # For categorical columns, use top values
                try:
                    value_counts = df[col].value_counts()
                    
                    values = []
                    for val, count in value_counts.items():
                        if count >= min_trades and pd.notna(val) and val != '':
                            # Calculate performance for this value
                            filtered_data = df[df[col] == val]
                            win_rate = filtered_data['win'].mean() * 100 if 'win' in filtered_data.columns else 0
                            avg_return = filtered_data[profit_column].mean()
                            
                            # Score this value
                            value_score = (win_rate / 100) * avg_return
                            
                            values.append({
                                'value': val,
                                'score': value_score,
                                'trade_count': count
                            })
                    
                    # Store values for this column
                    if values:
                        column_filters[col] = {
                            'type': 'categorical',
                            'values': values
                        }
                        
                except Exception as e:
                    print(f"Error processing categorical column {col}: {str(e)}")
                    continue
        
        # Sort filter options by score
        for col in column_filters:
            if column_filters[col]['type'] == 'range':
                column_filters[col]['ranges'] = sorted(column_filters[col]['ranges'], 
                                                      key=lambda x: x['score'], 
                                                      reverse=True)
            else:
                column_filters[col]['values'] = sorted(column_filters[col]['values'], 
                                                      key=lambda x: x['score'], 
                                                      reverse=True)
        
        # Generate combinations (start with columns that have the best individual performance)
        sorted_columns = sorted(column_filters.keys(), 
                                key=lambda col: max([opt['score'] for opt in column_filters[col]['ranges']] 
                                                   if column_filters[col]['type'] == 'range' 
                                                   else [opt['score'] for opt in column_filters[col]['values']]),
                                reverse=True)
        
        # Test combinations of 2-4 filters
        for combo_size in range(2, 5):
            if combinations_tested >= max_combinations:
                break
                
            # Use most promising columns first
            for cols in itertools.combinations(sorted_columns[:min(8, len(sorted_columns))], combo_size):
                if combinations_tested >= max_combinations:
                    break
                    
                # Generate combinations of filter options for these columns
                filter_options = []
                
                for col in cols:
                    if column_filters[col]['type'] == 'range':
                        # Use top 2 ranges for each column
                        col_options = [(col, r['range_str']) for r in column_filters[col]['ranges'][:2]]
                    else:
                        # Use top 2 values for each column
                        col_options = [(col, v['value']) for v in column_filters[col]['values'][:2]]
                        
                    filter_options.append(col_options)
                
                # Test each combination
                for filter_combo in itertools.product(*filter_options):
                    combinations_tested += 1
                    
                    if combinations_tested >= max_combinations:
                        break
                        
                    # Create filter mask
                    filter_mask = pd.Series(True, index=df.index)
                    filter_parts = []
                    
                    for col, val in filter_combo:
                        if column_filters[col]['type'] == 'range':
                            # Parse range
                            try:
                                range_parts = val.split(" to ")
                                if len(range_parts) == 2:
                                    lower = float(range_parts[0].replace("SOL", "").strip())
                                    upper = float(range_parts[1].replace("SOL", "").strip())
                                    
                                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                                    range_mask = (numeric_col >= lower) & (numeric_col <= upper)
                                    filter_mask = filter_mask & range_mask
                                    filter_parts.append(f"{col}: {val}")
                            except:
                                continue
                        else:
                            # Exact value
                            filter_mask = filter_mask & (df[col] == val)
                            filter_parts.append(f"{col}: {val}")
                    
                    # Apply filter and calculate performance
                    filtered_data = df[filter_mask]
                    
                    if len(filtered_data) >= min_trades:
                        win_rate = filtered_data['win'].mean() * 100 if 'win' in filtered_data.columns else 0
                        avg_return = filtered_data[profit_column].mean()
                        max_return = filtered_data[profit_column].max()
                        high_return_rate = filtered_data['high_return'].mean() * 100 if 'high_return' in filtered_data.columns else 0
                        
                        # Calculate combined score
                        combined_score = (win_rate / 100) * avg_return
                        
                        optimal_results.append({
                            'filter_value': " + ".join(filter_parts),
                            'column_count': len(filter_parts),
                            'trade_count': len(filtered_data),
                            'win_rate': win_rate,
                            'avg_return': avg_return,
                            'max_return': max_return,
                            'high_return_rate': high_return_rate,
                            'combined_score': combined_score
                        })
        
        # Convert to DataFrame
        optimal_df = pd.DataFrame(optimal_results)
        
        if not optimal_df.empty:
            # Sort by combined score
            optimal_df = optimal_df.sort_values('combined_score', ascending=False)
            
        return optimal_df
    
    except Exception as e:
        print(f"Error finding optimal combinations: {str(e)}")
        return pd.DataFrame()