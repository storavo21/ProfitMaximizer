import pandas as pd
import numpy as np

def analyze_trading_data(df, profit_column):
    """
    Analyzes trading data to prepare it for filter analysis.
    
    Args:
        df (pd.DataFrame): The trading data
        profit_column (str): The name of the column containing profit/return data
        
    Returns:
        pd.DataFrame: Processed trading data with additional metrics
    """
    # Make a copy of the dataframe to avoid modifying the original
    analyzed_df = df.copy()
    
    # Handle special format in the profit column (e.g., "Lost", "2.1X", "52%")
    if analyzed_df[profit_column].dtype == 'object':
        # Create a numeric version of the profit column
        numeric_profits = []
        
        for value in analyzed_df[profit_column]:
            if pd.isna(value):
                numeric_profits.append(np.nan)
                continue
                
            value_str = str(value).strip().upper()
            
            # Handle "Lost" as negative return (-100%)
            if value_str == 'LOST':
                numeric_profits.append(-100)
            # Handle 'X' multiplier notation (e.g., "2.1X" = 210%)
            elif 'X' in value_str:
                try:
                    multiplier = float(value_str.replace('X', '').strip())
                    # Convert multiplier to percentage (e.g., 2.1X = 210%)
                    percentage = (multiplier - 1) * 100
                    numeric_profits.append(percentage)
                except:
                    numeric_profits.append(np.nan)
            # Handle percentage notation (e.g., "52%")
            elif '%' in value_str:
                try:
                    percentage = float(value_str.replace('%', '').strip())
                    numeric_profits.append(percentage)
                except:
                    numeric_profits.append(np.nan)
            # Try to convert to number directly
            else:
                try:
                    numeric_profits.append(float(value_str))
                except:
                    numeric_profits.append(np.nan)
        
        # Add the numeric profit column
        analyzed_df['numeric_profit'] = numeric_profits
        # Use this as our working profit column
        working_profit_column = 'numeric_profit'
    else:
        # The profit column is already numeric
        working_profit_column = profit_column
    
    # Drop rows where numeric profit is NaN
    analyzed_df = analyzed_df.dropna(subset=[working_profit_column])
    
    # Calculate if the trade was a win or loss (profit >= 100%, or 2X return)
    # This means only trades with at least 2X return count as wins
    analyzed_df['win'] = analyzed_df[working_profit_column] >= 100
    
    # Calculate if the trade had a high return (2x or more, which is >= 100% profit)
    analyzed_df['high_return'] = analyzed_df[working_profit_column] >= 100  # 2x = original + 100%
    
    # Clean and convert categorical columns
    for column in analyzed_df.columns:
        if column not in [profit_column, working_profit_column] and analyzed_df[column].dtype == 'object':
            # Convert to string to handle any non-string objects
            analyzed_df[column] = analyzed_df[column].astype(str)
            # Remove excess whitespace
            analyzed_df[column] = analyzed_df[column].str.strip()
    
    return analyzed_df, working_profit_column

def calculate_summary_metrics(df, profit_column='numeric_profit'):
    """
    Calculates summary metrics for the trading data.
    
    Args:
        df (pd.DataFrame): The analyzed trading data
        profit_column (str): The name of the column containing profit/return data
        
    Returns:
        dict: Dictionary of summary metrics
    """
    metrics = {}
    
    # Ensure we're using the numeric profit column
    if profit_column not in df.columns and 'numeric_profit' in df.columns:
        profit_column = 'numeric_profit'
    elif profit_column not in df.columns and 'profit' in df.columns:
        profit_column = 'profit'
    
    # Calculate basic metrics
    metrics['total_trades'] = len(df)
    metrics['win_count'] = df['win'].sum()
    metrics['loss_count'] = metrics['total_trades'] - metrics['win_count']
    metrics['win_rate'] = (metrics['win_count'] / metrics['total_trades']) * 100 if metrics['total_trades'] > 0 else 0
    
    # Calculate return metrics
    metrics['avg_return'] = df[profit_column].mean()
    metrics['median_return'] = df[profit_column].median()
    metrics['max_return'] = df[profit_column].max()
    metrics['min_return'] = df[profit_column].min()
    
    # Calculate high return metrics
    metrics['high_return_count'] = df['high_return'].sum()
    metrics['high_return_rate'] = (metrics['high_return_count'] / metrics['total_trades']) * 100 if metrics['total_trades'] > 0 else 0
    
    return metrics

def find_profitable_filters(df, columns_to_analyze, profit_column, min_trades=5, target_return=10.0, 
                  use_combined_filters=True, max_columns_to_combine=7, limit_analysis=True, progress_callback=None):
    """
    Identify filters based on column values that maximize profit potential.
    
    Args:
        df (pd.DataFrame): The analyzed trading data
        columns_to_analyze (list): Columns to analyze for profitable patterns
        profit_column (str): The name of the column containing profit/return data
        min_trades (int): Minimum number of trades to consider a filter valid
        target_return (float): Target return multiplier (10.0 = 10x)
        use_combined_filters (bool): Whether to analyze combinations of column values
        max_columns_to_combine (int): Maximum number of columns to combine for filters
        limit_analysis (bool): Whether to limit analysis by using only most common values
        progress_callback (function): Function to call with progress updates
        
    Returns:
        pd.DataFrame: DataFrame with profitable filters
    """
    results = []
    
    # Ensure we're using numeric profit column
    if profit_column not in df.columns and 'numeric_profit' in df.columns:
        profit_column = 'numeric_profit'
    
    # Calculate target return as percentage
    # For a 10x return, it's 900% profit (original + 900% = 10x)
    target_return_pct = (target_return - 1) * 100
    
    # Function to calculate metrics for a filtered dataset
    def calculate_filter_metrics(filtered_data, filter_name, filter_value, column_count=1):
        # Skip if too few trades
        if len(filtered_data) < min_trades:
            return None
            
        # Calculate metrics
        trade_count = len(filtered_data)
        win_count = filtered_data['win'].sum()
        win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0
        
        avg_return = filtered_data[profit_column].mean()
        max_return = filtered_data[profit_column].max()
        
        # Calculate high return rate
        high_return_count = filtered_data['high_return'].sum()
        high_return_rate = (high_return_count / trade_count) * 100 if trade_count > 0 else 0
        
        # Return metrics
        return {
            'filter_name': filter_name,
            'filter_value': filter_value,
            'trade_count': trade_count,
            'win_count': win_count,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_return': max_return,
            'high_return_count': high_return_count,
            'high_return_rate': high_return_rate,
            'column_count': column_count  # Track how many columns are used in the combination
        }
    
    # Function to check if a column is numeric
    def is_numeric_column(column):
        # Known numeric columns by name
        known_numeric_columns = ['made', 'liq sol', 'scans', 'hodls', 'age']
        if column.lower() in known_numeric_columns:
            return True
            
        try:
            # Try to convert to numeric and check if most values are convertible
            numeric_values = pd.to_numeric(df[column], errors='coerce')
            # Consider it numeric if at least 80% of values can be converted
            if numeric_values.notna().mean() > 0.8:
                return True
        except:
            pass
            
        return False
    
    # Create ranges for numeric columns
    def generate_column_ranges(column, num_ranges=4):
        # Try to convert to numeric
        try:
            numeric_values = pd.to_numeric(df[column], errors='coerce')
            
            # Check if we have enough non-NaN values to analyze
            non_nan_count = numeric_values.notna().sum()
            if non_nan_count < min_trades:
                return []
                
            # Remove NaN values
            numeric_values = numeric_values.dropna()
            
            if len(numeric_values) < min_trades:
                return []
                
            # Generate ranges if we have numeric values
            if not numeric_values.empty:
                min_val = numeric_values.min()
                max_val = numeric_values.max()
                
                # If min and max are the same or too close, we can't create proper ranges
                if min_val == max_val:
                    # Just create a single range with the same value
                    if column.lower() == 'liq sol':
                        range_desc = f"{min_val:.1f} SOL"
                    elif column.lower() in ['scans', 'hodls', 'made']:
                        range_desc = f"{int(min_val)}"
                    else:
                        range_desc = f"{min_val:.2f}"
                    return [(min_val, max_val, range_desc)]
                
                # Create ranges
                range_size = (max_val - min_val) / num_ranges
                
                # If range size is too small, we can't create meaningful ranges
                if range_size <= 0.0001:
                    return []
                
                ranges = []
                
                for i in range(num_ranges):
                    range_min = min_val + (i * range_size)
                    range_max = min_val + ((i + 1) * range_size)
                    
                    # For the last range, include the max value
                    if i == num_ranges - 1:
                        range_max = max_val
                    
                    # Format the range description based on column type
                    if column.lower() == 'liq sol':
                        range_desc = f"{range_min:.1f} SOL to {range_max:.1f} SOL"
                    elif column.lower() in ['scans', 'hodls', 'made']:
                        # For integer-like columns, round the range values
                        range_desc = f"{int(range_min)} to {int(range_max)}"
                    else:
                        range_desc = f"{range_min:.2f} to {range_max:.2f}"
                    
                    ranges.append((range_min, range_max, range_desc))
                
                return ranges
        except Exception as e:
            # Print error for debugging but continue silently
            print(f"Error generating ranges for {column}: {str(e)}")
        
        return []

    # Process combinations of columns (required if use_combined_filters is True)
    if len(columns_to_analyze) >= 1:
        # Import itertools for generating combinations
        import itertools
        
        # Build a dictionary to store filter criteria per column
        column_filters = {}
        
        # Process each column to determine how to filter it (exact values vs ranges)
        for col in columns_to_analyze:
            if col == profit_column or col == 'win' or col == 'high_return':
                continue
                
            # Special handling for numeric columns - use ranges
            ranges = generate_column_ranges(col)
            
            # Check if it's known to be a numeric column by name or determined to be numeric
            known_numeric_columns = ['made', 'liq sol', 'scans', 'hodls', 'age']
            if ranges and (col.lower() in known_numeric_columns or is_numeric_column(col)):
                # This is a numeric column, use ranges
                column_filters[col] = {
                    'type': 'range',
                    'ranges': ranges
                }
            else:
                # Use exact values for non-numeric or specifically handled columns
                # Get the most frequent values to avoid too many combinations
                value_counts = df[col].value_counts().head(6)  # Limit to top 6 values
                values = [val for val in value_counts.index if not (pd.isna(val) or val == '' or val == 'nan')]
                
                column_filters[col] = {
                    'type': 'exact',
                    'values': values
                }
        
        # If we have a progress callback, set initial progress
        if progress_callback:
            progress_callback(5, "Analyzing column data...")
        
        # If limit_analysis is True, we'll restrict the analysis to reduce computation time
        if limit_analysis:
            # Limit the number of values per column to analyze
            max_values_per_column = 4
            for col in column_filters:
                if column_filters[col]['type'] == 'exact' and len(column_filters[col]['values']) > max_values_per_column:
                    # Take only the most common values
                    column_filters[col]['values'] = column_filters[col]['values'][:max_values_per_column]
        
        # Count total combinations to track progress
        total_combinations = 0
        combinations_processed = 0
        
        # Calculate the total number of combinations we'll process
        for combo_size in range(1, min(max_columns_to_combine + 1, len(column_filters) + 1)):
            for columns_combo in itertools.combinations(column_filters.keys(), combo_size):
                # Count the product of values/ranges for this column combination
                combo_count = 1
                for col in columns_combo:
                    filter_info = column_filters[col]
                    if filter_info['type'] == 'range':
                        combo_count *= len(filter_info['ranges'])
                    else:
                        combo_count *= len(filter_info['values'])
                total_combinations += combo_count
        
        # Update progress 
        if progress_callback:
            progress_callback(10, f"Processing {total_combinations} filter combinations...")
            
        # Start from 1-column filters and go up to max_columns_to_combine
        for combo_size in range(1, min(max_columns_to_combine + 1, len(column_filters) + 1)):
            # Get all combinations of columns
            for columns_combo in itertools.combinations(column_filters.keys(), combo_size):
                # Update progress based on which column size we're processing
                if progress_callback:
                    progress_percentage = 10 + (combo_size / max_columns_to_combine) * 80
                    progress_callback(progress_percentage, f"Analyzing {combo_size}-column filters...")
                
                # Create filter combinations
                filter_criteria = []
                
                # Create initial empty list for building combinations
                criteria_combinations = [[]]
                
                # For each column in this combination, add its filter criteria
                for col in columns_combo:
                    filter_info = column_filters[col]
                    new_combinations = []
                    
                    if filter_info['type'] == 'range':
                        # For range filters, add a criterion for each range
                        for range_min, range_max, range_desc in filter_info['ranges']:
                            for combo in criteria_combinations:
                                new_combo = combo.copy()
                                new_combo.append({
                                    'column': col,
                                    'type': 'range',
                                    'min': range_min,
                                    'max': range_max,
                                    'desc': f"{col}: {range_desc}"
                                })
                                new_combinations.append(new_combo)
                    else:
                        # For exact value filters, add a criterion for each value
                        for val in filter_info['values']:
                            for combo in criteria_combinations:
                                new_combo = combo.copy()
                                new_combo.append({
                                    'column': col, 
                                    'type': 'exact',
                                    'value': val,
                                    'desc': f"{col}: {val}"
                                })
                                new_combinations.append(new_combo)
                                
                    criteria_combinations = new_combinations
                
                # Show progress on number of combinations being evaluated
                if progress_callback:
                    progress_callback(progress_percentage, f"Evaluating {len(criteria_combinations)} filters for combination of {combo_size} columns...")
                
                # Process each combination in batches to avoid overwhelming memory
                batch_size = 500  # Process 500 combinations at a time
                for batch_start in range(0, len(criteria_combinations), batch_size):
                    batch_end = min(batch_start + batch_size, len(criteria_combinations))
                    criteria_batch = criteria_combinations[batch_start:batch_end]
                    
                    # Now evaluate each combination of filter criteria in this batch
                    for criteria_list in criteria_batch:
                        if not criteria_list:  # Skip empty criteria
                            continue
                            
                        # Create a mask that starts with all rows
                        filter_mask = pd.Series(True, index=df.index)
                        
                        # Apply each criterion to the mask
                        for criterion in criteria_list:
                            if criterion['type'] == 'range':
                                # For range criterion, select rows where column value is between min and max
                                col = criterion['column']
                                try:
                                    # Convert column to numeric for range comparison
                                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                                    range_min = criterion['min']
                                    range_max = criterion['max']
                                    # Apply the range filter
                                    filter_mask = filter_mask & (numeric_col >= range_min) & (numeric_col <= range_max)
                                except:
                                    # If conversion fails, skip this criterion
                                    continue
                            else:
                                # For exact value criterion, select rows where column equals the value
                                col = criterion['column']
                                val = criterion['value']
                                filter_mask = filter_mask & (df[col] == val)
                        
                        # Apply the combined filter - handle empty mask case
                        if filter_mask.sum() == 0:
                            # Skip this combination if no trades match the filter
                            continue
                            
                        try:
                            filtered_data = df[filter_mask]
                            
                            # Skip if we don't have enough trades
                            if len(filtered_data) < min_trades:
                                continue
                        except Exception as e:
                            print(f"Error filtering data: {str(e)}")
                            continue  # Skip this combination on error
                        
                        # Create filter name and description
                        combo_name = f"Combined ({len(criteria_list)} columns)"
                        filter_value = " + ".join([criterion['desc'] for criterion in criteria_list])
                        
                        # Calculate metrics
                        metrics = calculate_filter_metrics(filtered_data, combo_name, filter_value, len(criteria_list))
                        if metrics:
                            results.append(metrics)
                            
                        # Update combinations processed count for progress tracking
                        combinations_processed += 1
                        
                    # Update progress based on combinations processed
                    if progress_callback and total_combinations > 0:
                        progress_percentage = 10 + (combinations_processed / total_combinations) * 80
                        progress_callback(min(90, progress_percentage), f"Processed {combinations_processed} of {total_combinations} combinations...")
                        
        # Final progress update
        if progress_callback:
            progress_callback(95, "Finalizing results...")
    
    # Convert to DataFrame
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Filter to prioritize high 2X+ rates and avoid losing trades
    profitable_filters = results_df[
        # Remember, win_rate now means 2X+ rate since we're only counting 2X+ trades as wins
        (results_df['win_rate'] > 50) |  # Win rate > 50% (higher threshold for 2X+ trades)
        (results_df['high_return_rate'] > 50) |  # Same as win_rate since win = high_return now
        # Include filters that have excellent returns even if win rate is lower
        ((results_df['win_rate'] > 30) & (results_df['avg_return'] > target_return_pct))
    ]
    
    # If we're focusing only on combined filters, filter out any single column filters
    if use_combined_filters:
        profitable_filters = profitable_filters[profitable_filters['filter_name'].str.startswith('Combined')]
    
    return profitable_filters
