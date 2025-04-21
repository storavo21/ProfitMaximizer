import streamlit as st
import pandas as pd
import numpy as np
import io
import itertools
from datetime import datetime, time
from analyzer import analyze_trading_data, find_profitable_filters, calculate_summary_metrics
from visualization import plot_profit_distribution, plot_filter_effectiveness, plot_win_rate_by_filter
from advanced_analysis import (
    analyze_time_windows, calculate_filter_correlation_matrix,
    detect_conflicting_filters, calculate_synergy_scores,
    find_similar_trades, calculate_pattern_consistency,
    detect_outlier_trades, create_decision_tree_visualization,
    find_optimal_filter_combinations, analyze_daily_performance
)

# Set page configuration
st.set_page_config(
    page_title="Trading Data Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title and introduction
st.title("Trading Data Analyzer 📊")
st.markdown("""
This tool analyzes your trading data to identify specific filters that maximize profit potential, 
targeting 10x+ returns while minimizing losses. Upload your Excel trading sheet to get started.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your trading data Excel file", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Load the data
        with st.spinner('Loading and analyzing your trading data...'):
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Display basic info about the data
            st.subheader("Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total trades: {len(df)}")
            with col2:
                st.write(f"Columns: {', '.join(df.columns)}")
            
            # Display full data with option to see more
            with st.expander("View Full Data"):
                st.dataframe(df)
            
            # Ensure there are required columns for analysis
            required_columns = ["profit", "return", "result"]
            missing_columns = []
            
            # Check for profit/return column
            profit_column = None
            if "profit" in df.columns:
                profit_column = "profit"
            elif "return" in df.columns:
                profit_column = "return"
            elif "result" in df.columns:
                profit_column = "result"
            else:
                missing_columns.append("profit/return/result")
            
            if missing_columns:
                st.error(f"Your data is missing required columns: {', '.join(missing_columns)}. Please ensure your Excel file contains columns indicating trade profit or return values.")
            else:
                # Analyze the data
                analyzed_data, working_profit_column = analyze_trading_data(df, profit_column)
                
                # Calculate summary metrics
                summary_metrics = calculate_summary_metrics(analyzed_data, working_profit_column)
                
                # Display summary metrics
                st.subheader("Trading Performance Summary")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    st.metric("Win Rate", f"{summary_metrics['win_rate']:.2f}%")
                with metrics_col2:
                    st.metric("Avg Return", f"{summary_metrics['avg_return']:.2f}%")
                with metrics_col3:
                    st.metric("Max Return", f"{summary_metrics['max_return']:.2f}%")
                with metrics_col4:
                    st.metric("10x+ Trades", f"{summary_metrics['high_return_count']}")
                
                # Plot profit distribution
                st.subheader("Profit Distribution")
                profit_fig = plot_profit_distribution(analyzed_data, working_profit_column)
                st.plotly_chart(profit_fig, use_container_width=True)
                
                # Advanced Analysis Features Toggle
                st.subheader("🚀 Advanced Analysis Features")
                
                # Create a session state to store advanced features state
                if 'advanced_enabled' not in st.session_state:
                    st.session_state.advanced_enabled = False
                
                # Add option to disable if there are issues
                col1, col2 = st.columns([3, 1])
                with col1:
                    enable_advanced = st.checkbox("Enable Advanced Analysis Features", value=st.session_state.advanced_enabled,
                                            help="Toggle advanced analysis features like time windows, pattern recognition, etc.")
                with col2:
                    if st.button("⚠️ Disable All Advanced Features", help="Click this to disable all advanced features if they cause issues"):
                        st.session_state.advanced_enabled = False
                        st.rerun()
                
                # Update session state
                st.session_state.advanced_enabled = enable_advanced
                
                # Advanced Analysis Section
                if enable_advanced:
                    # Create tabs for hourly and daily analysis
                    time_tabs = st.tabs(["🕒 Hourly Analysis", "📅 Daily Analysis"])
                    
                    # Hourly analysis tab
                    with time_tabs[0]:
                        st.write("## 🕒 Trading Time Windows Analysis")
                        st.write("This analysis shows how your trading performance varies by hour of the day based on the 'Timestamp' column.")
                        
                        if 'Timestamp' in df.columns:
                            time_fig, hourly_df = analyze_time_windows(analyzed_data, working_profit_column)
                            
                            if time_fig is not None:
                                st.plotly_chart(time_fig, use_container_width=True)
                                
                                # Show best hour recommendations
                                st.write("### 🏆 Best Hours to Trade")
                                
                                # Add detailed explanation about the best hours
                                if not hourly_df.empty and len(hourly_df) > 0:
                                    # Get top 3 hours by performance score
                                    best_hours = hourly_df.head(3)
                                    
                                    # Create metrics display
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if len(best_hours) >= 1:
                                            first_best = best_hours.iloc[0]
                                            st.metric(
                                                f"🥇 BEST HOUR: {int(first_best['hour']):02d}:00", 
                                                f"{first_best['performance_score']:.2f} score",
                                                f"{first_best['trade_count']} trades"
                                            )
                                            
                                    with col2:
                                        if len(best_hours) >= 2:
                                            second_best = best_hours.iloc[1]
                                            st.metric(
                                                f"🥈 2nd BEST: {int(second_best['hour']):02d}:00", 
                                                f"{second_best['performance_score']:.2f} score",
                                                f"{second_best['trade_count']} trades"
                                            )
                                            
                                    with col3:
                                        if len(best_hours) >= 3:
                                            third_best = best_hours.iloc[2]
                                            st.metric(
                                                f"🥉 3rd BEST: {int(third_best['hour']):02d}:00", 
                                                f"{third_best['performance_score']:.2f} score",
                                                f"{third_best['trade_count']} trades"
                                            )
                                
                                # Show detailed hourly performance table
                                with st.expander("📊 View Detailed Hourly Performance Table"):
                                    if not hourly_df.empty:
                                        # Format percentages for display
                                        display_df = hourly_df.copy()
                                        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2f}%")
                                        display_df['avg_return'] = display_df['avg_return'].apply(lambda x: f"{x:.2f}%")
                                        if 'high_return_rate' in display_df.columns:
                                            display_df['high_return_rate'] = display_df['high_return_rate'].apply(lambda x: f"{x:.2f}%")
                                        display_df['performance_score'] = display_df['performance_score'].apply(lambda x: f"{x:.2f}")
                                        
                                        # Add hour of day for better readability
                                        display_df['hour_of_day'] = display_df['hour'].apply(lambda x: f"{int(x):02d}:00")
                                        
                                        # Reorder columns for better display
                                        columns_order = ['hour_of_day', 'performance_score', 'win_rate', 'avg_return']
                                        if 'high_return_rate' in display_df.columns:
                                            columns_order.append('high_return_rate')
                                        columns_order.append('trade_count')
                                        
                                        # Create a styled dataframe with better column names
                                        st.dataframe(
                                            display_df[columns_order].rename(columns={
                                                'hour_of_day': 'Hour of Day',
                                                'performance_score': 'Performance Score',
                                                'win_rate': 'Win Rate',
                                                'avg_return': 'Avg Return',
                                                'high_return_rate': '2X+ Return Rate',
                                                'trade_count': 'Trade Count'
                                            }),
                                            use_container_width=True
                                        )
                                
                                # Add trading time insights
                                st.write("### 📝 Trading Time Insights")
                                
                                if not hourly_df.empty and len(hourly_df) > 2:
                                    # Best time ranges (morning, afternoon, evening)
                                    morning_hours = hourly_df[(hourly_df['hour'] >= 6) & (hourly_df['hour'] < 12)]
                                    afternoon_hours = hourly_df[(hourly_df['hour'] >= 12) & (hourly_df['hour'] < 18)]
                                    evening_hours = hourly_df[(hourly_df['hour'] >= 18) | (hourly_df['hour'] < 6)]
                                    
                                    morning_score = morning_hours['performance_score'].mean() if not morning_hours.empty else 0
                                    afternoon_score = afternoon_hours['performance_score'].mean() if not afternoon_hours.empty else 0
                                    evening_score = evening_hours['performance_score'].mean() if not evening_hours.empty else 0
                                    
                                    best_time = "Morning (6:00-11:59)" if morning_score >= max(afternoon_score, evening_score) else \
                                               "Afternoon (12:00-17:59)" if afternoon_score >= max(morning_score, evening_score) else \
                                               "Evening/Night (18:00-5:59)"
                                    
                                    # Calculate the day segments with the best performance
                                    segments = [
                                        {"name": "Morning (6:00-11:59)", "score": morning_score},
                                        {"name": "Afternoon (12:00-17:59)", "score": afternoon_score},
                                        {"name": "Evening/Night (18:00-5:59)", "score": evening_score}
                                    ]
                                    segments.sort(key=lambda x: x["score"], reverse=True)
                                    
                                    # Display insights
                                    st.info(f"🕐 Based on your trading data, the best time of day to trade is: **{best_time}**")
                                    
                                    # Additional insights about best hours
                                    best_hour = hourly_df.iloc[0]
                                    st.success(f"🎯 The single best hour for trading is **{int(best_hour['hour']):02d}:00** with a win rate of **{best_hour['win_rate']:.2f}%** and average return of **{best_hour['avg_return']:.2f}%**")
                                    
                                    # Provide ranking of day segments
                                    st.markdown("**Day Segment Ranking:**")
                                    for i, segment in enumerate(segments, 1):
                                        st.markdown(f"{i}. **{segment['name']}** - Score: {segment['score']:.2f}")
                            else:
                                st.warning("Not enough time data to perform time window analysis. Ensure your dataset has timestamps and trades across different hours.")
                        else:
                            st.error("Time window analysis requires a 'Timestamp' column in your data. Please make sure your trading data includes this column.")

                    # Daily analysis tab
                    with time_tabs[1]:
                        st.write("## 📅 Day-by-Day Trading Analysis")
                        st.write("This analysis shows your trading performance for each individual day based on the 'Timestamp' column.")
                        
                        if 'Timestamp' in df.columns:
                            day_fig, daily_df = analyze_daily_performance(analyzed_data, working_profit_column)
                            
                            if day_fig is not None:
                                st.plotly_chart(day_fig, use_container_width=True)
                                
                                # Show daily performance metrics
                                if not daily_df.empty and len(daily_df) > 0:
                                    # Calculate top performing days
                                    best_days_by_performance = daily_df.sort_values('performance_score', ascending=False).head(3)
                                    best_days_by_profit = daily_df.sort_values('total_profit', ascending=False).head(3)
                                    
                                    # Create metrics display with tabs
                                    day_metric_tabs = st.tabs(["🏆 Best Days by Performance", "💰 Best Days by Total Profit", "📊 Weekday Analysis"])
                                    
                                    # Tab 1: Best Days by Performance Score
                                    with day_metric_tabs[0]:
                                        st.write("### Top Trading Days by Performance Score")
                                        
                                        # Create columns for top days
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            if len(best_days_by_performance) >= 1:
                                                first_best = best_days_by_performance.iloc[0]
                                                st.metric(
                                                    f"🥇 {first_best['date'].strftime('%Y-%m-%d')} ({first_best['weekday']})", 
                                                    f"{first_best['performance_score']:.2f} score",
                                                    f"{first_best['trade_count']} trades"
                                                )
                                                
                                        with col2:
                                            if len(best_days_by_performance) >= 2:
                                                second_best = best_days_by_performance.iloc[1]
                                                st.metric(
                                                    f"🥈 {second_best['date'].strftime('%Y-%m-%d')} ({second_best['weekday']})", 
                                                    f"{second_best['performance_score']:.2f} score",
                                                    f"{second_best['trade_count']} trades"
                                                )
                                                
                                        with col3:
                                            if len(best_days_by_performance) >= 3:
                                                third_best = best_days_by_performance.iloc[2]
                                                st.metric(
                                                    f"🥉 {third_best['date'].strftime('%Y-%m-%d')} ({third_best['weekday']})", 
                                                    f"{third_best['performance_score']:.2f} score",
                                                    f"{third_best['trade_count']} trades"
                                                )
                                    
                                    # Tab 2: Best Days by Total Profit
                                    with day_metric_tabs[1]:
                                        st.write("### Top Trading Days by Total Profit")
                                        
                                        # Create columns for top days
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            if len(best_days_by_profit) >= 1:
                                                first_best = best_days_by_profit.iloc[0]
                                                st.metric(
                                                    f"🥇 {first_best['date'].strftime('%Y-%m-%d')} ({first_best['weekday']})", 
                                                    f"{first_best['total_profit']:.2f}% total",
                                                    f"{first_best['trade_count']} trades"
                                                )
                                                
                                        with col2:
                                            if len(best_days_by_profit) >= 2:
                                                second_best = best_days_by_profit.iloc[1]
                                                st.metric(
                                                    f"🥈 {second_best['date'].strftime('%Y-%m-%d')} ({second_best['weekday']})", 
                                                    f"{second_best['total_profit']:.2f}% total",
                                                    f"{second_best['trade_count']} trades"
                                                )
                                                
                                        with col3:
                                            if len(best_days_by_profit) >= 3:
                                                third_best = best_days_by_profit.iloc[2]
                                                st.metric(
                                                    f"🥉 {third_best['date'].strftime('%Y-%m-%d')} ({third_best['weekday']})", 
                                                    f"{third_best['total_profit']:.2f}% total",
                                                    f"{third_best['trade_count']} trades"
                                                )
                                    
                                    # Tab 3: Weekday Analysis
                                    with day_metric_tabs[2]:
                                        st.write("### Performance by Day of Week")
                                        
                                        # Group by weekday
                                        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                        
                                        if 'weekday' in daily_df.columns:
                                            # Group by weekday and calculate average metrics
                                            weekday_df = daily_df.groupby('weekday').agg({
                                                'win_rate': 'mean',
                                                'avg_return': 'mean',
                                                'trade_count': 'sum',
                                                'performance_score': 'mean',
                                                'high_return_rate': 'mean' if 'high_return_rate' in daily_df.columns else None,
                                                'total_profit': 'sum'
                                            }).reset_index()
                                            
                                            # Sort by weekday in correct order
                                            weekday_df['sort_order'] = weekday_df['weekday'].apply(lambda x: weekday_order.index(x) if x in weekday_order else 999)
                                            weekday_df = weekday_df.sort_values('sort_order').drop('sort_order', axis=1)
                                            
                                            # Format for display
                                            display_weekday = weekday_df.copy()
                                            display_weekday['win_rate'] = display_weekday['win_rate'].apply(lambda x: f"{x:.2f}%")
                                            display_weekday['avg_return'] = display_weekday['avg_return'].apply(lambda x: f"{x:.2f}%")
                                            if 'high_return_rate' in display_weekday.columns:
                                                display_weekday['high_return_rate'] = display_weekday['high_return_rate'].apply(lambda x: f"{x:.2f}%")
                                            display_weekday['performance_score'] = display_weekday['performance_score'].apply(lambda x: f"{x:.2f}")
                                            display_weekday['total_profit'] = display_weekday['total_profit'].apply(lambda x: f"{x:.2f}%")
                                            
                                            # Reorder columns
                                            columns_order = ['weekday', 'performance_score', 'win_rate', 'avg_return']
                                            if 'high_return_rate' in display_weekday.columns:
                                                columns_order.append('high_return_rate')
                                            columns_order.extend(['total_profit', 'trade_count'])
                                            
                                            # Create a styled dataframe with better column names
                                            st.dataframe(
                                                display_weekday[columns_order].rename(columns={
                                                    'weekday': 'Day of Week',
                                                    'performance_score': 'Performance Score',
                                                    'win_rate': 'Win Rate',
                                                    'avg_return': 'Avg Return',
                                                    'high_return_rate': '2X+ Return Rate',
                                                    'total_profit': 'Total Profit',
                                                    'trade_count': 'Trade Count'
                                                }),
                                                use_container_width=True
                                            )
                                            
                                            # Find best weekday
                                            best_weekday = weekday_df.loc[weekday_df['performance_score'].idxmax()]
                                            st.success(f"🌟 The best day of the week for trading is **{best_weekday['weekday']}** with a performance score of **{best_weekday['performance_score']:.2f}**")
                                            
                                            # Find worst weekday
                                            worst_weekday = weekday_df.loc[weekday_df['performance_score'].idxmin()]
                                            st.error(f"⚠️ The worst day of the week for trading is **{worst_weekday['weekday']}** with a performance score of **{worst_weekday['performance_score']:.2f}**")
                                        else:
                                            st.warning("Weekday information not available. Please ensure your timestamp data includes valid dates.")
                                
                                # Show detailed daily performance table
                                with st.expander("📊 View Detailed Daily Performance Table"):
                                    if not daily_df.empty:
                                        # Format daily data for display
                                        display_daily = daily_df.copy()
                                        display_daily['date'] = display_daily['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
                                        display_daily['win_rate'] = display_daily['win_rate'].apply(lambda x: f"{x:.2f}%")
                                        display_daily['avg_return'] = display_daily['avg_return'].apply(lambda x: f"{x:.2f}%")
                                        if 'high_return_rate' in display_daily.columns:
                                            display_daily['high_return_rate'] = display_daily['high_return_rate'].apply(lambda x: f"{x:.2f}%")
                                        display_daily['performance_score'] = display_daily['performance_score'].apply(lambda x: f"{x:.2f}")
                                        display_daily['total_profit'] = display_daily['total_profit'].apply(lambda x: f"{x:.2f}%")
                                        
                                        # Reorder columns
                                        columns_order = ['date', 'weekday', 'performance_score', 'win_rate', 'avg_return']
                                        if 'high_return_rate' in display_daily.columns:
                                            columns_order.append('high_return_rate')
                                        columns_order.extend(['total_profit', 'trade_count'])
                                        
                                        # Create dataframe
                                        st.dataframe(
                                            display_daily[columns_order].rename(columns={
                                                'date': 'Date',
                                                'weekday': 'Day of Week',
                                                'performance_score': 'Performance Score',
                                                'win_rate': 'Win Rate',
                                                'avg_return': 'Avg Return',
                                                'high_return_rate': '2X+ Return Rate',
                                                'total_profit': 'Total Profit',
                                                'trade_count': 'Trade Count'
                                            }),
                                            use_container_width=True
                                        )
                            else:
                                st.warning("Not enough daily data to perform analysis. Ensure your dataset has timestamps with multiple days of trading.")
                        else:
                            st.error("Daily analysis requires a 'Timestamp' column in your data. Please make sure your trading data includes this column.")
                        
                # Standard Profitable Filters Analysis
                st.subheader("Profitable Filters Analysis")
                
                # Guide for understanding filters
                with st.expander("📌 How to use filters for maximum profit (Click to expand)"):
                    st.markdown("""
                    ### Understanding Trading Filters
                    
                    This tool analyzes your trading data to identify specific factors that lead to more profitable trades. Here's how to interpret and use the results:
                    
                    **What is a filter?**  
                    A filter is a specific value in a column of your data that correlates with better trading results. For example, if trades with "Fish: 9 🐟" have a high win rate, this becomes a filter you could apply to future trades.
                    
                    **Single Column Filters:**  
                    The basic analysis identifies which individual values in your data columns correlate with profitable trades. Use these to quickly identify factors that tend to produce better results.
                    
                    **Combined Filters:**  
                    For more powerful analysis, you can activate the "Use combined filters" option below. This will analyze combinations of values across multiple columns to find the most profitable trading patterns. For example, it might discover that "Fish: 9 🐟" + "Snipers: Low" + "Age: New" produces the best combination of results.
                    
                    **How to use these insights:**
                    1. Look for filters with a good balance of win rate, average return, and sufficient trade count
                    2. Focus on filters that produce 10x+ returns consistently (high "10x+ Rate" value)
                    3. Avoid filters with very few trades, as these might not be statistically significant
                    4. Use the visualizations to understand how each filter affects your trading results
                    
                    The goal is to identify specific patterns or conditions that maximize your chances of hitting 10x+ returns on future trades.
                    """)
                
                st.write("Finding the best combinations of column values that lead to high profits...")
                
                # Allow user to select columns to analyze
                exclude_columns = [profit_column, 'numeric_profit', 'date', 'time', 'datetime', 'timestamp', 'win', 'high_return']
                analysis_columns = [col for col in df.columns if col not in exclude_columns and col in analyzed_data.columns]
                
                # Define the preferred columns for combination analysis
                preferred_columns = ['Made', 'Fish', 'Age', 'Liq SOL', 'Warnings', 'Scans', 'Hodls']
                
                # Find which preferred columns actually exist in the dataset
                available_preferred_columns = [col for col in preferred_columns if col in analysis_columns]
                
                # If none of the preferred columns exist, use the first 7 columns (or all if fewer than 7)
                if len(available_preferred_columns) == 0:
                    default_columns = analysis_columns[:min(7, len(analysis_columns))]
                else:
                    default_columns = available_preferred_columns
                
                if not analysis_columns:
                    st.warning("No suitable columns found for filter analysis. Please ensure your data contains categorical or numerical columns besides profit/date/time.")
                else:
                    st.write("### Range-Based Combination Filter Analysis")
                    st.write("This analysis finds optimal combinations of value ranges that lead to the highest profits.")
                    
                    with st.expander("📊 How Range-Based Filtering Works and Win Rate Definition (Click to expand)"):
                        st.markdown("""
                        ### Understanding Range-Based Filters
                        
                        This analyzer now uses **range-based filtering** for numeric columns and **symbol-based filtering** for categorical columns:
                        
                        **For Numeric Columns** (Made, Liq SOL, Scans, Hodls, Age, etc.):
                        - Instead of exact values like "Made: 10", the analyzer creates ranges like "Made: 2 to 10"
                        - The ranges automatically adjust based on your data's distribution
                        - Special formatting applies to certain columns (e.g., "Liq SOL: 10.0 SOL to 50.0 SOL")
                        
                        **For Symbol/Category Columns** (Fish, etc.):
                        - The analyzer identifies the specific symbols/values that correlate with the best profits
                        - These act as categorical filters in combination with numeric ranges
                        
                        **Benefits of Range-Based Analysis:**
                        - Finds broader patterns instead of overly specific combinations
                        - More trades match each filter, improving statistical significance
                        - Easier to apply findings to future trading decisions
                        - Better identifies the optimal value ranges for each variable
                        
                        The results will show you which combinations of ranges and values consistently produce the best returns.
                        
                        ### Win Rate Definition
                        
                        In this analyzer, we use a strict definition of winning trades:
                        
                        - **Winning trades**: Only trades with 2X returns or better (100%+ profit)
                        - **Losing trades**: Any trade below 2X return (including trades with 50%-99% profit)
                        
                        This strict definition helps identify filter combinations that maximize your chances of achieving significant profits (2X or greater) while minimizing exposure to trades with smaller returns or losses.
                        
                        The Win Rate shown is the percentage of trades that achieved 2X or better returns.
                        """)
                    
                    # Single Column Analysis section has been removed
                    
                    # Combination filter analysis section
                    st.subheader("🔍 Combination Filter Analysis")
                    st.write("Analyze combinations of multiple columns to find the most profitable patterns.")
                    
                    selected_columns = st.multiselect(
                        "Select columns to include in your filter combinations:",
                        options=analysis_columns,
                        default=default_columns
                    )
                    
                    min_trades = st.slider("Minimum number of trades for a filter to be considered", 
                                         min_value=20, max_value=200, value=50, 
                                         help="Higher values ensure more statistical significance")
                    
                    target_return = st.slider("Target minimum return multiplier (1x = 100% return)", 
                                            min_value=1.0, max_value=20.0, value=2.0, step=0.5,
                                            help="2x means filtering for 100% return or higher")
                    
                    # Set combined filters to always be used
                    use_combined_filters = True
                    
                    # Maximum column combinations
                    max_columns_to_combine = st.slider("Maximum columns to combine in filters", 
                                                     min_value=1, max_value=7, value=7,
                                                     help="Higher values find more specific patterns across multiple columns")
                    
                    if selected_columns:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("Starting filter analysis...")
                        
                        # Add a checkbox to limit analysis time 
                        limit_analysis = st.checkbox("Limit analysis time (faster results, fewer combinations)", value=True,
                                                help="Enable this to get faster results by analyzing only the most common values in each column")
                        
                        # Start the analysis
                        status_text.text("Analyzing filters... This may take a moment, especially with combined filters enabled.")
                        
                        # We'll use this function to update the progress bar
                        def update_progress(progress_pct, status_msg):
                            # Convert percentage (0-100) to a value between 0 and 1
                            progress_value = progress_pct / 100.0
                            progress_bar.progress(progress_value)
                            status_text.text(status_msg)
                            
                        profitable_filters = find_profitable_filters(
                            analyzed_data, 
                            selected_columns, 
                            working_profit_column, 
                            min_trades=min_trades,
                            target_return=target_return,
                            use_combined_filters=use_combined_filters,
                            max_columns_to_combine=max_columns_to_combine,
                            limit_analysis=limit_analysis,
                            progress_callback=update_progress
                        )
                        
                        # Analysis complete
                        progress_bar.progress(1.0)
                        status_text.text("Analysis complete!")
                        
                        if profitable_filters.empty:
                            st.warning(f"No profitable filters found with the current settings. Try reducing the minimum trades or target return requirements.")
                        else:
                            # Display the profitable filters table
                            st.subheader("Most Profitable Filters")
                            st.write(f"Filters that maximize profits with at least {min_trades} trades and targeting {target_return}x returns:")
                            
                            # Allow sorting by different metrics
                            sort_by = st.selectbox(
                                "Sort filters by:",
                                options=["Column Count", "Win Rate", "Avg Return", "Max Return", "2X+ Rate", "Trade Count"],
                                index=0  # Default to sorting by Column Count
                            )
                            
                            sort_mapping = {
                                "Column Count": "column_count",
                                "Win Rate": "win_rate",
                                "Avg Return": "avg_return",
                                "Max Return": "max_return",
                                "2X+ Rate": "high_return_rate",
                                "Trade Count": "trade_count"
                            }
                            
                            # Sort and display the top filters
                            sorted_filters = profitable_filters.sort_values(
                                by=sort_mapping[sort_by], 
                                ascending=False
                            ).reset_index(drop=True)
                            
                            # Format the table for display
                            display_filters = sorted_filters.copy()
                            display_filters["win_rate"] = display_filters["win_rate"].apply(lambda x: f"{x:.2f}%")
                            display_filters["avg_return"] = display_filters["avg_return"].apply(lambda x: f"{x:.2f}%")
                            display_filters["max_return"] = display_filters["max_return"].apply(lambda x: f"{x:.2f}%")
                            display_filters["high_return_rate"] = display_filters["high_return_rate"].apply(lambda x: f"{x:.2f}%")
                            
                            # Rename columns for better display
                            display_filters = display_filters.rename(columns={
                                "filter_name": "Filter",
                                "filter_value": "Value",
                                "column_count": "# Columns",
                                "trade_count": "Trades",
                                "win_rate": "Win Rate",
                                "avg_return": "Avg Return",
                                "max_return": "Max Return",
                                "high_return_rate": "2X+ Rate"
                            })
                            
                            # Show all rows in the dataframe without pagination
                            st.dataframe(
                                display_filters, 
                                use_container_width=True,
                                height=800,  # Make it taller to accommodate more rows
                                column_config={
                                    "Filter": st.column_config.TextColumn("Filter"),
                                    "Value": st.column_config.TextColumn("Value", width="large"),
                                    "# Columns": st.column_config.NumberColumn("# Columns", help="Number of columns used in this combined filter"),
                                    "Trades": st.column_config.NumberColumn("Trades"),
                                    "Win Rate": st.column_config.TextColumn("Win Rate"),
                                    "Avg Return": st.column_config.TextColumn("Avg Return"),
                                    "Max Return": st.column_config.TextColumn("Max Return"),
                                    "2X+ Rate": st.column_config.TextColumn("2X+ Rate", help="Percentage of trades that achieved 2X or better returns")
                                }
                            )
                            
                            # Function to convert dataframe to Excel file
                            def convert_df_to_excel(df):
                                # Create a BytesIO object
                                output = io.BytesIO()
                                # Create Excel writer using pandas
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    df.to_excel(writer, index=False, sheet_name='Profitable Filters')
                                    # Auto-adjust columns' width
                                    for column in df:
                                        column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
                                        col_idx = df.columns.get_loc(column)
                                        writer.sheets['Profitable Filters'].column_dimensions[chr(65 + col_idx)].width = column_width
                                
                                processed_data = output.getvalue()
                                return processed_data
                            
                            # Create Excel download button
                            excel_data = convert_df_to_excel(display_filters)
                            st.download_button(
                                label="📥 Download Filter Results as Excel",
                                data=excel_data,
                                file_name=f"trading_profitable_filters_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download the filter results as an Excel file"
                            )
                            
                            # Visualize filter effectiveness
                            st.subheader("Filter Effectiveness Visualization")
                            
                            # Select a filter to visualize in detail
                            if len(sorted_filters) > 0:
                                # Show all filter options, not just the top 10
                                filter_options = [f"{row['filter_name']}: {row['filter_value']}" for _, row in sorted_filters.iterrows()]
                                selected_filter = st.selectbox("Select a filter to visualize:", filter_options)
                                
                                if selected_filter:
                                    # Get the filter index from the options
                                    selected_idx = filter_options.index(selected_filter)
                                    selected_row = sorted_filters.iloc[selected_idx]
                                    
                                    filter_name = selected_row['filter_name']
                                    filter_value = selected_row['filter_value']
                                    
                                    # Handle combined filters differently
                                    if filter_name == "Combined":
                                        st.write("### Combined Filter Analysis")
                                        st.write(f"This filter combines multiple conditions: **{filter_value}**")
                                        
                                        # Process the combined filter
                                        filter_conditions = filter_value.split(" + ")
                                        
                                        # Create a mask for the combined filter
                                        combined_mask = pd.Series(True, index=analyzed_data.index)
                                        for condition in filter_conditions:
                                            parts = condition.split(": ", 1)
                                            if len(parts) == 2:
                                                col, val = parts
                                                
                                                # Check if this is a range condition
                                                if " to " in val and col.lower() in ['made', 'liq sol', 'scans', 'hodls', 'age'] or any(c.isdigit() for c in val):
                                                    try:
                                                        # This is likely a range condition
                                                        range_parts = val.split(" to ")
                                                        
                                                        # Handle different formats: "10 to 20", "10.0 SOL to 50.0 SOL", etc.
                                                        min_val = range_parts[0]
                                                        max_val = range_parts[1]
                                                        
                                                        # Clean up the values
                                                        min_val = min_val.split(" ")[0]  # Get first part before any space
                                                        
                                                        # For max_val, handle things like "50.0 SOL"
                                                        if " " in max_val:
                                                            max_val = max_val.split(" ")[0]
                                                            
                                                        # Convert to numeric
                                                        min_val = float(min_val)
                                                        max_val = float(max_val)
                                                        
                                                        # Apply range filter
                                                        numeric_col = pd.to_numeric(analyzed_data[col], errors='coerce')
                                                        combined_mask = combined_mask & (numeric_col >= min_val) & (numeric_col <= max_val)
                                                    except:
                                                        # If there's an error in parsing the range, try exact match
                                                        combined_mask = combined_mask & (analyzed_data[col] == val)
                                                else:
                                                    # This is an exact match condition
                                                    combined_mask = combined_mask & (analyzed_data[col] == val)
                                        
                                        # Calculate metrics for trades with and without the filter
                                        with_filter = analyzed_data[combined_mask].copy()
                                        without_filter = analyzed_data[~combined_mask].copy()
                                        
                                        # Show visualizations
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Create a custom box plot for combined filter
                                            import plotly.graph_objects as go
                                            fig = go.Figure()
                                            
                                            # Add box plot for trades with the filter
                                            fig.add_trace(go.Box(
                                                y=with_filter[working_profit_column],
                                                name=f'With Filter ({len(with_filter)} trades)',
                                                marker_color='green',
                                                boxmean=True
                                            ))
                                            
                                            # Add box plot for trades without the filter
                                            fig.add_trace(go.Box(
                                                y=without_filter[working_profit_column],
                                                name=f'Without Filter ({len(without_filter)} trades)',
                                                marker_color='red',
                                                boxmean=True
                                            ))
                                            
                                            # Update layout
                                            fig.update_layout(
                                                title=f"Return Distribution: Combined Filter",
                                                yaxis_title="Return (%)",
                                                showlegend=True
                                            )
                                            
                                            # Add horizontal line at 0% to mark profit/loss boundary
                                            fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
                                            
                                            # Add horizontal line at 100% to mark 2x return
                                            fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="green")
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                        with col2:
                                            # Create a bar chart showing win rate comparison
                                            data = []
                                            
                                            # Win rate with filter
                                            win_rate_with = with_filter['win'].mean() * 100
                                            # Win rate without filter
                                            win_rate_without = without_filter['win'].mean() * 100
                                            
                                            # 10x+ rate with filter
                                            high_return_with = with_filter['high_return'].mean() * 100
                                            # 10x+ rate without filter
                                            high_return_without = without_filter['high_return'].mean() * 100
                                            
                                            import plotly.graph_objects as go
                                            fig = go.Figure(data=[
                                                go.Bar(name='With Filter', 
                                                       x=['Win Rate', '2X+ Rate'], 
                                                       y=[win_rate_with, high_return_with],
                                                       marker_color='green'),
                                                go.Bar(name='Without Filter', 
                                                       x=['Win Rate', '2X+ Rate'], 
                                                       y=[win_rate_without, high_return_without],
                                                       marker_color='red')
                                            ])
                                            
                                            # Change the bar mode
                                            fig.update_layout(
                                                barmode='group',
                                                title="Performance Comparison",
                                                yaxis_title="Percentage (%)"
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                    else:
                                        # Regular single-column filter visualization
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            effectiveness_fig = plot_filter_effectiveness(
                                                analyzed_data,
                                                filter_name,
                                                filter_value,
                                                working_profit_column
                                            )
                                            st.plotly_chart(effectiveness_fig, use_container_width=True)
                                            
                                        with col2:
                                            win_rate_fig = plot_win_rate_by_filter(
                                                analyzed_data,
                                                filter_name,
                                                working_profit_column
                                            )
                                            st.plotly_chart(win_rate_fig, use_container_width=True)
                            
                            # Generate a report of recommendations
                            st.subheader("Recommended Trading Filters")
                            st.write("Based on the analysis, here are the top recommended filters to maximize your trading profits:")
                            
                            for i, (_, row) in enumerate(sorted_filters.head(5).iterrows()):
                                with st.container():
                                    st.markdown(f"**{i+1}. {row['filter_name']}: {row['filter_value']}**")
                                    st.markdown(f"- Trade Count: {row['trade_count']}")
                                    st.markdown(f"- Win Rate: {row['win_rate']:.2f}%")
                                    st.markdown(f"- Average Return: {row['avg_return']:.2f}%")
                                    st.markdown(f"- Maximum Return: {row['max_return']:.2f}%")
                                    st.markdown(f"- 2X+ Return Rate: {row['high_return_rate']:.2f}%")
                                    st.markdown("---")
                            
                            # Export options
                            st.subheader("Export Results")
                            
                            @st.cache_data
                            def convert_df_to_csv(df):
                                return df.to_csv(index=False).encode('utf-8')
                            
                            csv = convert_df_to_csv(sorted_filters)
                            st.download_button(
                                "Download Analysis as CSV",
                                csv,
                                "trading_filters_analysis.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            
                            # Export Excel format
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                sorted_filters.to_excel(writer, sheet_name='Profitable Filters', index=False)
                                summary_df = pd.DataFrame([summary_metrics])
                                summary_df.to_excel(writer, sheet_name='Summary Metrics', index=False)
                            
                            st.download_button(
                                "Download Analysis as Excel",
                                buffer.getvalue(),
                                "trading_filters_analysis.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key='download-excel'
                            )
    
    except Exception as e:
        st.error(f"Error analyzing the data: {str(e)}")
        st.exception(e)

else:
    # Show instructions when no file is uploaded
    st.info("Please upload an Excel file (.xlsx) containing your trading data to begin the analysis.")
    
    # Example of data structure that the tool expects
    st.subheader("Expected Data Format")
    st.markdown("""
    Your Excel file should contain columns with information about your trades. At minimum, you need:
    
    - A column showing profit/return/result for each trade
    - Additional columns with factors you want to analyze (e.g., entry time, strategy, market conditions)
    
    Example columns that work well with this tool:
    
    - **profit/return/result**: The profit or return percentage for each trade
    - **entry_time**: Time when you entered the trade
    - **symbol**: The trading symbol or instrument
    - **strategy**: Trading strategy used
    - **market_condition**: Bull market, bear market, etc.
    - **position_size**: Size of the position
    - **trade_duration**: How long the trade was active
    """)

# Footer
st.markdown("---")
st.markdown("Trading Data Analyzer Tool - Helping you find profitable trading patterns")
