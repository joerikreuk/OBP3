import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline   

def load_data():
    data = pd.read_csv('actuals.csv', header=None, names=['day', 'volume'], skiprows=1)
    return data

def prepare_features(data):
    # Create time features
    data['day_num'] = np.arange(len(data)) + 1 
    data['week_num'] = (data['day_num'] - 1) // 7 + 1  
    data['day_of_week'] = (data['day_num'] - 1) % 7  
    data['week_of_year'] = (data['week_num'] - 1) % 52 + 1  
    data['year'] = (data['week_num'] - 1) // 52 + 1  
    return data

def add_transformed_features(df):
    # Avoid log of 0 or negative
    safe_df = df.copy()
    for col in ['volume_roll7', 'volume_roll14', 'volume_roll30', 'lag_1', 'lag_7', 'lag_14', 'lag_30']:
        safe_df[f'log_{col}'] = np.log(safe_df[col].clip(lower=1e-3))
        safe_df[f'exp_{col}'] = np.exp(safe_df[col].clip(upper=10))  # avoid too large numbers
    return safe_df


def calculate_wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def build_model(data, test_weeks=10):
    df = data.copy()

    # Add rolling statistics
    df['volume_roll7'] = df['volume'].shift(1).rolling(7).mean()
    df['volume_roll14'] = df['volume'].shift(1).rolling(14).mean()
    df['volume_roll30'] = df['volume'].shift(1).rolling(30).mean()
    df['lag_1'] = df['volume'].shift(1)
    df['lag_7'] = df['volume'].shift(7)
    df['lag_14'] = df['volume'].shift(14)
    df['lag_30'] = df['volume'].shift(30)
    df.dropna(inplace=True)

    df = add_transformed_features(df)

    # Features
    categorical_features = ['day_of_week', 'week_of_year']
    numeric_features = [
        'day_num',
        'volume_roll7', 'volume_roll14', 'volume_roll30',
        'lag_1', 'lag_7', 'lag_14', 'lag_30',
        'log_volume_roll7', 'log_volume_roll14', 'log_volume_roll30',
        'log_lag_1', 'log_lag_7', 'log_lag_14', 'log_lag_30',
        'exp_volume_roll7', 'exp_volume_roll14', 'exp_volume_roll30',
        'exp_lag_1', 'exp_lag_7', 'exp_lag_14', 'exp_lag_30'
    ]

    # Column transformer with imputers
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ]), numeric_features)
    ])

    # Train-test split
    test_cutoff = df['day_num'].max() - test_weeks * 7
    train_mask = df['day_num'] <= test_cutoff
    test_mask = df['day_num'] > test_cutoff

    X_train = df.loc[train_mask, categorical_features + numeric_features]
    X_test = df.loc[test_mask, categorical_features + numeric_features]
    y_train = df.loc[train_mask, 'volume']
    y_test = df.loc[test_mask, 'volume']

    # Linear regression model
    model = make_pipeline(
        preprocessor,
        LinearRegression()
    )

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_wape = calculate_wape(y_train, y_pred_train)
    test_wape = calculate_wape(y_test, y_pred_test)

    return model, train_wape, test_wape, X_test, y_test, y_pred_test



def forecast_week_260(model, data):
    # Week 260 starts at day 1813
    start_day = 1813
    forecast_days = pd.DataFrame({'day_num': range(start_day, start_day + 7)})
    forecast_days['day_of_week'] = (forecast_days['day_num'] - 1) % 7
    forecast_days['week_of_year'] = 51  # Assume week 260 maps to week 51 of the year

    # Prepare full data including forecast placeholder
    recent_data = data[['day_num', 'volume']].copy()
    extended_data = pd.concat([
        recent_data,
        pd.DataFrame({'day_num': forecast_days['day_num'], 'volume': np.nan})
    ]).sort_values('day_num', ignore_index=True)

    # Compute rolling means and lag features
    extended_data['volume_roll7'] = extended_data['volume'].shift(1).rolling(7).mean()
    extended_data['volume_roll14'] = extended_data['volume'].shift(1).rolling(14).mean()
    extended_data['volume_roll30'] = extended_data['volume'].shift(1).rolling(30).mean()
    extended_data['lag_1'] = extended_data['volume'].shift(1)
    extended_data['lag_7'] = extended_data['volume'].shift(7)
    extended_data['lag_14'] = extended_data['volume'].shift(14)
    extended_data['lag_30'] = extended_data['volume'].shift(30)

    # Add log and exp transformations
    forecast_features = add_transformed_features(extended_data)
    forecast_days = forecast_days.merge(
        forecast_features[[
            'day_num',
            'volume_roll7', 'volume_roll14', 'volume_roll30',
            'lag_1', 'lag_7', 'lag_14', 'lag_30',
            'log_volume_roll7', 'log_volume_roll14', 'log_volume_roll30',
            'log_lag_1', 'log_lag_7', 'log_lag_14', 'log_lag_30',
            'exp_volume_roll7', 'exp_volume_roll14', 'exp_volume_roll30',
            'exp_lag_1', 'exp_lag_7', 'exp_lag_14', 'exp_lag_30'
        ]],
        on='day_num',
        how='left'
    )


    # Reorder columns for prediction
    X_forecast = forecast_days[
        ['day_of_week', 'week_of_year', 'day_num',
         'volume_roll7', 'volume_roll14', 'volume_roll30',
         'lag_1', 'lag_7', 'lag_14', 'lag_30',
         'log_volume_roll7', 'log_volume_roll14', 'log_volume_roll30',
         'log_lag_1', 'log_lag_7', 'log_lag_14', 'log_lag_30',
         'exp_volume_roll7', 'exp_volume_roll14', 'exp_volume_roll30',
         'exp_lag_1', 'exp_lag_7', 'exp_lag_14', 'exp_lag_30']
    ]


    # Predict
    forecast = model.predict(X_forecast)

    return forecast_days['day_num'], forecast


def main():
    # Load and prepare data
    data = load_data()
    data = prepare_features(data)
    
    # Build final model
    model, train_wape, test_wape, X_test, y_test, y_pred_test = build_model(data)
    print(f"Final Model Performance:")
    print(f"Train WAPE: {train_wape:.2f}%")
    print(f"Test WAPE: {test_wape:.2f}%")
    
    # Forecast week 260
    days, forecast = forecast_week_260(model, data)
    print("\nForecast for Week 260 (days 1813-1819):")
    for day, vol in zip(days, forecast):
        print(f"Day {day}: {vol:.1f} calls")
    
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines

    # Plotting: Broken x-axis to show last 100 days and forecast separately
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)

    cutoff_day = data['day_num'].max() - 100
    recent_data = data[data['day_num'] > cutoff_day]
    recent_day_nums = recent_data['day_num']
    recent_volumes = recent_data['volume']

    test_day_nums = data.loc[X_test.index, 'day_num']

    # --- Left subplot (recent data) ---
    ax1 = plt.subplot(gs[0])
    line_actual, = ax1.plot(recent_day_nums, recent_volumes, label='Actual Volume', linewidth=2)
    line_pred, = ax1.plot(test_day_nums, y_pred_test, label='Test Set Predictions', color='orange', linewidth=2)
    ax1.set_xlim(cutoff_day, data['day_num'].max())
    ax1.set_ylabel('Call Volume')
    ax1.set_xlabel('Day Number')
    ax1.set_title('Call Volume Forecasting')
    ax1.grid(True)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelright=False)

    # --- Right subplot (forecast) ---
    ax2 = plt.subplot(gs[1], sharey=ax1)
    scatter_forecast = ax2.scatter(days, forecast, color='red', label='Week 260 Forecast', zorder=5)
    ax2.set_xlim(1812.5, 1819.5)
    ax2.set_xlabel('Day Number')
    ax2.grid(True)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(labelleft=False)

    # Diagonal break marks
    d = .005
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Manually construct legend (including red dot from ax2)
    forecast_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='Week 260 Forecast')
    ax1.legend(
        handles=[line_actual, line_pred, forecast_dot],
        loc='upper left', 
        fontsize=10, 
        frameon=True
    )

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()