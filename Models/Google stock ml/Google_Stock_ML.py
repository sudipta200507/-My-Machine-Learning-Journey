import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────
# 1. LOAD & FILTER TO RECENT 3 YEARS
# ─────────────────────────────────────────
df_full = pd.read_csv('GOOGLE_daily.csv', parse_dates=['Date'])
df_full = df_full.sort_values('Date').reset_index(drop=True)

# ✅ KEY FIX: Only use last 3 years so model learns current price regime
cutoff = df_full['Date'].max() - pd.DateOffset(years=3)
df = df_full[df_full['Date'] >= cutoff].reset_index(drop=True)
print(f"Using {len(df)} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────
def add_features(df):
    d = df[['Date','Open','High','Low','Close','Adj Close','Volume']].copy().reset_index(drop=True)

    # ✅ KEY FIX: Predict next-day RETURN, not raw price
    d['Return_1d']  = d['Close'].pct_change(1)
    d['Return_5d']  = d['Close'].pct_change(5)
    d['Return_10d'] = d['Close'].pct_change(10)
    d['Return_21d'] = d['Close'].pct_change(21)

    # Moving averages as RATIOS (scale-independent)
    for w in [5, 10, 20, 50, 100, 200]:
        d[f'MA_{w}']       = d['Close'].rolling(w).mean()
        d[f'MA_ratio_{w}'] = d['Close'] / (d[f'MA_{w}'] + 1e-9)  # ratio, not raw price

    # EMA
    d['EMA_12'] = d['Close'].ewm(span=12).mean()
    d['EMA_26'] = d['Close'].ewm(span=26).mean()
    d['EMA_ratio_12'] = d['Close'] / (d['EMA_12'] + 1e-9)
    d['EMA_ratio_26'] = d['Close'] / (d['EMA_26'] + 1e-9)

    # MACD (normalized)
    macd_raw        = d['EMA_12'] - d['EMA_26']
    d['MACD']        = macd_raw / (d['Close'] + 1e-9)
    d['MACD_signal'] = d['MACD'].ewm(span=9).mean()
    d['MACD_hist']   = d['MACD'] - d['MACD_signal']

    # RSI
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Bollinger Bands (position & width — both scale-independent)
    bb_mid           = d['Close'].rolling(20).mean()
    bb_std           = d['Close'].rolling(20).std()
    d['BB_upper']    = bb_mid + 2 * bb_std
    d['BB_lower']    = bb_mid - 2 * bb_std
    d['BB_width']    = (d['BB_upper'] - d['BB_lower']) / (bb_mid + 1e-9)
    d['BB_position'] = (d['Close'] - d['BB_lower']) / (d['BB_upper'] - d['BB_lower'] + 1e-9)

    # Volatility
    d['Volatility_5d']  = d['Return_1d'].rolling(5).std()
    d['Volatility_10d'] = d['Return_1d'].rolling(10).std()
    d['Volatility_21d'] = d['Return_1d'].rolling(21).std()

    # Volume (as ratio, not raw)
    d['Volume_MA_10']  = d['Volume'].rolling(10).mean()
    d['Volume_ratio']  = d['Volume'] / (d['Volume_MA_10'] + 1e-9)
    d['Volume_change'] = d['Volume'].pct_change()

    # Candle shape
    d['HL_range'] = (d['High'] - d['Low'])  / (d['Close'] + 1e-9)
    d['OC_range'] = (d['Close'] - d['Open']) / (d['Open'] + 1e-9)

    # Lagged RETURNS (not lagged raw prices)
    for lag in [1, 2, 3, 5, 10, 20]:
        d[f'Return_lag_{lag}'] = d['Return_1d'].shift(lag)

    # ✅ KEY FIX: Target = next day's return (not raw price)
    d['Target_Return'] = d['Close'].pct_change(1).shift(-1)
    # Keep actual next price for evaluation
    d['Target_Price']  = d['Close'].shift(-1)

    return d

df = add_features(df)
df = df.dropna().reset_index(drop=True)
print(f"After features: {len(df)} rows, price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

# ─────────────────────────────────────────
# 3. FEATURES & SPLIT
# ─────────────────────────────────────────
exclude = {'Date', 'Target_Return', 'Target_Price', 'Adj Close',
           'MA_5','MA_10','MA_20','MA_50','MA_100','MA_200',  # raw MAs excluded, ratios kept
           'EMA_12','EMA_26','BB_upper','BB_lower',           # raw bands excluded
           'Open','High','Low','Close','Volume'}              # raw prices excluded

feature_cols = [c for c in df.columns if c not in exclude]
print(f"Features used ({len(feature_cols)}): {feature_cols}")

X      = df[feature_cols].values
y_ret  = df['Target_Return'].values   # train on returns
y_price= df['Target_Price'].values    # evaluate on prices
close  = df['Close'].values
dates  = df['Date'].values

split = int(len(df) * 0.80)
X_train, X_test   = X[:split], X[split:]
y_train           = y_ret[:split]
y_test_ret        = y_ret[split:]
y_test_price      = y_price[split:]
close_test        = close[split:]
dates_test        = dates[split:]

print(f"Train: {split} | Test: {len(X_test)}")

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 4. TRAIN ON RETURNS
# ─────────────────────────────────────────
print("\nTraining Random Forest on returns...")
rf = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=5,
                           max_features=0.6, n_jobs=-1, random_state=42)
rf.fit(X_train_sc, y_train)

print("Training Gradient Boosting on returns...")
gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.03,
                                max_depth=4, subsample=0.8,
                                min_samples_leaf=8, random_state=42)
gb.fit(X_train_sc, y_train)

# Predict returns → convert to prices
rf_ret_pred  = rf.predict(X_test_sc)
gb_ret_pred  = gb.predict(X_test_sc)
ens_ret_pred = 0.45 * rf_ret_pred + 0.55 * gb_ret_pred

# ✅ Convert predicted return → predicted price
rf_price_pred  = close_test * (1 + rf_ret_pred)
gb_price_pred  = close_test * (1 + gb_ret_pred)
ens_price_pred = close_test * (1 + ens_ret_pred)

# ─────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────
def get_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return mae, rmse, r2, mape

print("\n── Test Set Performance (Price) ──")
for name, pred in [("Random Forest", rf_price_pred),
                   ("Gradient Boosting", gb_price_pred),
                   ("Ensemble", ens_price_pred)]:
    m = get_metrics(y_test_price, pred)
    print(f"  {name:20s} | MAE=${m[0]:.2f}  RMSE=${m[1]:.2f}  R²={m[2]:.4f}  MAPE={m[3]:.2f}%")

ens_m = get_metrics(y_test_price, ens_price_pred)

# ─────────────────────────────────────────
# 6. FUTURE 30-DAY FORECAST
# ─────────────────────────────────────────
def forecast_future(df_orig, feature_cols, scaler, rf, gb, n_days=30):
    raw_cols = ['Date','Open','High','Low','Close','Adj Close','Volume']

    # Use only recent 3 years for the rolling window
    cutoff = df_orig['Date'].max() - pd.DateOffset(years=3)
    last_raw = df_orig[df_orig['Date'] >= cutoff][raw_cols].copy().reset_index(drop=True)

    future_prices = []
    last_price = last_raw['Close'].iloc[-1]
    last_date  = pd.Timestamp(last_raw['Date'].iloc[-1])

    for i in range(n_days):
        feat_df = add_features(last_raw)
        feat_df = feat_df.dropna().reset_index(drop=True)
        if len(feat_df) == 0:
            break

        row    = feat_df.iloc[-1][feature_cols].values.reshape(1, -1)
        row_sc = scaler.transform(row)

        pred_ret = 0.45 * rf.predict(row_sc)[0] + 0.55 * gb.predict(row_sc)[0]
        pred_price = last_price * (1 + pred_ret)
        future_prices.append(pred_price)

        # Advance date (skip weekends)
        new_date = last_date + pd.Timedelta(days=1)
        while new_date.weekday() >= 5:
            new_date += pd.Timedelta(days=1)
        last_date  = new_date
        last_price = pred_price

        avg_vol = last_raw['Volume'].tail(10).mean()
        new_row = pd.DataFrame([{
            'Date': new_date, 'Open': pred_price,
            'High': pred_price * 1.005, 'Low': pred_price * 0.995,
            'Close': pred_price, 'Adj Close': pred_price, 'Volume': avg_vol
        }])
        last_raw = pd.concat([last_raw, new_row], ignore_index=True)

    future_dates = pd.bdate_range(
        pd.Timestamp(df_orig['Date'].iloc[-1]) + pd.Timedelta(days=1),
        periods=len(future_prices))
    return future_dates, future_prices

df_full_raw = pd.read_csv('GOOGLE_daily.csv', parse_dates=['Date'])
df_full_raw = df_full_raw.sort_values('Date').reset_index(drop=True)

print("\nGenerating 30-day forecast...")
future_dates, future_prices = forecast_future(df_full_raw, feature_cols, scaler, rf, gb, n_days=30)
print(f"Last known price : ${df_full_raw['Close'].iloc[-1]:.2f}")
print(f"Forecast range   : ${min(future_prices):.2f} - ${max(future_prices):.2f}")

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': [round(p, 2) for p in future_prices]
})
print(forecast_df.to_string(index=False))

# ─────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=feature_cols)
top_features = importances.nlargest(15)
residuals = y_test_price - ens_price_pred

plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 22), facecolor='#0d0d0d')
gs  = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

ACCENT='#00d4ff'; GREEN='#00ff88'; ORANGE='#ff8c00'
PURPLE='#b266ff'; GRID_CLR='#222222'; PANEL='#111111'

# Panel 1: Full 3yr history + test preds
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(PANEL)
ax1.plot(df['Date'], df['Close'], color='#444', lw=0.8, label='Training Data', alpha=0.6)
ax1.plot(pd.to_datetime(dates_test), y_test_price,   color=ACCENT, lw=1.5, label='Actual (Test)')
ax1.plot(pd.to_datetime(dates_test), ens_price_pred, color=GREEN,  lw=1.5, label='Predicted (Ensemble)', linestyle='--')
ax1.fill_between(pd.to_datetime(dates_test), y_test_price, ens_price_pred, alpha=0.12, color=GREEN)
ax1.set_title('Google Stock — Actual vs Predicted (Test Set) | Return-Based Model', color='white', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', color='#aaa'); ax1.set_ylabel('Price (USD)', color='#aaa')
ax1.tick_params(colors='#aaa'); ax1.grid(color=GRID_CLR, lw=0.5)
for s in ax1.spines.values(): s.set_color('#333')
ax1.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='white', fontsize=10)

# Panel 2: All models on test
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(PANEL)
ax2.plot(pd.to_datetime(dates_test), y_test_price,   color=ACCENT,  lw=1.8, label='Actual')
ax2.plot(pd.to_datetime(dates_test), ens_price_pred, color=GREEN,   lw=1.8, label='Ensemble', linestyle='--')
ax2.plot(pd.to_datetime(dates_test), rf_price_pred,  color=ORANGE,  lw=1.0, label='RF Only',  linestyle=':',  alpha=0.7)
ax2.plot(pd.to_datetime(dates_test), gb_price_pred,  color=PURPLE,  lw=1.0, label='GB Only',  linestyle='-.', alpha=0.7)
ax2.set_title('Test Period — All Models', color='white', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', color='#aaa'); ax2.set_ylabel('Price (USD)', color='#aaa')
ax2.tick_params(colors='#aaa'); ax2.grid(color=GRID_CLR, lw=0.5)
for s in ax2.spines.values(): s.set_color('#333')
ax2.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='white', fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

# Panel 3: 30-Day Forecast
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(PANEL)
lb = 60
ax3.plot(df_full_raw['Date'].iloc[-lb:], df_full_raw['Close'].iloc[-lb:], color=ACCENT, lw=1.8, label='Recent History')
ax3.plot(future_dates, future_prices, color=GREEN, lw=2, linestyle='--', label='30-Day Forecast')
ax3.fill_between(future_dates,
                 [p * 0.97 for p in future_prices],
                 [p * 1.03 for p in future_prices],
                 alpha=0.2, color=GREEN, label='±3% Band')
ax3.axvline(x=future_dates[0], color='#555', linestyle=':', lw=1)
ax3.set_title('30-Day Future Forecast (Return-Based)', color='white', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date', color='#aaa'); ax3.set_ylabel('Price (USD)', color='#aaa')
ax3.tick_params(colors='#aaa'); ax3.grid(color=GRID_CLR, lw=0.5)
for s in ax3.spines.values(): s.set_color('#333')
ax3.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='white', fontsize=9)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')

# Panel 4: Feature Importance
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor(PANEL)
clrs = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_features)))
ax4.barh(range(len(top_features)), top_features.values[::-1], color=clrs[::-1])
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features.index[::-1], color='#ddd', fontsize=9)
ax4.set_title('Top 15 Feature Importances (RF)', color='white', fontsize=12, fontweight='bold')
ax4.set_xlabel('Importance', color='#aaa')
ax4.tick_params(colors='#aaa'); ax4.grid(color=GRID_CLR, lw=0.5, axis='x')
for s in ax4.spines.values(): s.set_color('#333')

# Panel 5: Residuals
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor(PANEL)
ax5.hist(residuals, bins=50, color=ACCENT, alpha=0.7, edgecolor='#0d0d0d')
ax5.axvline(0, color=GREEN, lw=2, linestyle='--', label='Zero error')
ax5.axvline(residuals.mean(), color=ORANGE, lw=1.5, linestyle=':', label=f'Mean={residuals.mean():.2f}')
ax5.set_title('Residuals Distribution (Ensemble)', color='white', fontsize=12, fontweight='bold')
ax5.set_xlabel('Prediction Error (USD)', color='#aaa'); ax5.set_ylabel('Frequency', color='#aaa')
ax5.tick_params(colors='#aaa'); ax5.grid(color=GRID_CLR, lw=0.5)
for s in ax5.spines.values(): s.set_color('#333')
ax5.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='white', fontsize=9)

metrics_txt = (
    f"── Ensemble Model Metrics ──\n"
    f"  MAE  : ${ens_m[0]:.2f}\n"
    f"  RMSE : ${ens_m[1]:.2f}\n"
    f"  R²   : {ens_m[2]:.4f}\n"
    f"  MAPE : {ens_m[3]:.2f}%\n\n"
    f"── 30-Day Forecast ──\n"
    f"  Last price : ${df_full_raw['Close'].iloc[-1]:.2f}\n"
    f"  Day 1      : ${future_prices[0]:.2f}\n"
    f"  Day 30     : ${future_prices[-1]:.2f}\n"
    f"  Δ          : {((future_prices[-1]/future_prices[0])-1)*100:+.2f}%"
)
fig.text(0.01, 0.01, metrics_txt, fontsize=9.5, color='#cccccc', fontfamily='monospace',
         bbox=dict(facecolor='#1a1a1a', edgecolor='#444', boxstyle='round,pad=0.6'))

fig.suptitle('Google (GOOG) Stock Price Prediction — Return-Based Model (Last 3 Years)\nRandom Forest + Gradient Boosting Ensemble',
             color='white', fontsize=15, fontweight='bold', y=0.998)

plt.savefig('google_stock_prediction.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
forecast_df.to_csv('google_30day_forecast.csv', index=False)
print("\nSaved: google_stock_prediction.png & google_30day_forecast.csv")