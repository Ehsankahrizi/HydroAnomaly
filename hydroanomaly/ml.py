import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt

# ============= Helper Function: Match nearest USGS turbidity by datetime ================================
def match_nearest_datetime(sentinel_dt, usgs):
    # usgs is indexed by 'datetime'
    if usgs.empty:
        return np.nan
    # Find the closest datetime in usgs to sentinel_dt
    i = usgs.index.get_indexer([sentinel_dt], method='nearest')[0]
    return usgs.iloc[i]['turbidity']

# ============= Preprocessing and Feature Engineering ====================================================
def preprocess_data(sentinel, usgs):
    # Expect both to have DatetimeIndex named 'datetime'
    if sentinel.index.name != 'datetime':
        raise ValueError("Sentinel dataframe must have DatetimeIndex named 'datetime'")
    if usgs.index.name != 'datetime':
        raise ValueError("USGS dataframe must have DatetimeIndex named 'datetime'")

    # Add matched turbidity to sentinel (by nearest datetime)
    sentinel = sentinel.copy()
    usgs = usgs.copy()
    sentinel['turbidity'] = [match_nearest_datetime(dt, usgs) for dt in sentinel.index]
    df = sentinel.dropna(subset=['turbidity'])

    # Water pixel filtering
    if 'SCL' in df.columns and (df['SCL'] == 6).sum() > 0:
        df = df[df['SCL'] == 6].drop_duplicates(subset=['B2', 'B3', 'B4'])

    # Feature engineering
    bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'])
    df['NDSI'] = (df['B3'] - df['B11']) / (df['B3'] + df['B11'])

    df = df.sort_index()
    df['turbidity_diff1'] = df['turbidity'].diff()
    df['turbidity_diff2'] = df['turbidity_diff1'].diff()
    thresh = 2 * df['turbidity_diff2'].std()
    df['spike'] = (df['turbidity_diff2'].abs() > thresh).astype(int)
    df = df.dropna()

    # Class label
    df['Classe'] = (df['turbidity'] > 20).astype(int)
    return df, bands

# ============= Anomaly Detection: One-Class SVM ========================================================
def run_oneclass_svm(sentinel, usgs, plot=True):
    """
    Apply One-Class SVM anomaly detection on Sentinel/USGS data.
    Inputs must have DatetimeIndex named 'datetime'.
    """
    df, bands = preprocess_data(sentinel, usgs)
    features = bands + ['NDVI','NDWI','NDSI','turbidity_diff1','turbidity_diff2','spike']
    X = df[features].fillna(df[features].mean()).values
    y = df['Classe'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_class0 = X_scaled[y == 0]
    X_class1 = X_scaled[y == 1]

    train_size = max(1, int(0.8 * len(X_class0)))
    X_train = X_class0[:train_size]
    X_test = np.vstack([X_class0[train_size:], X_class1])
    y_test = np.array([0]*(len(X_class0)-train_size) + [1]*len(X_class1))

    best_f1 = -1
    best_model, best_y_pred, best_params = None, None, None
    for gamma in ['auto', 'scale']:
        for nu in [0.01, 0.05, 0.1, 0.2]:
            model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
            model.fit(X_train)
            y_pred = np.where(model.predict(X_test) == 1, 0, 1)
            if len(np.unique(y_pred)) > 1:
                f1 = f1_score(y_test, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_y_pred = y_pred
                    best_params = {'gamma': gamma, 'nu': nu}

    if best_f1 > -1:
        df_out = df.iloc[-len(y_test):].copy()
        df_out['predicted'] = best_y_pred
        if plot:
            plt.figure(figsize=(15,6))
            plt.plot(df_out.index, df_out['turbidity'], label='turbidity', color='blue')
            plt.scatter(df_out[df_out['Classe']==1].index, df_out[df_out['Classe']==1]['turbidity'],
                        color='red', marker='x', label='True Anomaly', s=100)
            plt.scatter(df_out[df_out['predicted']==1].index, df_out[df_out['predicted']==1]['turbidity'],
                        edgecolors='orange', facecolors='none', marker='o', label='Predicted Anomaly', s=80)
            plt.title("True vs Predicted Anomalies (OneClassSVM)")
            plt.xlabel("Datetime")
            plt.ylabel("turbidity")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        return df_out, best_params, best_f1
    else:
        print("Could not find a good model. Try different hyperparameters.")
        return None, None, None

# ============= Anomaly Detection: Isolation Forest ======================================================
def run_isolation_forest(sentinel, usgs, plot=True):
    """
    Apply Isolation Forest anomaly detection on Sentinel/USGS data.
    Inputs must have DatetimeIndex named 'datetime'.
    """
    df, bands = preprocess_data(sentinel, usgs)
    features = bands + ['NDVI','NDWI','NDSI','turbidity_diff1','turbidity_diff2','spike']
    X = df[features].fillna(df[features].mean()).values
    y = df['Classe'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_class0 = X_scaled[y == 0]
    X_class1 = X_scaled[y == 1]

    train_size = max(1, int(0.8 * len(X_class0)))
    X_train = X_class0[:train_size]
    X_test = np.vstack([X_class0[train_size:], X_class1])
    y_test = np.array([0]*(len(X_class0)-train_size) + [1]*len(X_class1))

    best_f1 = -1
    best_model, best_y_pred, best_params = None, None, None
    for contamination in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            max_samples='auto',
            bootstrap=True,
            random_state=42
        )
        model.fit(X_train)
        y_pred = np.where(model.predict(X_test) == 1, 0, 1)
        if len(np.unique(y_pred)) > 1:
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_y_pred = y_pred
                best_params = {'contamination': contamination}

    if best_f1 > -1:
        df_out = df.iloc[-len(y_test):].copy()
        df_out['predicted'] = best_y_pred
        if plot:
            plt.figure(figsize=(15,6))
            plt.plot(df_out.index, df_out['turbidity'], label='turbidity', color='blue')
            plt.scatter(df_out[df_out['Classe']==1].index, df_out[df_out['Classe']==1]['turbidity'],
                        color='red', marker='x', label='True Anomaly', s=100)
            plt.scatter(df_out[df_out['predicted']==1].index, df_out[df_out['predicted']==1]['turbidity'],
                        edgecolors='orange', facecolors='none', marker='o', label='Predicted Anomaly', s=80)
            plt.title("True vs Predicted Anomalies (Isolation Forest)")
            plt.xlabel("Datetime")
            plt.ylabel("turbidity")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        return df_out, best_params, best_f1
    else:
        print("Could not find a good model. Try different hyperparameters.")
        return None, None, None
