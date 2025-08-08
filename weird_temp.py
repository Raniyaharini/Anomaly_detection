import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # switch to a GUI backend for external windows
import matplotlib.pyplot as plt
import joblib

# Speed & performance
import tensorflow as tf
from tensorflow.keras import mixed_precision
# Enable mixed precision for faster GPU training
mixed_precision.set_global_policy('mixed_float16')

# Machine learning libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIGURATION ===
TRAIN_H5_PATH   = '/Users/raniyaharinirajendran/Documents/Factor4Solutions/Machine_learning/Project_AI/h5_file/KRGC_2024_aligned.h5'
INFER_H5_PATH   = '/Users/raniyaharinirajendran/Documents/Factor4Solutions/Machine_learning/Project_AI/h5_file/KRGC_2024_aligned.h5'
SCALER_PATH     = '/Users/raniyaharinirajendran/Documents/Factor4Solutions/Machine_learning/Project_AI/Results1/scaler.gz'
ISO_PATH        = '/Users/raniyaharinirajendran/Documents/Factor4Solutions/Machine_learning/Project_AI/Results1/iso_model.gz'
AE_PATH         = '/Users/raniyaharinirajendran/Documents/Factor4Solutions/Machine_learning/Project_AI/Results1/gru_autoencoder.h5'
THRESH_PATH     = '/Users/raniyaharinirajendran/Documents/Factor4Solutions/Machine_learning/Project_AI/Results1/threshold.npy'

# === FUNCTIONS ===

def train_models(train_h5_path=TRAIN_H5_PATH):
    # 1. Load and index reference data
    with h5py.File(train_h5_path, 'r') as f:
        dt = f['datetime'][:].astype('S19').astype(str)
        t_ref = f['t_71Xx_r_S'][:]
    df = pd.DataFrame({'t_ref': t_ref}, index=pd.to_datetime(dt))

    # 2. Feature engineering
    df['rolling_mean'] = df['t_ref'].rolling(12).mean()
    df['rolling_std']  = df['t_ref'].rolling(12).std()
    df['hour']         = df.index.hour
    df['dayofweek']    = df.index.dayofweek
    df.dropna(inplace=True)

    # 3. Prepare features
    features = ['t_ref','rolling_mean','rolling_std','hour','dayofweek']
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4A. Train IsolationForest
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_scaled)

    # 4B. Build tf.data pipeline for GRU autoencoder
    seq_len = 50
    batch_size = 512
    shift = seq_len

    ds = tf.data.Dataset.from_tensor_slices(X_scaled)
    ds = ds.window(seq_len, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(seq_len))
    ds = ds.map(lambda seq: (seq, seq), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(10000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    total = sum(1 for _ in ds)
    val_batches = max(1, int(0.1 * total))
    train_ds = ds.skip(val_batches).repeat()
    val_ds   = ds.take(val_batches).repeat()

    # 5. Define model
    model = Sequential([
        tf.keras.Input(shape=(seq_len, X_scaled.shape[1])),
        GRU(32, return_sequences=False),
        RepeatVector(seq_len),
        GRU(32, return_sequences=True),
        TimeDistributed(Dense(X_scaled.shape[1]))
    ])
    model.compile(optimizer='adam', loss='mse')

    # 6. Train
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        train_ds,
        epochs=50,
        steps_per_epoch=500,
        validation_data=val_ds,
        validation_steps=20,
        callbacks=[early]
    )

    # 7. Compute threshold
    def compute_mse(dataset, steps):
        mses = []
        it = iter(dataset)
        for _ in range(steps):
            seq, _ = next(it)
            pred = model.predict(seq)
            mses.extend(np.mean((seq - pred)**2, axis=(1,2)))
        return np.array(mses)

    val_mse = compute_mse(val_ds, 20)
    threshold = np.percentile(val_mse, 99)

    # 8. Save artifacts
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(iso, ISO_PATH)
    model.save(AE_PATH)
    np.save(THRESH_PATH, threshold)

    print("Saved:")
    print(" - Scaler ->", SCALER_PATH)
    print(" - ISO model ->", ISO_PATH)
    print(" - AE model ->", AE_PATH)
    print(" - Threshold ->", THRESH_PATH)


def infer_anomalies(infer_h5_path=INFER_H5_PATH):
    # Load new sensor data from separate H5 file
    with h5py.File(infer_h5_path, 'r') as f:
        dates = f['datetime'][:].astype('S19').astype(str)
        vals  = f['t_71Xx_S'][:]
    df = pd.DataFrame({'t_val': vals}, index=pd.to_datetime(dates))

    # Feature engineering
    df['rolling_mean'] = df['t_val'].rolling(12).mean()
    df['rolling_std']  = df['t_val'].rolling(12).std()
    df['hour']         = df.index.hour
    df['dayofweek']    = df.index.dayofweek
    df.dropna(inplace=True)

    # Scale
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    X = scaler.transform(df[['t_val','rolling_mean','rolling_std','hour','dayofweek']].values)

    # IsolationForest
    iso: IsolationForest = joblib.load(ISO_PATH)
    df['iso_score']   = iso.decision_function(X)
    df['iso_anomaly'] = iso.predict(X) == -1

    # Autoencoder
    ae = load_model(AE_PATH, compile=False)  # load without compiling to avoid legacy loss function issues
    seq_len = ae.input_shape[1]
    seqs = []
    idxs = []
    for i in range(0, len(X) - seq_len + 1, seq_len):
        seqs.append(X[i:i+seq_len])
        idxs.append((i, i+seq_len))
    seqs = np.stack(seqs)

    recons = ae.predict(seqs)
    mses = np.mean((seqs - recons)**2, axis=(1,2))
    threshold = np.load(THRESH_PATH)
    flags = mses > threshold
    pt_flags = np.zeros(len(X), bool)
    for (s,e), f in zip(idxs, flags):
        pt_flags[s:e] = f

    df['ae_mse']      = np.nan
    for (s,e), m in zip(idxs, mses):
        df.iloc[e-1, df.columns.get_loc('ae_mse')] = m
    df['ae_anomaly'] = pt_flags

    # Plot results
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True)
    ax1.plot(df.index, df['t_val'], label='Sensor')
    ax1.scatter(df.index[df['iso_anomaly']], df['t_val'][df['iso_anomaly']],
                color='red', s=5, label='ISO Anomaly')
    ax1.legend(); ax1.set_title('Isolation Forest')

    ax2.plot(df.index, df['t_val'], label='Sensor')
    ax2.scatter(df.index[df['ae_anomaly']], df['t_val'][df['ae_anomaly']],
                color='orange', s=5, label='AE Anomaly')
    ax2.legend(); ax2.set_title('Autoencoder')

    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    train_models()
    infer_anomalies()
