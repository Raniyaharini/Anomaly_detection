# 🌡️ Ambient Temperature Anomaly Detection 

This project explores the use of AI models to detect irregular or unexpected behavior in ambient temperature sensor data from industrial installations. The motivation is to improve operational efficiency by identifying measurement anomalies and environmental interferences (e.g., direct sunlight, sensor misplacement) that can impact downstream system optimization.

---

## 📌 Objective

The goal is to detect:

- **Measurement anomalies**
- **External influences** such as sunlight, heat sources, or faulty sensor placement

Since ambient temperature directly impacts system cooling load and optimal operation predictions, detecting inaccurate readings is critical. The focus is on identifying:

- **Isolated anomalies**: Short-lived deviations due to local effects
- **Recurring anomalies**: Repeated patterns suggesting persistent issues

Example: At several sites (e.g., Karlsruhe, Rossenheim, Berlin Zentrum), morning/afternoon spikes followed by quick returns to baseline were observed — likely due to direct sunlight on the sensor.

The detection system aims to identify when anomalies **start**, **end**, and whether one is **ongoing**.

---

## 🧪 Methodology

### 1. Data Sources

- **Training**: DWD (German Weather Service) data – ambient temperature (2020–2024)
- **Validation**: Sensor data from KRGC installation site

### 2. Feature Engineering

- Rolling mean (12-sample window)
- Rolling standard deviation
- Hour of day
- Day of week

These features provide both statistical and temporal context.

### 3. Models Used

#### Isolation Forest
- Tree-based anomaly detector
- Trained on scaled DWD features

#### GRU Autoencoder
- Reconstructs input sequences; reconstruction error used to flag anomalies
- Built using TensorFlow with GPU acceleration

---

## ⚙️ Training Details

- **Sequence length**: 50 time steps  
- **Batch size**: 512  
- **Training/Validation split**: 90% / 10%  
- **Thresholding**: Anomalies flagged based on 90th percentile of MSE on validation data

---

## 📊 Model Evaluation

### 1. Trained on DWD 2024 data

- **Isolation Forest** detected extreme spikes/dips but missed moderate ones
- **GRU Autoencoder** minimized false positives but failed to detect steep short anomalies

### 2. Trained on DWD 2020–2024 data

- Isolation Forest maintained low false positives
- Still under-sensitive to subtle or moderate anomalies

### 3. Adding Humidity

- Combined temperature + humidity training
- No significant change in model performance

### 4. Trained on Site-Specific Sensor Data

#### Without "Hanging Signal"
- Detected short spikes and midday anomalies
- Missed long flatline (e.g., 0°C constant reading for weeks)

#### With "Hanging Signal" Logic
- Added rule-based detection for prolonged constant values
- Detected sensor faults, but also introduced false positives

---

## ✅ Outcomes

- Successfully explored multiple anomaly detection strategies
- Detected both extreme and subtle short-term anomalies
- Rule-based extension captured long flat signals

---


## 🧭 Conclusion & Next Steps

This project provides a solid baseline for future AI-assisted anomaly detection in ambient sensor data. Further improvements may include:

- Improved sensitivity tuning
- Context-aware filtering (e.g., distinguishing solar effects from faults)
- Multimodal learning (incorporating more sensor types)

---

## 🔑 Keywords

`Anomaly Detection` • `Ambient Temperature` • `GRU Autoencoder` • `Isolation Forest` • `Industrial AI` • `Sensor Faults`
