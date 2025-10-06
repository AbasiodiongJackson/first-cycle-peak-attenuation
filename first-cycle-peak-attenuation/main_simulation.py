import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.signal import StateSpace, lsim

# === 1. Load ML models ===
rf_firstpeak = joblib.load("rf_firstpeak_model.pkl")
rf_zerocross = joblib.load("rf_zerocross_model.pkl")

# === 2. System definition (state-space with two inputs: disturbance, control) ===
A = np.array([[0, 1],
              [-7108.78, -0.86]])
B = np.array([[105.61, 105.61],
              [10845.61, 10845.61]])
C = np.array([[1, 0]])
D = np.zeros((1,2))

sys = StateSpace(A, B, C, D)

# === 3. Simulation parameters ===
dt = 0.001
T_total = 1.0
t = np.arange(0, T_total, dt)
n_samples = len(t)

# === 4. Generate real disturbance (sin² pulse) ===
Io = 0.25
Ts = 0.0372
u_dist = np.zeros(n_samples)
pulse_idx = t <= Ts
u_dist[pulse_idx] = Io * np.sin(np.pi * t[pulse_idx] / Ts)**2

# === 5. Simulate disturbance-only system (without control) ===
sysA = StateSpace(A, B[:,[0]], C, D[:,[0]])
_, y_no_control, _ = lsim(sysA, U=u_dist, T=t)

# === 6. Extract ML features from disturbance-only response ===
threshold = 0.01
idx = np.argmax(y_no_control > threshold)
y_thresh = y_no_control[idx:idx+3]
slope3pts = (y_thresh[2] - y_thresh[0]) / (2*dt)
third_val = y_thresh[2]

X = np.array([[slope3pts, third_val]])
X_enhanced = np.column_stack([X, X[:,0]**2, X[:,1]**2, X[:,0]*X[:,1]])

# === 7. ML Predictions ===
firstpeak_pred = rf_firstpeak.predict(X_enhanced)[0]
zerocross_pred = rf_zerocross.predict(np.column_stack([X_enhanced, [firstpeak_pred]]))[0]

# === 8. Construct ML-predicted impulse ===
u_ml = np.zeros(n_samples)
launch_time = t[idx + 3 + 1]
zero_cross_time = zerocross_pred
for k, tk in enumerate(t):
    if launch_time <= tk <= zero_cross_time:
        u_ml[k] = firstpeak_pred * np.sin(np.pi * (tk - launch_time) / (zero_cross_time - launch_time))

# === 9. Scale ML impulse ===
_, y_ml_test, _ = lsim(sysA, U=u_ml, T=t)
alpha = np.max(np.abs(y_no_control)) / np.max(np.abs(y_ml_test))
u_ml_scaled = -alpha * u_ml

# === 10. ML + PID simulation ===
Kp, Ki, Kd = 0.1, 0.0, 0.001
y_setpoint = 0.0
integral = 0.0
prev_error = 0.0

x_c = np.zeros((2, n_samples))
y_ml_pid = np.zeros(n_samples)

for k in range(1, n_samples):
    # PID calculation
    error = y_setpoint - y_ml_pid[k-1]
    integral += error*dt
    derivative = (error - prev_error)/dt
    u_pid = Kp*error + Ki*integral + Kd*derivative
    prev_error = error

    # Total input: disturbance + ML control + PID
    u_total = np.array([u_dist[k-1], u_ml_scaled[k-1] + u_pid])
    x_c[:,k] = x_c[:,k-1] + dt*(A @ x_c[:,k-1] + B @ u_total)
    y_ml_pid[k] = C @ x_c[:,k]

# === 11. PID-only simulation ===
x_pid = np.zeros((2, n_samples))
y_pid_only = np.zeros(n_samples)
integral = 0.0
prev_error = 0.0

for k in range(1, n_samples):
    error = y_setpoint - y_pid_only[k-1]
    integral += error*dt
    derivative = (error - prev_error)/dt
    u_pid = Kp*error + Ki*integral + Kd*derivative
    prev_error = error

    # PID-only input reacts to disturbance
    u_total = u_dist[k-1] + u_pid
    x_pid[:,k] = x_pid[:,k-1] + dt*(A @ x_pid[:,k-1] + B[:,0]*u_total)
    y_pid_only[k] = C @ x_pid[:,k]

# === 12. Plot all responses ===
plt.figure(figsize=(12,7))
plt.plot(t, y_no_control, label="Without Control (Disturbance Only)", color='navy', linewidth=1.5)
plt.plot(t, y_pid_only, label="PID Only", color='orange', linewidth=1.5)
plt.plot(t, y_ml_pid, label="With Control (ML + PID)", color='darkgreen', linewidth=1.5)
plt.xlabel("Time [s]")
plt.ylabel("Displacement")
plt.title("Beam Tip Response: Without Control, PID Only, ML + PID")
plt.grid(True)
plt.legend()
plt.show()
