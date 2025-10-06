# Beam Vibration Control: ML-Predicted Impulse + PID

## Project Overview
This project implements a **hybrid control strategy** for attenuating the first cycle peak in a vibrating beam. It combines:

- **Machine Learning (ML)** models to predict the first cycle peak and zero-cross time from the initial disturbance.
- **ML-generated impulses** to counteract the predicted peak in real-time.
- **PID control** to handle the subsequent vibrations and improve settling time.

The simulation is implemented in Python using `NumPy`, `SciPy`, and `Matplotlib`.

---

## Features

1. **ML Prediction**  
   - Predicts first cycle peak amplitude and zero-cross timing from system response.
   - Generates a scaled counteracting impulse for the beam.

2. **PID Control**  
   - Simple PID controller (`Kp`, `Ki`, `Kd`) reduces residual oscillations.
   - Can be run standalone or combined with the ML-predicted impulse.

3. **Visualization**  
   - Compares system response:
     - Without control (disturbance only)
     - PID only
     - ML + PID control

---

## Dependencies

- Python 3.8+  
- `numpy`  
- `scipy`  
- `matplotlib`  
- `joblib` (for loading trained ML models)  

Install dependencies via pip:

```bash
pip install numpy scipy matplotlib joblib

```

## How to Run

1. Clone or download the repository.

2. .pkl ML model files are available upon request.

3. Run the main simulation:

```bash
python main_simulation.py
```

4. The plots will display system response comparisons.

## Notes

- For a different system, the ML models should be pre-trained and saved in .pkl format.

- The scaling factor (alpha) ensures the ML-generated impulse amplitude matches the system response.

- PID gains can be adjusted in the scripts (Kp, Ki, Kd) for tuning.

## Author

Dr. Abasiodiong Jackson

## License

MIT License




