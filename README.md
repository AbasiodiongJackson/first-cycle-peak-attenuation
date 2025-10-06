# ML-Based First-Cycle Peak Attenuation for Beam Vibrations

## Project Overview

This project demonstrates a **machine learning-driven vibration control method** designed to reduce the first-cycle peak of a beam’s free vibration following an impulsive disturbance. Traditional vibration controllers primarily focus on damping the system and cannot prevent the initial peak from exceeding safety limits. By predicting the first-cycle peak early, this approach allows for a **counter pulse** to be applied to mitigate potential damage.

Key highlights:  
- **Machine Learning Prediction**: A model trained on 15,000 simulated vibration datasets predicts the **first-cycle peak** and **zero-crossing time** within the first few samples of motion.  
- **Counter Pulse Generation**: Using the predicted peak, a counteracting impulse is applied to reduce the first-cycle amplitude.  
- **PID Control Integration**: A PID controller handles the residual vibration, ensuring effective damping and faster settling time.  
- **Real-Time Simulation**: The workflow is implemented in Python with **runtime evaluation** of system response and control generation.  

## Features

- ML-based first-cycle peak and zero-crossing prediction  
- Scaled counter impulse for destructive interference  
- PID controller for residual vibration damping  
- Stepwise beam simulation in Python  
- Visualization of system response with and without control  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/ml-beam-control.git
cd ml-beam-control
```
2. Install required Python packages:
   ```bash
pip install numpy matplotlib scipy scikit-learn joblib
```
## Usage
1. Ensure the trained ML models are available in the repository:
 ```bash - rf_firstpeak_model.pkl ```
 ```bash - rf_zerocross_model.pkl ```

2. Run the main simulation script:
```bash
python first-cycle-peak-attenuation.py
```
3. The script will produce plots shoowing
  - Beam response without control
  - Beam response with ML + PID control
  - PID-only control response

## How It Works

1. Disturbance Simulation: A sin² impulse excites the beam.

2. ML Feature Extraction: Initial samples after threshold crossing are used to compute slope and amplitude features.

3. Peak & Zero-Cross Prediction: Trained Random Forest models predict the first-cycle peak and zero-crossing time.

4. Counter Impulse Generation: A scaled, half-sine counter pulse is applied to reduce the peak amplitude.

5. PID Control: A PID loop mitigates the remaining vibration to reduce settling time.

## Results

The approach achieves:

- Significant reduction (>50% in some cases) of the first-cycle peak

- Effective damping of subsequent cycles via PID control

- Demonstration of ML-based predictive control integrated with classical PID

## License

This project is released under the MIT License.
