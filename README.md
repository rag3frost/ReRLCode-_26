<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Algorithm-SARSA(%CE%BB)-brightgreen" alt="Algorithm">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

# SARSA(&#955;) Traffic Signal Control

A reinforcement learning implementation for intelligent traffic signal optimization using the SARSA(&#955;)(lambda) algorithm. This project trains an adaptive traffic light controller that learns to minimize vehicle queue lengths and waiting times at a four-way intersection.

## Overview

Traditional traffic signals operate on fixed-timing schedules regardless of real-time traffic demand. This system learns an **adaptive control policy** through reinforcement learning, responding dynamically to changing traffic conditions and outperforming conventional fixed-cycle controllers.

### How It Works

- **State**: Discretized queue lengths on four intersection approaches (North, South, East, West)
- **Actions**: Signal phase selection -- NS through, EW through, N turn, E turn
- **Reward**: Negative weighted sum of queue length and delay, penalizing congestion
- **Learning**: On-policy SARSA(&#955;) with eligibility traces for efficient credit assignment

## Project Structure

| File | Description |
|---|---|
| `sarsa_lambda_traffic_control.py` | Full training pipeline -- environment, agent, evaluation, and visualization |
| `sarsa_lambda_traffic_signal_control.html` | Interactive web-based simulator with live training and visualization |
| `SARSA_Lambda_Traffic_Lab_Report.pdf` | Detailed lab report with analysis and results |

## Key Components

### Traffic Environment

Simulates a four-way intersection with configurable traffic modes:

- **Peak** -- high arrival rates (0.7-0.8 vehicles/step)
- **Normal** -- moderate traffic (0.4-0.5 vehicles/step)
- **Light** -- low demand (0.2-0.3 vehicles/step)

Automatically transitions between modes during training to learn robust policies.

### SARSA(&#955;) Agent

| Hyperparameter | Symbol | Default |
|---|---|---|
| Learning rate | alpha | 0.10 |
| Discount factor | gamma | 0.95 |
| Eligibility trace decay | lambda | 0.60 |
| Exploration rate | epsilon | 0.20 (decaying) |

Eligibility traces enable multi-step backup, propagating temporal-difference errors to previously visited state-action pairs for faster convergence.

### Evaluation

The trained agent is evaluated against a **fixed-timing baseline** (10-step cyclic phase rotation) across three metrics:

- Cumulative reward
- Total waiting time
- Vehicles served (throughput)

## Usage

### Python Training Pipeline

```bash
python sarsa_lambda_traffic_control.py
```

Runs the full experiment:
1. Train agent for 1,000 episodes
2. Evaluate trained policy over 50 episodes
3. Compare against fixed-timing baseline
4. Generate visualization charts (result plots and policy heatmaps)

### Interactive Simulator

Open `sarsa_lambda_traffic_signal_control.html` in a browser. The dashboard provides:

- Adjustable hyperparameter sliders (alpha, gamma, lambda, epsilon)
- Traffic demand and episode count controls
- Real-time training progress charts
- Live intersection visualization with animated vehicles
- Performance comparison bars against fixed-time control

## Results

The SARSA(&#955;) agent demonstrates measurable improvement over fixed-timing control across all evaluation metrics. Key findings:

- **Reduced waiting time** through adaptive phase selection
- **Increased throughput** by prioritizing lanes with longer queues
- **Robust behavior** across varying traffic demand modes

Refer to `SARSA_Lambda_Traffic_Lab_Report.pdf` for detailed numerical results and analysis.

## Dependencies

```
numpy
matplotlib
seaborn
pandas
```