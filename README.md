# Accurate Contact-Aware Control of a 2-Link Hopper

Overview
- Hybrid model-based control pipeline for a planar 2-link hopping robot (2D hopper) in MuJoCo.
- Combine an analytical Lagrangian dynamics model $f(x, u)$ with a learned residual $g_{\theta}(x, u)$ that captures contact-induced nonlinearities.
- Use a PPO-trained policy to collect safe, physically meaningful trajectories for residual learning, then embed the hybrid model in an MPPI controller.

Theoretical topics
- Lagrangian derivation of robot dynamics: mass, Coriolis, gravity terms.
- Hybrid dynamics modeling: $\dot x = f(x, u) + g_{\theta}(x, u)$.
- Residual dynamics learning (supervised regression of residuals from $(x, u, \dot x)$ data).
- Reinforcement learning for safe data collection (PPO).
- Model Predictive Path Integral (MPPI) control for nonlinear/black-box dynamics.
- Practical issues: data coverage, out-of-distribution residual outputs, numerical stability during rollouts.

Quick results/highlights
- PPO-generated trajectories provided high-quality coverage for learning.
- Residual neural network (3×128 hidden layers, ReLU, Adam) achieved strong regression performance on collected data.
- MPPI integrated with the hybrid model; rollout numerical instabilities (NaNs) were observed — mitigation strategies are suggested (input normalization, tighter action sampling, fallback dynamics).

Tech stack
- Simulator: MuJoCo (Gym-compatible custom environment)
- RL: PPO (PyTorch)
- Residual model: Feedforward NN (PyTorch)
- Control: MPPI (custom implementation)
- Utilities: NumPy, SymPy (symbolic derivation), Matplotlib
- Data: NumPy arrays for logged $(x, u, x′)$ / computed $\dot x$
