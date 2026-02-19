# ⚡ Smart Grid Agentic AI Monitor

A production-ready Streamlit web application that visualises the live output of every agent in the Agentic AI Smart Grid framework in real time — including runtime configuration drift, adaptive learning, and per-agent performance metrics.

---

## Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Repository Structure

All files live at the **root level** (no subdirectories):

```
app.py                        ← Streamlit web application
runtime_config.py             ← Centralised RuntimeConfig class
agentic_framework.py          ← Framework orchestrator
base_agent.py                 ← Abstract BaseAgent
layer1_perceptual_agents.py   ← BehavioralEnvelopeAgent + AnomalyDetectionAgent
layer2_cognitive_agents.py    ← CascadePredictorAgent
layer3_strategic_agents.py    ← MitigationGeneratorAgent (ε-greedy)
adaptive_controller.py        ← AdaptiveController + LearningController
agent_performance_monitor.py  ← Per-agent observability + config drift
evaluation_pipeline.py        ← 50-cycle research evaluation harness
integration_example.py        ← End-to-end demo (4 scenarios)
requirements.txt
README.md
```

---

## Architecture

### Agent Layers

| Layer | Agent | Role |
|---|---|---|
| Perceptual | `BehavioralEnvelopeAgent` | Rolling σ-envelope anomaly detection |
| Perceptual | `AnomalyDetectionAgent` | IsolationForest unsupervised detection |
| Cognitive  | `CascadePredictorAgent` | BFS cascade propagation simulation |
| Strategic  | `MitigationGeneratorAgent` | ε-greedy strategy selection with Q-table |

### RuntimeConfig — Single Source of Truth

All four tunable parameters live in one `RuntimeConfig` object owned by the framework:

| Parameter | Default | Controls |
|---|---|---|
| `behavioral_sensitivity` | 3.0 | σ multiplier for envelope width |
| `anomaly_contamination` | 0.05 | IsolationForest expected outlier ratio |
| `cascade_propagation_threshold` | 0.5 | Minimum edge weight for cascade spread |
| `strategy_exploration_rate` | 0.2 | ε in ε-greedy mitigation policy |

Agents read these values **on every inference call** — no restart needed.

---

## How the Adaptive Loop Works

1. **`AgenticFramework.run_cycle()`** executes all four agent layers and returns a structured output dict.
2. **`AdaptiveController.adapt()`** extracts false-positive / false-negative proxies from the cycle output and passes them to the internal `LearningController`.
3. **`LearningController.update()`** adjusts all four parameters using online gradient-style rules.
4. **`framework.runtime_config.update_from_dict()`** pushes the new values to the shared config — all agents pick them up on the very next call.
5. The **`AgentPerformanceMonitor`** records every config snapshot, enabling convergence and drift analysis.

After ~50 cycles the system typically:
- Reduces false positives (sensitivity widens, contamination drops)
- Stabilises cascade aggressiveness
- Reduces mitigation exploration as Q-values converge

---

## Web App Guide

### Sidebar Controls

| Control | Description |
|---|---|
| **Synthetic Data Toggle** | Use built-in sine-wave + noise generator vs. manual values |
| **Manual Sensor Values** | Comma-separated floats when synthetic is off |
| **Cycles to Run** | 1–100 cycles per "Run Multiple" click |
| **Enable Adaptive Controller** | Attaches `AdaptiveController`; updates config after each cycle |
| **Manual Config Override** | Directly set any parameter mid-run |
| **Reset Framework** | Clears all state and reinitialises |

### Tabs

| Tab | Content |
|---|---|
| 🔬 Envelope Agent | Anomaly list, z-scores, σ threshold used |
| 🧠 Anomaly Detection | IsolationForest predictions, scores, buffer size |
| 🌊 Cascade Predictor | Affected nodes, propagation paths, cascade depth |
| 🛡️ Mitigation | Selected strategy, exploration/exploitation mode, Q-values |
| ⚙️ Runtime Config | Live parameter values, change log, drift indicators |
| 📊 Performance | Per-agent latency, detection stats, strategy distribution |

---

## Deployment on Streamlit Community Cloud

1. Push the repository to GitHub (all files at root level).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, set branch to `main`, and set **Main file path** to `app.py`.
4. Click **Deploy** — dependencies are installed automatically from `requirements.txt`.

> `scikit-learn` is listed as a dependency; the framework falls back to a z-score heuristic if it is unavailable, so the app is functional in minimal environments too.

---

## Screenshots

<!-- Add screenshots here after deployment -->
| Overview | Adaptive Loop |
|---|---|
| *(screenshot placeholder)* | *(screenshot placeholder)* |

---

## Research Metrics

Run `evaluation_pipeline.py` directly for a full 50-cycle research evaluation:

```bash
python evaluation_pipeline.py
```

Outputs include `config_adjustment_count`, `convergence_rate_of_thresholds`, and `stability_of_detection_over_time` — directly citeable in the thesis.
