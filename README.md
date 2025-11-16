# 🌊 CandorFlow

**Early Warning System for Training Instabilities**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## ⚠️ Important Notice

**This repository contains a SIMPLIFIED, PUBLIC DEMONSTRATION of CandorFlow concepts.**

This is NOT the full proprietary system. Many advanced features, algorithms, and optimizations are intentionally excluded. See [What Is NOT Included](#what-is-not-included-proprietary) for details.

---

## 📖 Overview

CandorFlow is a training stability monitoring and intervention system designed to detect and prevent neural network training instabilities before they cause divergence.

This public repository demonstrates:
- A simplified stability metric **λ(t)** based on gradient variance
- Basic threshold-based monitoring
- Automatic checkpoint rollback on instability detection
- Learning rate reduction for recovery
- Minimal working examples with toy models

### What is λ(t)?

The lambda metric **λ(t)** is a stability indicator that tracks training health over time. In this simplified demo, it measures gradient norm variance as a proxy for instability.

**High λ(t) → Training is becoming unstable**  
**Low λ(t) → Training is stable**

---

## 🎯 Features in This Demo

### ✅ What This Repo Contains (Safe/Public Demo)

- **Simplified λ(t) metric**: Gradient norm variance-based instability detection
- **Basic stability controller**: Threshold monitoring with rollback capabilities
- **Checkpoint management**: Automatic saving and restoration
- **Learning rate adaptation**: Halving on instability detection
- **Minimal training loop**: Toy example with intentional instability
- **Visualization tools**: Plot λ(t) curves and stability phases
- **Jupyter notebook**: Interactive demo with explanations
- **Reproducible examples**: Fully runnable on CPU or GPU

---

## 🚫 What Is NOT Included (Proprietary)

The full CandorFlow system includes many advanced features that are **NOT** in this public demo:

### Core Algorithms
- ❌ **Universal scaling law** for λ(t)
- ❌ **Reflexive ridge equation** and closed-form solutions
- ❌ **Cross-domain invariants** (works across NLP, vision, RL, etc.)
- ❌ **Jacobian spectral analysis** for stability prediction
- ❌ **Multi-signal fusion** (loss, gradients, activations, etc.)

### Advanced Control
- ❌ **Real-time stability engine** with predictive modeling
- ❌ **Reflexive decay algorithms** for adaptive intervention
- ❌ **Temporal smoothing with active inference**
- ❌ **Dynamic threshold adaptation** based on training phase
- ❌ **HPC-optimized control loops** for large-scale training

### Domain Extensions
- ❌ **ECG anomaly detection** applications
- ❌ **Earthquake early warning** systems
- ❌ **Financial market stability** monitoring
- ❌ **General-purpose time series** instability detection

### Performance
- ❌ **Production-grade optimizations** for minimal overhead
- ❌ **Distributed training integration** (DeepSpeed, FSDP, etc.)
- ❌ **Hardware acceleration** (CUDA kernels, etc.)

**For access to the full proprietary system, please contact us.**

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch transformers numpy matplotlib jupyter notebook
```

---

## 💻 Usage

### Quick Start: Run the Training Demo

```bash
python examples/demo_training_loop.py
```

This will:
1. Create a small neural network
2. Train it on dummy data
3. Compute λ(t) at each step
4. Intentionally inject instability after step 30
5. Demonstrate automatic detection and rollback
6. Save results to `plots/training_results.pt`

**Expected output:**
```
================================================================
CandorFlow Training Stability Demo
================================================================
NOTE: This is a simplified demonstration.
The full proprietary system includes many additional features.
================================================================

📦 Creating model...
🔧 Initializing stability controller...
🚀 Starting training for 50 steps...

Step   0 | Loss: 0.6931 | λ(t): 0.0000 | Action: none
Step   5 | Loss: 0.6895 | λ(t): 0.0234 | Action: none
...
⚠️  INSTABILITY DETECTED at step 35: λ(t)=3.4521 (threshold=2.0)
✓ Rolled back to stable checkpoint from step 25
✓ Reduced learning rate: 0.001000 → 0.000500
```

### Visualize Results

```bash
python examples/demo_plots.py
```

This generates:
- `plots/lambda_curve.png` - λ(t) over time with intervention markers
- `plots/stability_phases.png` - Color-coded stability zones

### Interactive Notebook

```bash
jupyter notebook notebooks/CandorFlow_Demo.ipynb
```

The notebook includes:
- Step-by-step explanations
- Live training visualization
- Interactive parameter tuning
- Educational content about stability monitoring

---

## 📁 Repository Structure

```
CandorFlow/
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
│
├── candorflow/                 # Main package
│   ├── __init__.py            # Package initialization
│   ├── lambda_metric.py       # Simplified λ(t) computation
│   ├── stability_controller.py # Basic monitoring & intervention
│   ├── utils.py               # Checkpoint and logging utilities
│   └── version.py             # Version information
│
├── examples/                   # Runnable demos
│   ├── demo_training_loop.py  # Training with stability monitoring
│   └── demo_plots.py          # Visualization generation
│
├── notebooks/                  # Jupyter notebooks
│   └── CandorFlow_Demo.ipynb  # Interactive tutorial
│
└── plots/                      # Output directory for plots
    └── (generated files)
```

---

## 🔬 How It Works (Simplified Version)

### 1. Monitor Training with λ(t)

```python
from candorflow import compute_lambda_metric, StabilityController

# During training loop
lambda_value = compute_lambda_metric(
    model=model,
    loss=loss,
    gradient_history=gradient_history
)
```

### 2. Automatic Intervention

```python
controller = StabilityController(threshold=2.0)

action = controller.update(
    lambda_value=lambda_value,
    model=model,
    optimizer=optimizer,
    step=step
)

if action["action"] == "rollback":
    print("Instability detected - rolling back to stable checkpoint")
```

### 3. Training Continues Safely

The controller automatically:
- Saves checkpoints when training is stable
- Detects when λ(t) exceeds threshold
- Rolls back to last stable state
- Reduces learning rate
- Resumes training

---

## 📊 Example Results

After running the demo, you'll see plots like this:

**Lambda Curve with Interventions:**
- Blue line: λ(t) stability metric over time
- Purple dashed line: Instability threshold
- Orange markers: Rollback + LR reduction events
- Red markers: Warnings

**Stability Phases:**
- Green zone: Stable training
- Orange zone: Warning (approaching threshold)
- Red zone: Unstable (intervention triggered)

---

## 🧪 Running Tests

The demo includes built-in validation:

```bash
# Run training demo (includes self-checks)
python examples/demo_training_loop.py

# Generate plots (validates results)
python examples/demo_plots.py
```

---

## 📚 Documentation

### API Reference

#### `compute_lambda_metric(model, loss, history_window=10, gradient_history=None)`

Compute simplified λ(t) stability metric.

**Parameters:**
- `model` (torch.nn.Module): Neural network model
- `loss` (torch.Tensor): Current loss value (with grad_fn)
- `history_window` (int): Number of past gradient norms to track
- `gradient_history` (list): List to store gradient history (modified in-place)

**Returns:**
- `lambda_value` (float): Stability metric (higher = more unstable)

#### `StabilityController(threshold, checkpoint_dir, lr_reduction_factor)`

Training stability monitor and intervention system.

**Parameters:**
- `threshold` (float): λ(t) value above which to trigger intervention
- `checkpoint_dir` (str): Directory for saving checkpoints
- `lr_reduction_factor` (float): Factor to reduce LR by (default: 0.5)

**Methods:**
- `update(lambda_value, model, optimizer, step)`: Update controller and take action if needed
- `get_summary()`: Get training statistics

---

## 🤝 Contributing

This is a demonstration repository. Contributions are welcome for:
- Bug fixes in demo code
- Documentation improvements
- Additional visualization examples
- Educational content

**Note:** This repo intentionally excludes proprietary algorithms. Please do not submit PRs attempting to implement advanced features from the full system.

---

## 📧 Contact

For questions about this demo:
- Open an issue on GitHub

For inquiries about the full proprietary CandorFlow system:
- Email: [your-email@example.com]
- Website: [https://candorflow.example.com]
- Patents: [Patent application numbers]

---

## 📄 License

This simplified demonstration code is released under the MIT License. See [LICENSE](LICENSE) for details.

**Important:** The full CandorFlow system, including its proprietary algorithms and commercial applications, is NOT covered by this license. Please contact us for commercial licensing.

---

## 📖 Citation

If you use this demo code in your research or project, please cite:

```bibtex
@software{candorflow2025,
  title={CandorFlow: Training Stability Monitoring System},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/CandorFlow},
  note={Simplified public demonstration version}
}
```

---

## 🙏 Acknowledgments

This simplified demo is provided for educational purposes to demonstrate basic concepts in training stability monitoring.

The full CandorFlow system represents significant research and development investment and is protected by pending patents.

---

## ⭐ Star History

If you find this demo helpful, please consider starring the repository!

---

**Built with ❤️ for safer, more reliable AI training**

