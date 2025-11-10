---
title: MNIST GAN Digit Generator
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# MNIST Digit Generator

A production-ready Generative Adversarial Network (GAN) for synthesizing realistic handwritten digits, deployed as an interactive web application.

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a Generative Adversarial Network from scratch using PyTorch, demonstrating proficiency in:
- Deep learning architecture design and implementation
- Adversarial training with stability techniques
- Production-grade code with error handling and logging
- Interactive application deployment

**[Launch Demo on Hugging Face](https://huggingface.co/spaces/rvikranth10/mnist-gan)** 🚀

## Key Features

- **High-Quality Generation**: Synthesizes realistic handwritten digits indistinguishable from MNIST samples
- **Production-Ready Code**: Comprehensive error handling, logging, input validation, and type hints
- **Interactive Interface**: Professional Gradio application with configurable parameters
- **Stable Training**: Achieved convergence over 200 epochs without mode collapse

## Technical Specifications

### Model Architecture

**Generator Network**
```
Input: 100-dim noise vector (N(0,1))
  ↓ Linear(100→256) + LeakyReLU + BatchNorm
  ↓ Linear(256→512) + LeakyReLU + BatchNorm  
  ↓ Linear(512→1024) + LeakyReLU + BatchNorm
  ↓ Linear(1024→784) + Tanh
Output: 28×28 image (range: [-1,1])

Parameters: 1,489,936
```

**Discriminator Network**
```
Input: 784-dim flattened image
  ↓ Linear(784→1024) + LeakyReLU + Dropout(0.3)
  ↓ Linear(1024→512) + LeakyReLU + Dropout(0.3)
  ↓ Linear(512→256) + LeakyReLU + Dropout(0.3)
  ↓ Linear(256→1) + Sigmoid
Output: Probability [0,1]

Parameters: 1,460,225
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (60,000 samples) |
| Epochs | 200 |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Beta1 | 0.5 |
| Loss Function | Binary Cross-Entropy |
| Device | CPU / GPU (MPS/CUDA) |
| Training Time | ~30-45 min (GPU) / ~2-3 hours (CPU) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Final Generator Loss | 0.981 |
| Final Discriminator Loss | 1.213 |
| Mode Collapse | None observed |
| Output Quality | Realistic, diverse |
| Convergence | Stable |

## Results

### Training Progress

| Epoch | Discriminator Loss | Generator Loss | Quality |
|-------|-------------------|----------------|---------|
| 1 | 0.698 | 1.638 | Random noise |
| 50 | 1.314 | 0.816 | Emerging shapes |
| 100 | 1.298 | 0.838 | Clear digits |
| 150 | 1.253 | 0.916 | Refined quality |
| 200 | 1.213 | 0.981 | High quality |

### Sample Outputs

<table>
  <tr>
    <td align="center">
      <img src="samples/epoch_050.png" width="200"/><br/>
      <b>Epoch 50</b>
    </td>
    <td align="center">
      <img src="samples/epoch_100.png" width="200"/><br/>
      <b>Epoch 100</b>
    </td>
    <td align="center">
      <img src="samples/epoch_200.png" width="200"/><br/>
      <b>Epoch 200</b>
    </td>
  </tr>
</table>

![Training Loss](losses.png)

## Project Structure

```
mnist-gan/
├── app.py                          # Production Gradio application
├── GAN_MNIST_Assignment.ipynb      # Training notebook with analysis
├── generator_model.pth             # Trained model weights (~17 MB)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── GITHUB_SETUP.md                 # Deployment instructions
├── LICENSE                         # MIT License
├── .gitignore                      # Git configuration
├── losses.png                      # Training visualization
├── samples/                         # Generated samples
│   ├── epoch_001.png
│   ├── epoch_050.png
│   ├── epoch_100.png
│   ├── epoch_150.png
│   └── epoch_200.png
└── data/                           # MNIST dataset (created automatically)
    └── MNIST/                      # Downloaded by torchvision on first run
```

**Note:** The `data/` directory is created automatically when you run the notebook. It's excluded from git via `.gitignore` to keep the repository size small.

## Hardware Acceleration

This project automatically detects and uses GPU acceleration when available:

| Device Type | Technology | Training Time | Speedup |
|------------|-----------|---------------|---------|
| **Apple Silicon** (M1/M2/M3/M4) | Metal (MPS) | ~30-45 min | 3-5x |
| **NVIDIA GPU** | CUDA | ~20-30 min | 5-10x |
| **CPU** | Native | ~2-3 hours | 1x (baseline) |

No configuration needed - the code automatically selects the best available device!

## Quick Start

### Option 1: Use Pre-trained Model (Fast)

If you just want to run the application with the pre-trained model:

```bash
# Clone repository
git clone https://github.com/vikranth1000/mnist-gan.git
cd mnist-gan

# Install dependencies
pip install -r requirements.txt

# Launch application (uses pre-trained generator_model.pth)
python app.py
```

Access at `http://localhost:7860`

### Option 2: Reproduce Training from Scratch

To reproduce the entire training process:

#### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab
- ~2-3 hours for training (on CPU)

#### Step-by-Step Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/vikranth1000/mnist-gan.git
   cd mnist-gan
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the training notebook**
   ```bash
   jupyter notebook GAN_MNIST_Assignment.ipynb
   # Or use: jupyter lab GAN_MNIST_Assignment.ipynb
   ```

4. **Run all cells sequentially**
   - The notebook will automatically download the MNIST dataset (stored in `data/` directory)
   - Dataset size: ~60MB (downloaded automatically on first run)
   - Training takes approximately 2-3 hours on CPU

5. **Model is automatically saved**
   - Cell 26 automatically saves `generator_model.pth` after training completes
   - This file is required for the Gradio application

6. **Verify outputs**
   - Check `samples/` directory for generated images at epochs 1, 50, 100, 150, 200
   - Check `losses.png` for training loss visualization
   - Verify `generator_model.pth` exists (~17 MB)

#### Expected Results

After training, you should see:
- **Final Generator Loss:** ~0.98
- **Final Discriminator Loss:** ~1.21
- **Training Time:** 
  - Apple Silicon M4: ~30-45 minutes
  - NVIDIA GPU: ~20-30 minutes
  - CPU: ~2-3 hours
- **Model File:** `generator_model.pth` (~17 MB)
- **Sample Images:** `samples/epoch_*.png` (5 files)
- **Loss Plot:** `losses.png`

#### Notes

- **Data Download:** MNIST dataset downloads automatically via `torchvision.datasets.MNIST` (no manual download needed)
- **Reproducibility:** The notebook uses `seed=42` by default for reproducible results
- **GPU Acceleration:** Automatically detects and uses available GPU:
  - **Apple Silicon (M1/M2/M3/M4)**: Uses Metal Performance Shaders (MPS) - 3-5x faster
  - **NVIDIA GPUs**: Uses CUDA - 5-10x faster
  - **CPU Fallback**: Works on any system, just slower
- **Memory:** Training requires ~2-4 GB RAM

### Deployment

See [GITHUB_SETUP.md](GITHUB_SETUP.md) for detailed deployment instructions to Hugging Face Spaces.

## Code Quality

### Production Features

- **Type Hints**: Full type annotation for better code maintainability
- **Error Handling**: Comprehensive exception handling with graceful degradation
- **Logging**: Structured logging for debugging and monitoring
- **Input Validation**: Parameter validation and sanitization
- **Memory Management**: Proper resource cleanup and matplotlib figure handling
- **Documentation**: Detailed docstrings and inline comments
- **Configuration**: Constants and configuration management
- **Modularity**: Clean separation of concerns with dedicated classes

### Best Practices

- PEP 8 style compliance
- Defensive programming
- Single Responsibility Principle
- DRY (Don't Repeat Yourself)
- Clear error messages
- Resource management

## Technical Skills Demonstrated

- **Deep Learning**: GAN architecture, adversarial training, loss functions
- **PyTorch**: Model building, training loops, gradient management
- **Software Engineering**: Code organization, error handling, logging
- **Web Development**: Gradio interface, user experience design
- **DevOps**: Deployment, monitoring, documentation
- **Best Practices**: Type hints, validation, testing

## Implementation Details

### Key Design Decisions

1. **LeakyReLU Activation**: Prevents dying neurons in discriminator (α=0.2)
2. **Batch Normalization**: Stabilizes generator training and improves convergence
3. **Dropout Regularization**: Prevents discriminator overfitting (p=0.3)
4. **Tanh Output**: Maps to [-1, 1] range matching normalized MNIST data
5. **Temperature Scaling**: Allows control over output diversity

### Training Stability Techniques

- Balanced learning rates for generator and discriminator
- Batch normalization in generator
- Dropout in discriminator
- Separate optimization steps
- Gradient clipping (implicit through Adam)

## Future Enhancements

- [ ] Conditional GAN (cGAN) for digit-specific generation
- [ ] Deep Convolutional GAN (DCGAN) architecture
- [ ] Wasserstein GAN (WGAN) for improved stability
- [ ] FID score evaluation
- [ ] Latent space interpolation visualization
- [ ] Batch generation API
- [ ] Model versioning and A/B testing

## Reproducing This Project

**Quick Summary:**
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `GAN_MNIST_Assignment.ipynb` in Jupyter
4. Run all cells sequentially (Cells 1-26)
5. MNIST dataset downloads automatically (~60 MB)
6. Training automatically uses GPU if available:
   - Apple Silicon: ~30-45 minutes
   - NVIDIA GPU: ~20-30 minutes
   - CPU: ~2-3 hours
7. Cell 26 automatically saves `generator_model.pth` after training completes

## References

- Goodfellow et al. (2014) - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- Radford et al. (2015) - [DCGAN](https://arxiv.org/abs/1511.06434)
- MNIST Database - [Yann LeCun](http://yann.lecun.com/exdb/mnist/)

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Author

**Vikranth Reddimasu**
- GitHub: [@vikranth1000](https://github.com/vikranth1000)
- LinkedIn: [linkedin.com/in/vikranthreddimasu](https://www.linkedin.com/in/vikranthreddimasu/)
- Hugging Face: [@rvikranth10](https://huggingface.co/rvikranth10)

## Acknowledgments

- MNIST dataset by Yann LeCun, Corinna Cortes, and Christopher Burges
- PyTorch team for the deep learning framework
- Hugging Face for the Spaces platform
- Ian Goodfellow for pioneering GANs

---

<p align="center">
  <i>Built with PyTorch and Gradio</i>
</p>
