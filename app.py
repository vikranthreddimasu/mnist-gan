"""
MNIST GAN Digit Generator Application
A production-ready Gradio interface for generating handwritten digits using a trained GAN.

Author: Vikranth Reddimasu
License: MIT
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server deployment
import matplotlib.pyplot as plt
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
NOISE_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 784
IMAGE_SIZE = 28
MODEL_PATH = "generator_model.pth"
DEFAULT_SEED = 42
MIN_IMAGES = 1
MAX_IMAGES = 16
MIN_TEMPERATURE = 0.5
MAX_TEMPERATURE = 2.0


class Generator(nn.Module):
    """
    Fully Connected Generator for MNIST digit generation.
    
    Architecture:
        - Input: Random noise vector (100-dim)
        - Fully connected layers with LeakyReLU and BatchNorm
        - Output: 784-dim vector (28×28 image) with Tanh activation
    
    Args:
        noise_dim: Dimension of input noise vector (default: 100)
        hidden_dim: Dimension of hidden layers (default: 256)
        output_dim: Dimension of output vector (default: 784)
    """
    
    def __init__(
        self, 
        noise_dim: int = NOISE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM
    ):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # First layer
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            # Second layer
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            
            # Third layer
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 4),
            
            # Output layer
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()  # Output in range [-1, 1] to match normalized images
        )
        
        logger.info(f"Generator initialized with {self._count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        return self.model(x)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = Path(model_path)
        
        # Auto-detect best available device (CUDA > MPS > CPU)
        # Use try-except for MPS check as it may not be available on all platforms
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        except Exception:
            # Fallback to CPU if device detection fails
            self.device = torch.device('cpu')
        
        self.generator = None
        self.model_info = None
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained generator model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Initialize generator
            self.generator = Generator(
                noise_dim=NOISE_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=OUTPUT_DIM
            ).to(self.device)
            
            # Load checkpoint if available
            if self.model_path.exists():
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True
                )
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.generator.eval()
                
                epochs = checkpoint.get('epoch', 'N/A')
                g_loss = checkpoint.get('generator_loss', 'N/A')
                
                self.model_info = (
                    f"Model trained for {epochs} epochs | "
                    f"Generator Loss: {g_loss:.4f}" if isinstance(g_loss, float) 
                    else f"Model trained for {epochs} epochs"
                )
                logger.info(f"Model loaded successfully: {self.model_info}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Using untrained model.")
                self.model_info = "Warning: Model weights not found. Generate them by running Cell 26 in the notebook."
                self.generator.eval()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self.model_info = f"Error loading model: {str(e)}"
            raise
    
    @torch.no_grad()
    def generate(
        self,
        num_images: int,
        seed: int,
        temperature: float
    ) -> torch.Tensor:
        """
        Generate digit images.
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            temperature: Temperature for sampling diversity
            
        Returns:
            Generated images as tensor (N, 1, 28, 28)
        """
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate noise
        noise = torch.randn(num_images, NOISE_DIM).to(self.device) * temperature
        
        # Generate images
        generated = self.generator(noise)
        
        # Reshape to image format (N, 1, 28, 28)
        generated = generated.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        
        return generated.cpu()


def validate_inputs(
    num_images: int,
    seed: int,
    temperature: float
) -> Tuple[int, int, float]:
    """
    Validate and clamp input parameters.
    
    Args:
        num_images: Requested number of images
        seed: Random seed
        temperature: Sampling temperature
        
    Returns:
        Validated (num_images, seed, temperature)
    """
    num_images = max(MIN_IMAGES, min(MAX_IMAGES, int(num_images)))
    seed = max(0, int(seed))
    temperature = max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, float(temperature)))
    
    return num_images, seed, temperature


def create_image_grid(
    images: np.ndarray,
    num_images: int
) -> Image.Image:
    """
    Create a grid visualization of generated images.
    
    Args:
        images: Array of images (N, 1, 28, 28)
        num_images: Number of images to display
        
    Returns:
        PIL Image containing the grid
    """
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = np.clip(images, 0, 1)
    
    # Determine grid dimensions
    n_cols = min(4, num_images)
    n_rows = (num_images + n_cols - 1) // n_cols
    
    # Create figure
    fig_width = n_cols * 2
    fig_height = n_rows * 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        facecolor='white'
    )
    
    # Handle single image case
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot images
    for idx in range(num_images):
        axes[idx].imshow(images[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[idx].axis('off')
        axes[idx].set_title(f'Sample {idx+1}', fontsize=10, pad=5)
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(pad=0.5)
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)  # Important: close figure to free memory
    
    return image


def generate_digits(
    num_images: int,
    seed: int,
    temperature: float
) -> Image.Image:
    """
    Main generation function for Gradio interface.
    
    Args:
        num_images: Number of digits to generate (1-16)
        seed: Random seed for reproducibility
        temperature: Diversity control (0.5-2.0)
        
    Returns:
        PIL Image containing generated digits
    """
    try:
        # Validate inputs
        num_images, seed, temperature = validate_inputs(num_images, seed, temperature)
        
        logger.info(
            f"Generating {num_images} images with seed={seed}, "
            f"temperature={temperature:.2f}"
        )
        
        # Generate images
        images = model_manager.generate(num_images, seed, temperature)
        
        # Create visualization
        image_grid = create_image_grid(images.numpy(), num_images)
        
        logger.info("Generation completed successfully")
        return image_grid
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        # Return error image
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5, 0.5,
            f"Error: {str(e)}\nPlease check logs.",
            ha='center', va='center',
            fontsize=12, color='red'
        )
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        error_img = Image.open(buf)
        plt.close(fig)
        return error_img


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="MNIST GAN Digit Generator",
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px !important}"
    ) as interface:
        
        # Header
        gr.Markdown(
            """
            # MNIST Digit Generator
            ## Generative Adversarial Network for Handwritten Digit Synthesis
            
            This application demonstrates a trained Generative Adversarial Network (GAN) that synthesizes 
            realistic handwritten digits from random noise vectors. The model was trained on the MNIST 
            dataset for 200 epochs and achieves high-quality, diverse digit generation.
            """
        )
        
        # Model status
        gr.Markdown(f"**Model Status:** {model_manager.model_info}")
        
        # Main interface
        with gr.Row():
            # Controls column
            with gr.Column(scale=1):
                gr.Markdown("### Generation Parameters")
                
                num_images = gr.Slider(
                    minimum=MIN_IMAGES,
                    maximum=MAX_IMAGES,
                    value=9,
                    step=1,
                    label="Number of Samples",
                    info=f"Generate {MIN_IMAGES}-{MAX_IMAGES} digit samples"
                )
                
                seed = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    value=DEFAULT_SEED,
                    step=1,
                    label="Random Seed",
                    info="Set seed for reproducible generation"
                )
                
                temperature = gr.Slider(
                    minimum=MIN_TEMPERATURE,
                    maximum=MAX_TEMPERATURE,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="Controls output diversity (higher = more diverse)"
                )
                
                generate_btn = gr.Button(
                    "Generate Digits",
                    variant="primary",
                    size="lg"
                )
            
            # Output column
            with gr.Column(scale=2):
                output_image = gr.Image(
                    label="Generated Digits",
                    type="pil",
                    height=400
                )
        
        # Example configurations
        gr.Markdown("### Example Configurations")
        gr.Examples(
            examples=[
                [4, 42, 1.0],
                [9, 123, 1.0],
                [16, 777, 0.8],
                [6, 999, 1.5],
            ],
            inputs=[num_images, seed, temperature],
            outputs=output_image,
            fn=generate_digits,
            cache_examples=True
        )
        
        # Technical details (collapsible)
        with gr.Accordion("Model Architecture & Performance", open=False):
            gr.Markdown(
                """
                ### Generator Network
                - **Input:** 100-dimensional random noise vector (sampled from N(0,1))
                - **Architecture:** Fully connected network with 4 layers
                  - Layer 1: 100 → 256 (LeakyReLU, BatchNorm)
                  - Layer 2: 256 → 512 (LeakyReLU, BatchNorm)
                  - Layer 3: 512 → 1024 (LeakyReLU, BatchNorm)
                  - Output: 1024 → 784 (Tanh)
                - **Output:** 28×28 grayscale image
                - **Parameters:** 1,489,936 trainable parameters
                
                ### Training Configuration
                - **Dataset:** MNIST (60,000 training samples)
                - **Duration:** 200 epochs
                - **Optimizer:** Adam (lr=0.0002, β₁=0.5)
                - **Loss:** Binary Cross-Entropy
                - **Final Generator Loss:** 0.972
                - **Final Discriminator Loss:** 1.213
                
                ### Model Performance
                - Stable convergence with no mode collapse
                - Generates diverse, realistic handwritten digits
                - Trained on CPU (approximately 2-3 hours)
                """
            )
        
        # Footer
        gr.Markdown(
            """
            ---
            **Technology Stack:** PyTorch 2.0.1 | Gradio 4.44.0 | Python 3.9+
            
            **Project Resources:** 
            - Training Notebook: `GAN_MNIST_Assignment.ipynb`
            - Model Weights: `generator_model.pth`
            - GitHub: [View Repository](https://github.com/vikranth1000/mnist-gan)
            """
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_digits,
            inputs=[num_images, seed, temperature],
            outputs=output_image
        )
        
        # Auto-generate on load
        interface.load(
            fn=generate_digits,
            inputs=[
                gr.Number(value=9, visible=False),
                gr.Number(value=DEFAULT_SEED, visible=False),
                gr.Number(value=1.0, visible=False)
            ],
            outputs=output_image
        )
    
    return interface


# Initialize model manager
logger.info("Initializing application...")
model_manager = ModelManager()

# Create interface
demo = create_interface()

if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
