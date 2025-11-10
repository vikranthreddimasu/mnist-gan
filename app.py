"""
MNIST GAN Digit Generator Application
A production-ready Gradio interface for generating handwritten digits using a trained GAN.

Author: Vikranth Reddimasu
License: MIT
"""

import logging
import random
import sys
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server deployment
import matplotlib.pyplot as plt
from PIL import Image
import io
try:
    from gradio_client import utils as gradio_client_utils
except ImportError:
    gradio_client_utils = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Patch gradio_client schema helpers to handle boolean schemas (older versions crash)
if gradio_client_utils is not None:
    try:
        _original_get_type = gradio_client_utils.get_type
        _original_schema_converter = gradio_client_utils._json_schema_to_python_type  # noqa: SLF001

        def _safe_get_type(schema):
            if isinstance(schema, bool):
                return "bool" if schema else "none"
            return _original_get_type(schema)

        def _safe_schema_converter(schema, defs=None):
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            return _original_schema_converter(schema, defs)

        gradio_client_utils.get_type = _safe_get_type
        gradio_client_utils._json_schema_to_python_type = _safe_schema_converter  # noqa: SLF001
        logger.info("Patched gradio_client utils to handle boolean JSON schemas")
    except Exception as patch_error:
        logger.warning("Could not patch gradio_client schema helpers: %s", patch_error)

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
HERO_TEXT = """
<h1 style="margin-bottom:0.3rem;">MNIST Digit Generator</h1>
<p style="font-size:1.05rem;margin:0;color:var(--body-text-color);">
Explore a trained GAN that synthesizes handwriting on demand. Set your parameters, tap generate, and download a fresh grid of digits.
</p>
"""
METRIC_CARDS = [
    ("Noise Dim", f"{NOISE_DIM}", "Latent space size"),
    ("Hidden Units", f"{HIDDEN_DIM}", "Per-layer width"),
    ("Epochs", "200", "Training duration"),
    ("Temp Range", f"{MIN_TEMPERATURE} – {MAX_TEMPERATURE}", "Diversity control"),
    ("Generator Loss", "0.981", "Final epoch"),
    ("Params", "1.49M", "Trainable weights"),
]
STYLE_PRESETS = [
    ("Precise", "Clean, focused digits", 6, 123, 0.8),
    ("Balanced", "Default training vibe", 9, 512, 1.0),
    ("Playful", "Add variety + contrast", 12, 777, 1.3),
]

CUSTOM_CSS = """
.gradio-container {
  max-width: 1100px !important;
  margin: auto;
  background: radial-gradient(circle at top, #111827, #0b1120 45%, #05070f 90%);
  color: #F8FAFC;
}
#hero {
  padding: 0.5rem 0 1rem 0;
}
#hero h1 {
  font-size: 2.2rem;
  font-weight: 700;
  color: #F8FAFC;
}
#hero p {
  color: #CBD5F5;
}
.surface {
  background: rgba(15, 23, 42, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 14px;
  padding: 1.2rem;
  box-shadow: 0 18px 40px rgba(2, 6, 23, 0.6);
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}
.metric-card {
  border-radius: 12px;
  padding: 0.8rem 1rem;
  background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(147,197,253,0.08));
  border: 1px solid rgba(59, 130, 246, 0.25);
}
.metric-label {
  margin: 0;
  font-size: 0.8rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #2563EB;
}
.metric-value {
  margin: 0.15rem 0 0 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: #F8FAFC;
}
.metric-hint {
  margin: 0.1rem 0 0 0;
  font-size: 0.8rem;
  color: rgba(248, 250, 252, 0.75);
}
.pill-row > button {
  flex: 1;
}
.tip-text {
  font-size: 0.92rem;
  color: var(--body-text-color-subdued);
  color: rgba(226, 232, 240, 0.9);
}
.quick-presets .table {
  background: rgba(15, 23, 42, 0.65);
}
.gr-box {
  background: rgba(15, 23, 42, 0.8) !important;
}
.style-chip {
  border-radius: 999px;
  border: 1px solid rgba(59,130,246,0.5);
  padding: 0.65rem 1rem;
  background: rgba(30,64,175,0.15);
  color: #E0E7FF;
  font-weight: 600;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}
.style-chip span {
  font-size: 0.8rem;
  font-weight: 400;
  color: rgba(226,232,240,0.8);
}
"""


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
            # Resolve absolute path for better debugging
            abs_path = self.model_path.resolve()
            logger.info(f"Loading model from {abs_path}")
            logger.info(f"Current working directory: {Path.cwd()}")
            logger.info(f"Model file exists: {self.model_path.exists()}")
            
            # Initialize generator
            self.generator = Generator(
                noise_dim=NOISE_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=OUTPUT_DIM
            ).to(self.device)
            logger.info("Generator initialized successfully")
            
            # Load checkpoint if available
            if self.model_path.exists():
                logger.info(f"Loading checkpoint from {abs_path}")
                checkpoint = torch.load(
                    str(self.model_path),  # Convert Path to string for torch.load
                    map_location=self.device,
                    weights_only=True
                )
                logger.info("Checkpoint loaded, extracting state dict...")
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.generator.eval()
                logger.info("Model state dict loaded successfully")
                
                epochs = checkpoint.get('epoch', 'N/A')
                g_loss = checkpoint.get('generator_loss', 'N/A')
                
                self.model_info = (
                    f"Model trained for {epochs} epochs | "
                    f"Generator Loss: {g_loss:.4f}" if isinstance(g_loss, float) 
                    else f"Model trained for {epochs} epochs"
                )
                logger.info(f"Model loaded successfully: {self.model_info}")
            else:
                logger.warning(f"Model file not found at {abs_path}. Using untrained model.")
                logger.warning(f"Files in current directory: {list(Path('.').glob('*.pth'))}")
                self.model_info = "Warning: Model weights not found. Using untrained model."
                self.generator.eval()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self.model_info = f"Error loading model: {str(e)}"
            # Don't raise - let the app start and show error in UI
            logger.error("Model loading failed, but continuing to start app...")
    
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
        if model_manager is None:
            raise RuntimeError("Model not loaded. Please check the logs for details.")
        
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


def _random_seed() -> int:
    """Return a random seed for the shuffle button."""
    return random.randint(0, 10000)


def _preset_values(num_images: int, seed: int, temperature: float) -> Tuple[int, int, float]:
    """Return preset tuple for slider updates."""
    return num_images, seed, temperature


def generate_comparison(
    num_images: int,
    seed_left: int,
    temperature_left: float,
    seed_right: int,
    temperature_right: float
) -> Tuple[Image.Image, Image.Image]:
    """Generate two grids for side-by-side comparison."""
    left = generate_digits(num_images, seed_left, temperature_left)
    right = generate_digits(num_images, seed_right, temperature_right)
    return left, right


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate"
    )
    
    with gr.Blocks(
        title="MNIST GAN Digit Generator",
        theme=theme,
        css=CUSTOM_CSS
    ) as interface:
        
        gr.Markdown(HERO_TEXT, elem_id="hero")
        
        with gr.Row(elem_classes="metric-grid"):
            for label, value, hint in METRIC_CARDS:
                gr.HTML(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value'>{value}</div>
                        <div class='metric-hint'>{hint}</div>
                    </div>
                    """
                )
        
        model_status = model_manager.model_info if model_manager else "Model unavailable"
        
        with gr.Tabs():
            with gr.Tab("Playground"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes="surface"):
                        gr.Markdown("### Controls")
                        
                        num_images = gr.Slider(
                            minimum=MIN_IMAGES,
                            maximum=MAX_IMAGES,
                            value=9,
                            step=1,
                            label="Samples",
                            info=f"Generate {MIN_IMAGES}-{MAX_IMAGES} digits"
                        )
                        
                        seed = gr.Slider(
                            minimum=0,
                            maximum=10000,
                            value=DEFAULT_SEED,
                            step=1,
                            label="Seed",
                            info="Same seed → same grid"
                        )
                        
                        temperature = gr.Slider(
                            minimum=MIN_TEMPERATURE,
                            maximum=MAX_TEMPERATURE,
                            value=1.0,
                            step=0.1,
                            label="Temperature",
                            info="Higher = wilder digits"
                        )
                        
                        gr.Markdown("#### Style presets")
                        with gr.Row():
                            preset_buttons = []
                            for label, desc, n_val, seed_val, temp_val in STYLE_PRESETS:
                                btn = gr.Button(
                                    f"{label} · {desc}",
                                    elem_classes="style-chip",
                                    variant="secondary"
                                )
                                btn.click(
                                    fn=lambda n=n_val, s=seed_val, t=temp_val: _preset_values(n, s, t),
                                    outputs=[num_images, seed, temperature],
                                    queue=False
                                )
                                preset_buttons.append(btn)
                        
                        with gr.Row(elem_classes="pill-row"):
                            random_btn = gr.Button("Shuffle Seed", variant="secondary")
                            generate_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column(scale=2, elem_classes="surface"):
                        gr.Markdown("### Output")
                        output_image = gr.Image(
                            label="",
                            type="pil",
                            height=430,
                            show_download_button=True,
                            interactive=False
                        )
                        gr.Markdown(
                            f"**Model status:** {model_status}",
                            elem_classes="tip-text"
                        )
                        gr.Markdown(
                            "Pro tip: Lower temperatures focus on clean digits, while higher values explore creative shapes.",
                            elem_classes="tip-text"
                        )
                
                with gr.Group(elem_classes="surface quick-presets"):
                    gr.Markdown("### Quick Presets")
                    gr.Examples(
                        label="Pick a vibe",
                        examples=[
                            [4, 21, 0.9],
                            [9, 512, 1.0],
                            [12, 777, 1.3],
                            [16, 999, 0.7],
                        ],
                        inputs=[num_images, seed, temperature],
                        outputs=output_image,
                        fn=generate_digits,
                        cache_examples=True
                    )
            
            with gr.Tab("Compare"):
                gr.Markdown("Explore two parameter sets side-by-side to understand how seeds and temperature shape samples.")
                num_images_cmp = gr.Slider(
                    minimum=MIN_IMAGES,
                    maximum=12,
                    value=6,
                    step=1,
                    label="Samples per grid"
                )
                with gr.Row(equal_height=True):
                    with gr.Column(elem_classes="surface"):
                        gr.Markdown("#### Left grid")
                        seed_left = gr.Slider(0, 10000, 42, step=1, label="Seed")
                        temp_left = gr.Slider(MIN_TEMPERATURE, MAX_TEMPERATURE, 0.9, step=0.1, label="Temperature")
                    with gr.Column(elem_classes="surface"):
                        gr.Markdown("#### Right grid")
                        seed_right = gr.Slider(0, 10000, 777, step=1, label="Seed")
                        temp_right = gr.Slider(MIN_TEMPERATURE, MAX_TEMPERATURE, 1.3, step=0.1, label="Temperature")
                compare_btn = gr.Button("Generate Comparison", variant="primary")
                with gr.Row():
                    compare_left = gr.Image(type="pil", height=360, label="Left output", show_download_button=True)
                    compare_right = gr.Image(type="pil", height=360, label="Right output", show_download_button=True)
            
            with gr.Tab("Insights"):
                gr.Markdown(
                    "Training insights highlight how the GAN converged. These snapshots come directly from the notebook used to build the weights."
                )
                gr.Image(
                    value=str(Path("losses.png")),
                    label="Training loss curves",
                    show_download_button=True
                )
                gr.Markdown("**Key checkpoints**")
                gr.Dataframe(
                    headers=["Metric", "Value", "Notes"],
                    value=[
                        ["Generator Loss", "0.981", "Converged after steady decline"],
                        ["Discriminator Loss", "1.213", "Balanced with generator"],
                        ["Best Epoch", "200", "Used for deployment weights"],
                    ],
                    interactive=False,
                    wrap=True
                )
                gr.Markdown(
                    "Want to dig deeper? Clone the repo and open `GAN_MNIST_Assignment.ipynb` to replay the full experiment."
                )
        
        with gr.Accordion("What powers this demo?", open=False):
            gr.Markdown(
                """
                - **Generator:** Fully-connected GAN with 1.49M parameters (100-dim noise → 28×28 image).
                - **Training:** MNIST, 200 epochs, Adam (lr=2e-4), final generator loss ≈ 0.97.
                - **Deployment:** PyTorch 2.0.1 + Gradio 4.44 running on CPU in Hugging Face Spaces.
                """
            )
        
        random_btn.click(
            fn=_random_seed,
            outputs=seed,
            queue=False
        )
        
        generate_btn.click(
            fn=generate_digits,
            inputs=[num_images, seed, temperature],
            outputs=output_image
        )
        
        compare_btn.click(
            fn=generate_comparison,
            inputs=[num_images_cmp, seed_left, temp_left, seed_right, temp_right],
            outputs=[compare_left, compare_right]
        )
        
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


# Initialize model manager with error handling
logger.info("Initializing application...")
try:
    model_manager = ModelManager()
except Exception as e:
    logger.error(f"Failed to initialize model manager: {str(e)}", exc_info=True)
    # Create a dummy model manager that will show error in UI
    model_manager = None

# Create interface
demo = create_interface()

if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
