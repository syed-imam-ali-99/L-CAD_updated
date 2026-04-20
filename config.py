import yaml
import os
from pathlib import Path

class Config:
    """Configuration loader for L-CAD project"""

    def __init__(self, config_path='config.yaml'):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path

        # Check if config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create a config.yaml file or copy from config.yaml.example"
            )

        # Load YAML configuration
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to config value (e.g., 'models.init_model')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    # Model paths
    @property
    def init_model_path(self):
        return self.get('models.init_model')

    @property
    def resume_checkpoint(self):
        return self.get('models.resume_checkpoint')

    @property
    def largedecoder_checkpoint(self):
        return self.get('models.largedecoder_checkpoint')

    # Config files
    @property
    def cldm_v15_config(self):
        return self.get('configs.cldm_v15')

    @property
    def cldm_sample_config(self):
        return self.get('configs.cldm_sample')

    # Dataset paths
    @property
    def coco_img_dir(self):
        return self.get('datasets.coco.img_dir')

    @property
    def coco_caption_dir(self):
        return self.get('datasets.coco.caption_dir')

    @property
    def example_img_dir(self):
        return self.get('datasets.example.img_dir')

    @property
    def example_caption_dir(self):
        return self.get('datasets.example.caption_dir')

    @property
    def example_test_pairs(self):
        return self.get('datasets.example.test_pairs')

    @property
    def sam_caption_dir(self):
        return self.get('datasets.sam.caption_dir')

    @property
    def sam_pairs_json(self):
        return self.get('datasets.sam.pairs_json')

    @property
    def sam_select_masks_dir(self):
        return self.get('datasets.sam.select_masks_dir')

    # Training hyperparameters
    @property
    def n_gpu(self):
        return self.get('training.n_gpu', 2)

    @property
    def batch_size(self):
        return self.get('training.batch_size', 16)

    @property
    def logger_freq(self):
        return self.get('training.logger_freq', 1000)

    @property
    def learning_rate_multiplier(self):
        return self.get('training.learning_rate_multiplier', 1e-5)

    @property
    def sd_locked(self):
        return self.get('training.sd_locked', False)

    @property
    def only_mid_control(self):
        return self.get('training.only_mid_control', False)

    @property
    def num_workers(self):
        return self.get('training.num_workers', 0)

    @property
    def precision(self):
        return self.get('training.precision', 32)

    # Testing hyperparameters
    @property
    def test_batch_size(self):
        return self.get('testing.batch_size', 1)

    @property
    def test_num_workers(self):
        return self.get('testing.num_workers', 0)

    @property
    def test_precision(self):
        return self.get('testing.precision', 32)

    @property
    def test_n_gpu(self):
        return self.get('testing.n_gpu', 1)

    # Inference hyperparameters
    @property
    def ddim_steps(self):
        return self.get('inference.ddim_steps', 50)

    @property
    def ddim_eta(self):
        return self.get('inference.ddim_eta', 0.0)

    @property
    def unconditional_guidance_scale(self):
        return self.get('inference.unconditional_guidance_scale', 5.0)

    @property
    def use_attn_guidance(self):
        return self.get('inference.use_attn_guidance', True)

    # Dataset settings
    @property
    def img_size(self):
        return self.get('dataset.img_size', 256)

    @property
    def norm_mean(self):
        return self.get('dataset.norm_mean', [0.5, 0.5, 0.5])

    @property
    def norm_std(self):
        return self.get('dataset.norm_std', [0.5, 0.5, 0.5])

    # Output paths
    @property
    def image_log_dir(self):
        return self.get('output.image_log_dir', './image_log')

    @property
    def test_output_template(self):
        return self.get('output.test_output_template', './image_log/test_{timestamp}')

    # Memory optimization
    @property
    def save_memory(self):
        return self.get('memory.save_memory', False)


# Create global config instance
try:
    cfg = Config('config.yaml')
    save_memory = cfg.save_memory
except FileNotFoundError:
    # Fallback to default values if config file doesn't exist
    print("Warning: config.yaml not found. Using default values.")
    save_memory = False
    cfg = None
