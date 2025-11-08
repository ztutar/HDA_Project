from typing import Optional
import argparse
import os

from BAP.training.train_GlobalCNN import train_GlobalCNN
from BAP.training.train_ROI_CNN import train_ROI_CNN
from BAP.training.train_Fusion_CNN import train_FusionCNN

#from BAP.utils.logger import setup_logging
from BAP.utils.config import load_config
from BAP.utils.seeds import set_seeds
from BAP.utils.path_manager import incremental_path
from BAP.utils.dataset_loader import get_rsna_dataset


# Map user-friendly aliases to canonical model metadata (name, trainer, default config)
MODEL_REGISTRY = {
   "globalcnn": ("GlobalCNN", train_GlobalCNN),
   "global_cnn": ("GlobalCNN", train_GlobalCNN),
   "global": ("GlobalCNN", train_GlobalCNN),
   "roi_cnn": ("ROI_CNN", train_ROI_CNN),
   "roicnn": ("ROI_CNN", train_ROI_CNN),
   "roi": ("ROI_CNN", train_ROI_CNN),
   "fusion_cnn": ("Fusion_CNN", train_FusionCNN),
   "fusioncnn": ("Fusion_CNN", train_FusionCNN),
   "fusion": ("Fusion_CNN", train_FusionCNN),
}


def main(model_name: str, config_path: Optional[str]) -> None:

   # Check if the model is supported
   normalized_name = model_name.strip().lower()
   if normalized_name not in MODEL_REGISTRY:
      #logger.error("Model '%s' is not supported.", model_name)
      raise ValueError(f"Unsupported model: {model_name}")
   
   # Retrieve model name and training function
   canonical_name, train_fn = MODEL_REGISTRY[normalized_name]
   
   # Download dataset
   paths = get_rsna_dataset(force_download=False)

   # Setup output directory for the experiment
   config_filename = os.path.basename(config_path) if config_path else "default"
   save_dir = incremental_path(
      save_dir="experiments/checkpoints",
      model_name=canonical_name,
      config_name=os.path.splitext(config_filename)[0],
   )
   
   # Setup logging
   #logger = setup_logging(log_dir=save_dir)
   
   # Load the configuration file or use default
   config_bundle = load_config(config_path)
      
   # Set random seeds for reproducibility
   set_seeds()
   
   # Start training
   #logger.info("Starting training for model: %s, using config: %s", canonical_name, config_filename)
   train_fn(paths, config_bundle, save_dir)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Train a bone age prediction model.")
   parser.add_argument(
      "--model",
      type=str,
      required=True,
      help="Name of the model to train (e.g., 'GlobalCNN' or 'ROI_CNN')."
   )
   parser.add_argument(
      "--config",
      type=str,
      default=None,
      help="Path to the YAML/JSON config file. If not provided, default settings are used."
   )
   args = parser.parse_args()
   main(args.model, args.config)
