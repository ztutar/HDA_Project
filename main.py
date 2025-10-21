from typing import Optional
import argparse
import os
from BoneAgePrediction.training.train_B0 import train_GlobalCNN
from BoneAgePrediction.training.train_R1 import train_ROI_CNN
from BoneAgePrediction.utils.logger import get_logger, setup_logging

# Map user-friendly aliases to canonical model metadata (name, trainer, default config)
MODEL_REGISTRY = {
   "globalcnn": ("GlobalCNN", train_GlobalCNN, "global_only.yaml"),
   "b0": ("GlobalCNN", train_GlobalCNN, "global_only.yaml"),
   "roi_cnn": ("ROI_CNN", train_ROI_CNN, "roi_only.yaml"),
   "roicnn": ("ROI_CNN", train_ROI_CNN, "roi_only.yaml"),
   "roi": ("ROI_CNN", train_ROI_CNN, "roi_only.yaml"),
   "r1": ("ROI_CNN", train_ROI_CNN, "roi_only.yaml"),
}


def main(model_name: str, config_path: Optional[str]) -> None:
   """
   Main function to initiate training of the specified model with given configuration.
   Args:
      model_name: Name of the model to train (e.g., "GlobalCNN").
      config_path: Path to the configuration file. If None, default settings are used.
   """
   setup_logging(log_dir=os.path.join("experiments", "logs"))
   normalized_name = model_name.strip().lower()
   logger = get_logger(__name__)

   if normalized_name not in MODEL_REGISTRY:
      logger.error("Model '%s' is not supported.", model_name)
      raise ValueError(f"Unsupported model: {model_name}")

   canonical_name, train_fn, default_config = MODEL_REGISTRY[normalized_name]
   logger.info("Starting training for model: %s", canonical_name)

   train_fn(config_path or default_config)


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
