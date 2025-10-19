from typing import Optional
import argparse
from BoneAgePrediction.training.train_B0 import train_GlobalCNN
from BoneAgePrediction.utils.logger import get_logger

def main(model_name: str, config_path: Optional[str]) -> None:
   """
   Main function to initiate training of the specified model with given configuration.
   Args:
      model_name: Name of the model to train (e.g., "GlobalCNN").
      config_path: Path to the configuration file. If None, default settings are used.
   """
   logger = get_logger(__name__)
   logger.info("Starting training for model: %s", model_name)
   
   if model_name == "GlobalCNN":
      train_GlobalCNN(config_path or "global_only.yaml")
   else:
      logger.error("Model '%s' is not supported.", model_name)
      raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Train a bone age prediction model.")
   parser.add_argument(
      "--model",
      type=str,
      required=True,
      help="Name of the model to train (e.g., 'GlobalCNN')."
   )
   parser.add_argument(
      "--config",
      type=str,
      default=None,
      help="Path to the YAML/JSON config file. If not provided, default settings are used."
   )
   args = parser.parse_args()
   main(args.model, args.config)