import os
import shutil
import cv2
import argparse
import logging
from tqdm import tqdm
from dataset_loader import read_csv_labels, build_id_to_path
try:
   from utils.logger import get_logger, setup_logging  # type: ignore
except Exception:
   try:
      from src.utils.logger import get_logger, setup_logging  # type: ignore
   except Exception:
      get_logger = logging.getLogger  # type: ignore
      def setup_logging(*args, **kwargs):  # type: ignore
         logger = logging.getLogger()
         if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
         return logger

def offline_clahe(data_dir: str, out_dir: str) -> None:
   """
   Applies CLAHE to all images in the dataset and saves them to a new directory.
   Args:
       data_dir (str): Path to the original data directory containing 'train', 'val', 'test' subdirectories.
       out_dir (str): Path to the output directory where processed images will be saved.
   """
   logger = get_logger(__name__)
   if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      logger.info("Created output directory: %s", out_dir)
   
   for split in ['train', 'validation', 'test']:
      in_path = os.path.join(data_dir, split)
      out_path = os.path.join(out_dir, split)
      if not os.path.exists(out_path):
         os.makedirs(out_path)
         logger.info("Created split directory: %s", out_path)
      csv_in = os.path.join(data_dir, f"{split}.csv")
      rows = read_csv_labels(csv_in)
      id_to_path = build_id_to_path(in_path)
      logger.info("Processing %d images for split=%s", len(rows), split)
      for row in tqdm(rows, desc=f"Processing {split} set with CLAHE"):
         image_id = row["image_id"]
         src = id_to_path[image_id]
         if src is None or not os.path.exists(src):
            logger.warning("Image %s not found in %s. Skipping.", image_id, in_path)
            continue
         image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
         if image is None:
            logger.warning("Failed to read image %s. Skipping.", src)
            continue
         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
         out = clahe.apply(image)
         cv2.imwrite(os.path.join(out_path, f"{image_id}.png"), out)
      
      # Copy CSV as is (same ids), images are now in out_path
      shutil.copy(csv_in, os.path.join(out_dir, f"{split}.csv"))
      logger.info("Copied CSV for split=%s to %s", split, out_dir)
      
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Apply CLAHE to dataset images offline.")
   parser.add_argument("--data_dir", type=str, required=True, 
                       default="data/raw", help="Path to the original data directory.")
   parser.add_argument("--out_dir", type=str, required=True, 
                       default="data/proceesed/with_clahe", help="Path to the output directory for CLAHE processed images.")
   parser.add_argument("--log_dir", type=str, default="experiments/logs", help="Directory for log files.")
   parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG).")
   args = parser.parse_args()

   setup_logging(log_dir=args.log_dir, level=args.log_level)
   offline_clahe(args.data_dir, args.out_dir)
      
