import os
import shutil
import cv2
import argparse
from tqdm import tqdm
from dataset_loader import read_csv_labels, build_id_to_path

def offline_clahe(data_dir: str, out_dir: str) -> None:
   """
   Applies CLAHE to all images in the dataset and saves them to a new directory.
   Args:
       data_dir (str): Path to the original data directory containing 'train', 'val', 'test' subdirectories.
       out_dir (str): Path to the output directory where processed images will be saved.
   """
   if not os.path.exists(out_dir):
      os.makedirs(out_dir)
   
   for split in ['train', 'validation', 'test']:
      in_path = os.path.join(data_dir, split)
      out_path = os.path.join(out_dir, split)
      if not os.path.exists(out_path):
         os.makedirs(out_path)
      csv_in = os.path.join(data_dir, f"{split}.csv")
      rows = read_csv_labels(csv_in)
      id_to_path = build_id_to_path(in_path)
      for row in tqdm(rows, desc=f"Processing {split} set with CLAHE"):
         image_id = row["image_id"]
         src = id_to_path[image_id]
         if src is None or not os.path.exists(src):
            print(f"Warning: Image {image_id} not found in {in_path}. Skipping.")
            continue
         image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
         if image is None:
            print(f"Warning: Failed to read image {src}. Skipping.")
            continue
         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
         out = clahe.apply(image)
         cv2.imwrite(os.path.join(out_path, f"{image_id}.png"), out)
      
      # Copy CSV as is (same ids), images are now in out_path
      shutil.copy(csv_in, os.path.join(out_dir, f"{split}.csv"))
      
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Apply CLAHE to dataset images offline.")
   parser.add_argument("--data_dir", type=str, required=True, 
                       default="data/raw", help="Path to the original data directory.")
   parser.add_argument("--out_dir", type=str, required=True, 
                       default="data/proceesed/with_clahe", help="Path to the output directory for CLAHE processed images.")
   args = parser.parse_args()
   
   offline_clahe(args.data_dir, args.out_dir)
      