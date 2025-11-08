import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
from BAP.utils.dataset_loader import load_image_original


# Function to visualize sample images from the dataset
def display_sample_images(
   metadata: pd.DataFrame, 
   image_dir: Path, n_samples=4, target_size=256):
   sample_metadata = metadata.sample(n_samples)
   plt.figure(figsize=(15, 5))
   for i, (idx, row) in enumerate(sample_metadata.iterrows()):
      image_id = row['Image ID']
      image_path = os.path.join(image_dir, f'{image_id}.png')
      image = load_image_original(image_path)
      plt.subplot(1, n_samples, i + 1)
      plt.imshow(image)
      boneage = row['Bone Age (months)'] 
      plt.title(f"Image ID: {image_id}\nBone Age: {boneage} months\nGender: {'Male' if row['male'] else 'Female'}")
      plt.axis('off')
   plt.show()