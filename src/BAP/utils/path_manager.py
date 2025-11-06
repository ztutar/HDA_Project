
import os

def incremental_path(save_dir: str, model_name: str = None, config_name: str = None) -> str:
   """
   Create a unique directory path by appending an incremental number if needed.
   For example, if save_dir is "experiments", model_name is "model1", and config_name is 
   "config1", it will create "experiments/model1/config1/model1_config1_01", or 
   "experiments/model1/config1/model1_config1_02" if the first already exists, and so on.
   Args:
      save_dir (str): The base directory where the new folder will be created.
      model_name (str): The name of the model to include in the folder name. Optional.
      config_name (str): The base name for the new folder. Optional.
   Returns:
      str: The path to the newly created unique directory.
   """
   # Define the top-level folder based on the save_dir and configuration name.
   head_folder = os.path.join(save_dir, model_name, config_name)
   os.makedirs(head_folder, exist_ok=True)  # Ensure the top-level folder exists.

   # Loop to find a unique folder name by appending an incremental number.
   for n in range(1, 99):
      save_folder = os.path.join(head_folder, f"{model_name}_{config_name}_{n:02d}")  # Construct folder name with zero padding.
      if not os.path.exists(save_folder):  # Check if the folder already exists.
         os.makedirs(save_folder)  # Create the folder if it doesn't exist.
         return save_folder  # Return the unique folder path.

   # If the loop exceeds the limit, raise an error (unlikely in practice).
   raise RuntimeError(f"Too many folders created for {config_name}")
