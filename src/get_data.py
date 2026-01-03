import kagglehub
import shutil
from pathlib import Path

# Download dataset (goes to kagglehub cache)
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print(f"Dataset downloaded to cache: {path}")

# Copy to data/raw/
raw_data_dir = Path("data/raw")
raw_data_dir.mkdir(parents=True, exist_ok=True)

source_path = Path(path)
for file in source_path.glob("*"):
    if file.is_file():
        destination = raw_data_dir / file.name
        shutil.copy2(file, destination)
        print(f"Copied {file.name} to {destination}")

print(f"\nDataset now available in: {raw_data_dir.absolute()}")