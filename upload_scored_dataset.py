import os
from datasets import load_dataset
from huggingface_hub import HfApi

# --- Configuration ---
# 1. Define the target repository on the Hugging Face Hub.
HF_USERNAME = "Thermostatic"
DATASET_NAME = "CCMatrix-English-Spanish-1Sub-Scored"
FULL_REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

# 2. The name of the local Parquet file generated by your scoring script.
LOCAL_PARQUET_FILENAME = "ccmatrix-en-es-scored.parquet"

# 3. Define the maximum size for each uploaded file shard.
#    The format "500MB" is understood by the `push_to_hub` function.
MAX_SHARD_SIZE = "500MB"


def upload_to_hub():
    """
    Loads a scored Parquet file, converts it to a Hugging Face Dataset,
    and uploads it to the Hub with sharding.
    """
    # --- 1. Pre-flight Checks ---
    print("--- Starting Upload Process ---")

    if not os.path.exists(LOCAL_PARQUET_FILENAME):
        print(f"❌ Error: The file '{LOCAL_PARQUET_FILENAME}' was not found.")
        print("Please ensure you have run the scoring script first and the output file is in the same directory.")
        return
    
    print(f"✅ Found local file: '{LOCAL_PARQUET_FILENAME}'")

    # --- 2. Load the Parquet data into a Hugging Face Dataset ---
    print(f"Loading data from '{LOCAL_PARQUET_FILENAME}' into a Dataset object...")
    try:
        # Using `data_files` argument to specify the Parquet file
        scored_dataset_dict = load_dataset("parquet", data_files=LOCAL_PARQUET_FILENAME)
        # The dataset object is inside the 'train' split by default
        scored_dataset = scored_dataset_dict['train']
    except Exception as e:
        print(f"❌ Error loading Parquet file: {e}")
        return

    print("✅ Dataset loaded successfully. Features:")
    print(scored_dataset.features)
    print(f"Number of rows: {len(scored_dataset)}")
    
    # --- 3. Push the dataset to the Hub with Sharding ---
    # The `push_to_hub` method handles the conversion and upload.
    # We add `max_shard_size` to split the dataset into multiple smaller files.
    # It will also create the repository if it doesn't exist.
    print(f"\n🚀 Uploading dataset to the Hugging Face Hub...")
    print(f"   Target repository: {FULL_REPO_ID}")
    print(f"   Max shard size:    {MAX_SHARD_SIZE}")

    try:
        scored_dataset.push_to_hub(
            repo_id=FULL_REPO_ID,
            commit_message=f"Upload scored {DATASET_NAME} dataset with sharding",
            private=False,
            max_shard_size=MAX_SHARD_SIZE # This is the key argument for sharding
        )
        print("\n--- Upload Complete! ---")
        print(f"✅ Successfully uploaded the sharded dataset.")
        print(f"You can view your new dataset repository at:")
        print(f"   https://huggingface.co/datasets/{FULL_REPO_ID}")

    except Exception as e:
        print(f"\n❌ An error occurred during the upload process.")
        print(f"   Please ensure you are logged in to the Hugging Face CLI.")
        print(f"   You can log in by running: huggingface-cli login")
        print(f"   Original error: {e}")


if __name__ == "__main__":
    # Before running, make sure you are logged in via the terminal:
    # huggingface-cli login
    upload_to_hub()
