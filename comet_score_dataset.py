import torch
import pandas as pd
from datasets import load_dataset
from comet import download_model, load_from_checkpoint
import argparse
import os
import time

def score_en_es_dataset(args):
    """
    Loads a dataset in a memory-efficient way, scores it in batches using a COMET model,
    and saves the results to a Parquet file.
    """
    # --- 1. Download and Load the COMET Model ---
    # This is done first to ensure the model is ready and to fail fast if there's an issue.
    print(f"Downloading and loading model: {args.model_name}...")
    try:
        model_path = download_model(args.model_name, saving_directory=args.model_cache_dir)
        # On a multi-GPU system, you might want to specify the device.
        # For a single massive GPU, letting COMET manage it is fine.
        model = load_from_checkpoint(model_path)
        
        # PyTorch 2.0+ feature for significant speed-up.
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()... (This may take a few minutes)")
            model = torch.compile(model)
            
    except Exception as e:
        print(f"Error loading model. Please ensure you have accepted the license on Hugging Face")
        print(f"and are logged in via 'huggingface-cli login'.")
        print(f"Original error: {e}")
        return

    # --- 2. Load the Hugging Face Dataset (in streaming mode or memory-mapped) ---
    print(f"Loading the dataset: {args.dataset_name}...")
    # For massive datasets, streaming=True can be useful, but .map() already handles
    # batching, so standard loading is fine and often faster for processing.
    # The `num_proc` will significantly speed up the initial download and preparation.
    dataset = load_dataset(args.dataset_name, split="train", num_proc=args.num_cpu_workers)

    # Limit dataset to max_samples for faster processing
    LIMIT = min(args.max_samples, len(dataset))
    print(f"Limiting dataset to first {LIMIT} examples...")
    dataset = dataset.select(range(LIMIT))

    # --- 3. Define the Batch Scoring Function ---
    # This is the core of the memory-efficient approach.
    # The .map() function will feed this function batches of the dataset.
    def score_batch(batch):
        """
        Takes a batch from the dataset, formats it for COMET,
        and returns the scores.
        """
        # The 'batch' object is a dictionary where keys are column names
        # and values are lists of the data for that batch.
        data_to_score = [
            {"src": src, "mt": mt}
            for src, mt in zip(batch[args.src_column], batch[args.mt_column])
        ]
        
        # We don't need to specify gpus here because COMET's trainer will
        # automatically use the available GPU.
        model_output = model.predict(data_to_score, batch_size=args.batch_size, gpus=args.num_gpus)
        
        # Return a dictionary with the new column name and the scores.
        # This is the format required by datasets.map().
        return {"comet_score": model_output.scores}

    # --- 4. Score the Translations using .map() ---
    # This is the main processing step. It iterates through the dataset in batches,
    # applies our `score_batch` function, and builds the new dataset on disk.
    num_gpus = args.num_gpus if torch.cuda.is_available() else 0
    print(f"\nScoring {len(dataset)} sentence pairs...")
    print(f"Using {num_gpus} GPUs and a COMET internal batch size of {args.batch_size}.")
    print(f"Dataset.map will process in chunks of size {args.map_batch_size}.")

    start_time = time.time()
    
    # Using `with_rank=True` would be for multi-process logging, but the GPU is the bottleneck.
    # The progress bar from .map() is sufficient.
    scored_dataset = dataset.map(
        score_batch,
        batched=True,
        batch_size=args.map_batch_size, # How many rows to pass to score_batch at a time
        cache_file_name=os.path.join(args.cache_dir, "ccmatrix_en_es_scored.arrow"), # Caches progress
        num_proc=1 # Set to 1 because the bottleneck is the single GPU
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Scoring Complete in {duration / 3600:.2f} hours ---")

    # --- 5. Display and Save Results ---
    print("Top 5 scored sentence pairs:")
    # We need to specify the column names as they are in the final dataset
    for i in range(5):
        example = scored_dataset[i]
        print(f"\nExample {i+1}:")
        print(f"  English (src): {example[args.src_column]}")
        print(f"  Spanish (mt):  {example[args.mt_column]}")
        print(f"  COMET Score:   {example['comet_score']:.4f}")

    # Save the entire scored dataset to a Parquet file for analysis
    print(f"\nSaving scored dataset to '{args.output_file}'...")
    scored_dataset.to_parquet(args.output_file)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Score a large translation dataset with COMET.")
    
    # --- Dataset and Model Arguments ---
    parser.add_argument("--dataset_name", type=str, default="Thermostatic/CCMatrix-English-Spanish", help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--src_column", type=str, default="English", help="Name of the source language column.")
    parser.add_argument("--mt_column", type=str, default="Spanish", help="Name of the target language column.")
    parser.add_argument("--model_name", type=str, default="Unbabel/wmt22-cometkiwi-da", help="Name of the COMET model on Hugging Face Hub.")
    
    # --- Hardware and Performance Arguments ---
    # For your 180GB VRAM, you can use a very large batch size.
    # Start high and decrease if you get a CUDA Out of Memory error.
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for the COMET model prediction (fits in VRAM).")
    parser.add_argument("--map_batch_size", type=int, default=65536, help="Batch size for the datasets.map() function (number of rows to process per chunk).")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use. Defaults to all available.")
    parser.add_argument("--num_cpu_workers", type=int, default=os.cpu_count() // 2, help="Number of CPU workers for loading data.")
    
    # --- I/O Arguments ---
    parser.add_argument("--output_file", type=str, default="ccmatrix-en-es-scored.parquet", help="Path to save the final scored Parquet file.")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", help="Directory for caching models and datasets.")
    parser.add_argument("--model_cache_dir", type=str, default="./model_cache", help="Directory for caching COMET models.")
    
    # --- New: Dataset Size Limit ---
    parser.add_argument("--max_samples", type=int, default=40_000, help="Maximum number of samples to score from the dataset.")

    args = parser.parse_args()

    # Create cache directories if they don't exist
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.model_cache_dir, exist_ok=True)

    print("--- Configuration ---")
    for key, value in vars(args).items():
        print(f"{key:<20}: {value}")
    print("---------------------")

    score_en_es_dataset(args)

if __name__ == "__main__":
    main()
