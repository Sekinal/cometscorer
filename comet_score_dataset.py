import torch
from datasets import load_dataset
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

def score_en_es_dataset():
    """
    Loads the OPUS-100 EN-ES dataset, scores it with wmt22-cometkiwi-da,
    and saves the results to a CSV file.
    """
    # --- 1. Load the Hugging Face Dataset ---
    print("Loading the OPUS-100 EN-ES dataset...")
    # We will use the 'train' split which is the main part of this dataset.
    # You can also use 'test' or 'validation' if needed.
    dataset = load_dataset("Thermostatic/OPUS-100-EN-ES", split="train")

    # Optional: For quick testing, you can select a smaller subset
    # dataset = dataset.select(range(100))
    
    # --- 2. Prepare the Data for COMET ---
    # The model expects a list of dictionaries with "src" and "mt" keys.
    # "src" is the source language (English)
    # "mt" is the machine translation (Spanish)
    print("Preparing data for scoring...")
    data_to_score = [
        {"src": row["English"], "mt": row["Spanish"]} 
        for row in tqdm(dataset, desc="Formatting data")
    ]

    # --- 3. Download and Load the COMET Model ---
    # This will download the model from Hugging Face Hub.
    # The first time you run this, it will take a while.
    # Subsequent runs will use the cached model.
    model_name = "Unbabel/wmt22-cometkiwi-da"
    print(f"Downloading and loading model: {model_name}...")
    try:
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
    except Exception as e:
        print(f"Error loading model. Please ensure you have accepted the license on Hugging Face")
        print(f"and are logged in via 'huggingface-cli login'.")
        print(f"Original error: {e}")
        return

    # --- 4. Score the Translations ---
    # Set batch_size and GPU usage according to your hardware.
    # Using a GPU (gpus=1) is highly recommended for speed.
    # If no GPU is available, it will automatically fall back to CPU (gpus=0).
    batch_size = 128
    num_gpus = 1 if torch.cuda.is_available() else 0
    
    print(f"Scoring {len(data_to_score)} sentence pairs...")
    print(f"Using {num_gpus} GPUs and batch size of {batch_size}.")

    # The model.predict() method returns a object with scores, system_score, etc.
    # We are interested in the 'scores' list.
    model_output = model.predict(
        data_to_score, 
        batch_size=batch_size, 
        gpus=num_gpus,
        progress_bar=True # Show a progress bar during scoring
    )
    scores = model_output.scores

    # --- 5. Add Scores to the Dataset and Save ---
    print("Adding scores to the dataset...")
    # The 'add_column' method is a convenient way to add the scores list
    # as a new column to our original dataset.
    scored_dataset = dataset.add_column("comet_score", scores)

    # --- 6. Display and Save Results ---
    print("\n--- Scoring Complete ---")
    print("Top 5 scored sentence pairs:")
    for i in range(5):
        example = scored_dataset[i]
        print(f"\nExample {i+1}:")
        print(f"  English (src): {example['English']}")
        print(f"  Spanish (mt):  {example['Spanish']}")
        print(f"  COMET Score:   {example['comet_score']:.4f}")

    # Save the entire scored dataset to a CSV file for analysis
    output_filename = "opus_en_es_wmt22_scored.csv"
    print(f"\nSaving scored dataset to '{output_filename}'...")
    scored_dataset.to_csv(output_filename, index=False)
    print("Done!")


if __name__ == "__main__":
    score_en_es_dataset()