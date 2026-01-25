import os
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

def download_dtd_textures(output_dir="data/broden/dtd", textures=["striped", "dotted"]):
    """Download striped and dotted textures from DTD dataset."""
    print("Downloading DTD dataset...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the DTD dataset
    dataset = load_dataset("cansa/Describable-Textures-Dataset-DTD", split="train")
    
    # Counter for each texture type
    counters = {texture: 0 for texture in textures}
    
    # Iterate through dataset and filter for desired textures
    for item in tqdm(dataset, desc="Processing DTD images"):
        label = item['label']  # The texture category
        
        if label in textures:
            # Get the image
            image = item['image']
            
            # Save with numbered filename
            filename = f"{label}_{counters[label]:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Convert to RGB if necessary and save
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(filepath)
            
            counters[label] += 1
    
    print("\nDTD Download complete!")
    for texture, count in counters.items():
        print(f"  {texture}: {count} images")
    
    return counters


def download_imagenet_zebras(output_dir="data/imagenet/zebra", zebra_label=340):
    """Download zebra images from ImageNet (label 340)."""
    print("\nDownloading ImageNet zebra images...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load ImageNet dataset
    # Note: You may need authentication for ImageNet
    try:
        train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
        val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        print("Note: ImageNet requires authentication. You may need to:")
        print("1. Accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("2. Login with: huggingface-cli login")
        return
    
    train_count = 0
    val_count = 0
    
    # Process training set
    print("Processing training set...")
    for i, item in enumerate(train_dataset):
        if item['label'] == zebra_label:
            image = item['image']
            filename = f"n02391049_train_{train_count:04d}.JPEG"
            filepath = os.path.join(output_dir, filename)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(filepath, 'JPEG')
            
            train_count += 1
        
        # Optional: limit number of images to download
        if i > 100000:  # Adjust this limit as needed
            break
    
    # Process validation set
    print("Processing validation set...")
    for item in val_dataset:
        if item['label'] == zebra_label:
            image = item['image']
            filename = f"n02391049_val_{val_count:04d}.JPEG"
            filepath = os.path.join(output_dir, filename)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(filepath, 'JPEG')
            
            val_count += 1
    
    print("\nImageNet Download complete!")
    print(f"  Training zebras: {train_count} images")
    print(f"  Validation zebras: {val_count} images")
    
    return train_count, val_count


def main():
    """Main function to download all datasets."""
    print("=" * 60)
    print("Dataset Download Script")
    print("=" * 60)
    
    # Download DTD textures
    dtd_counts = download_dtd_textures(
        output_dir="data/broden/dtd",
        textures=["striped", "dotted"]
    )
    
    # Download ImageNet zebras
    imagenet_counts = download_imagenet_zebras(
        output_dir="data/imagenet/zebra"
    )

    print(f"Downloaded {dtd_counts}")
    print(f"Downloaded {imagenet_counts}")
    
    print("\n" + "=" * 60)
    print("All downloads complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
