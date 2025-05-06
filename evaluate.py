import torch
import random
from torch.utils.data import Subset, random_split
from models import ViTImageCaptioningModel
from dataset import Flickr30k
from transformers import CLIPProcessor, CLIPTokenizer
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import os
from tqdm import tqdm

# Reuse your existing process_batch function
def process_batch(batch, max_text_length=77):
    """Process a batch to get tokenized inputs and outputs for training"""
    clip_model_name="openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    images = batch["image"]
    captions = batch["caption"]
    
    # Process with CLIP processor
    tokenized_inputs = processor(
        images=images,
        text=captions,
        do_rescale=False,
        return_tensors="pt",
        padding="max_length",
        max_length=max_text_length,
        truncation=True
    )
    
    # Move to device if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
    
    # Split input_ids for teacher forcing (input is all but last, target is all but first)
    tokenized_images = tokenized_inputs["pixel_values"]
    input_ids = tokenized_inputs["input_ids"]
    tokenized_input_captions = input_ids[:, :-1]  # Remove last token (for input)
    tokenized_output_captions = input_ids[:, 1:]  # Remove first token (for target)
    
    return tokenized_images, tokenized_input_captions, tokenized_output_captions

def generate_caption(model, image, processor, tokenizer, max_length=20):
    """Generate a caption for an image using the trained model"""
    device = next(model.parameters()).device
    
    # Create a batch with just one image
    batch = {"image": [image], "caption": ["[PAD]"]}  # Dummy caption
    
    # Get tokenized image using your existing process_batch function
    tokenized_images, _, _ = process_batch(batch)
    
    # Start with BOS token
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    
    # Generate tokens auto-regressively
    for _ in range(max_length):
        # Forward pass
        output_logits = model(tokenized_images, input_ids)
        
        # Get the prediction for the last token
        next_token_logits = output_logits[0, -1, :]
        
        # Get most likely token
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        
        # Add to input sequence
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        
        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    
    # Decode the caption
    caption = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return caption

def run_inference(model_path, num_samples=10, seed=42):
    """Run inference on a selection of images from the test dataset"""
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Load dataset
    dataset = Flickr30k()
    
    # Split into train and test (using the same split logic as in training)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # Select a subset of test images
    if num_samples < len(test_dataset):
        indices = random.sample(range(len(test_dataset)), num_samples)
        test_samples = Subset(test_dataset, indices)
    else:
        test_samples = test_dataset
    
    # Load model
    clip_model_name = "openai/clip-vit-base-patch32"
    model = ViTImageCaptioningModel(clip_model_name=clip_model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load tokenizer for decoding
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    
    # Create results directory
    os.makedirs("inference_results", exist_ok=True)
    
    # Store results
    results = []
    
    # Generate captions
    print(f"Generating captions for {len(test_samples)} images...")
    for idx, sample in enumerate(tqdm(test_samples)):
        image = sample["image"]
        gt_caption = sample["caption"]
        
        # Generate caption
        with torch.no_grad():
            predicted_caption = generate_caption(model, image, processor, tokenizer)
        
        # Save result
        results.append({
            "idx": idx,
            "image": image,
            "ground_truth": gt_caption,
            "prediction": predicted_caption
        })
    
    # Visualize results
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, result in enumerate(results):
        # Display image
        ax = axes[idx]
        ax.imshow(to_pil_image(result["image"]))
        ax.set_title(f"Image {idx + 1}")
        
        # Add captions below image
        ax.text(0, -10, f"Predicted: {result['prediction']}", fontsize=12, wrap=True)
        ax.text(0, -30, f"Ground Truth: {result['ground_truth']}", fontsize=12, wrap=True)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("inference_results/caption_comparison.png")
    plt.close()
    
    # Print results
    print("\nInference Results:")
    for idx, result in enumerate(results):
        print(f"\nImage {idx + 1}:")
        print(f"Predicted: {result['prediction']}")
        print(f"Ground Truth: {result['ground_truth']}")
    
    print(f"\nResults visualization saved to inference_results/caption_comparison.png")

if __name__ == "__main__":
    model_path = "models/best_image_caption_model.pt"
    run_inference(model_path, num_samples=10, seed=42)