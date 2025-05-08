import torch
import random
from torch.utils.data import random_split
from dataset import Flickr30k  # Your dataset
from models import ViTImageCaptioningModel
from transformers import CLIPProcessor
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load full dataset
    print("Loading dataset...")
    dataset = Flickr30k()
    
    # Split into train and test (same as in your training script)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # Load model
    print("Loading model...")
    clip_model_name = "openai/clip-vit-base-patch32"
    model = ViTImageCaptioningModel(clip_model_name=clip_model_name).to(device)

    
    # Download model from Hugging Face
    print("Downloading model from Hugging Face...")
    model_file = hf_hub_download(
        repo_id="Apples96/MLI_Week4_multimodal_transformer_imgcaptions",
        filename="models/image_caption_model_20250508Epoch3.pt"
    )
    print(f"Model downloaded to: {model_file}")

    # Load model
    clip_model_name = "openai/clip-vit-base-patch32"
    model = ViTImageCaptioningModel(clip_model_name=clip_model_name).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    # Load processor
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    # Pick a random image from test set
    random_idx = random.randint(0, len(test_dataset) - 1)
    sample = test_dataset[random_idx]
    image = sample["image"]
    true_caption = sample["caption"]
    
    print(f"Selected random image with true caption: {true_caption}")
    
    # Make sure we have a PIL image for display
    if isinstance(image, torch.Tensor):
        # If this is a tensor, we need to convert it to a PIL image for later
        # We'll process with CLIP first to avoid modifying the original
        pil_image_for_display = None  # We'll create this later if needed
    else:
        # It's already a PIL image
        pil_image_for_display = image.copy()
    
    # Setup for generation
    max_length = 76
    bos_token_id = 49406
    eos_token_id = 49407

    # Process image with CLIP processor
    inputs = processor(images=image, 
        do_rescale=False,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )
                       
    pixel_values = inputs["pixel_values"].to(device)
    
    # Create input with BOS token followed by padding EOS tokens
    input_ids = torch.full((1, max_length), eos_token_id, dtype=torch.long, device=device)
    input_ids[:, 0] = bos_token_id  # Set first token to BOS
    padding_mask = torch.ones((1, max_length), device=device)
    
    # Generate caption with greedy search
    generated_sequence = [bos_token_id]  # Start with BOS token
    
    print("Generating caption...")
    with torch.no_grad():
        for i in range(1, max_length):
            # Forward pass
            caption_output_logits = model(pixel_values, input_ids, padding_mask)
            
            # Get predicted token
            next_token_logits = caption_output_logits[:, i-1, :]  # i-1 because output doesn't include image tokens
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            # Add to generated sequence
            generated_sequence.append(next_token)
            
            # Update input_ids
            input_ids[0, i] = next_token
            
            # Stop at EOS token
            if next_token == eos_token_id:
                break
    
    # Decode the generated tokens
    generated_caption = processor.tokenizer.decode(generated_sequence, skip_special_tokens=True)
    
    # Print results
    print("\nResults:")
    print(f"True caption: {true_caption}")
    print(f"Generated caption: {generated_caption}")
    
    # If we don't have a PIL image yet and the image is a tensor, try to convert it
    if pil_image_for_display is None and isinstance(image, torch.Tensor):
        try:
            # If tensor is in format [C, H, W], transpose to [H, W, C]
            if image.dim() == 3 and image.shape[0] == 3:
                # Scale to 0-255 range if needed
                if image.max() <= 1.0:
                    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                else:
                    img_np = image.permute(1, 2, 0).cpu().numpy().astype('uint8')
                pil_image_for_display = Image.fromarray(img_np)
            else:
                print("Image tensor format not supported for display")
        except Exception as e:
            print(f"Error converting tensor to PIL image: {e}")
    
    # Try to save the image with captions
    if pil_image_for_display is not None:
        try:
            # Create a new image with space for text
            width, height = pil_image_for_display.size
            new_img = Image.new('RGB', (width, height + 100), color=(255, 255, 255))
            new_img.paste(pil_image_for_display, (0, 0))
            
            # Add text
            draw = ImageDraw.Draw(new_img)
            try:
                # Try to use a nice font if available
                font = ImageFont.truetype("Arial", 16)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            draw.text((10, height + 10), f"True: {true_caption}", fill=(0, 0, 0), font=font)
            draw.text((10, height + 50), f"Generated: {generated_caption}", fill=(0, 0, 0), font=font)
            
            # Save the image
            new_img.save("caption_result.png")
            print("Visualization saved as caption_result.png")
        except Exception as e:
            print(f"Error saving visualization: {e}")

if __name__ == "__main__":
    main()