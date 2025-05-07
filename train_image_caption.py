import torch
import random
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import Adam
from dataset import Flickr30k
import wandb
import os
from models import ViTImageCaptioningModel
from transformers import CLIPProcessor
from tqdm import tqdm


def sample_dataset(dataset, max_samples=100, random_seed=42):
    """Create a smaller subset of the dataset"""
    from torch.utils.data import Subset
    import random
    
    if len(dataset) <= max_samples:
        return dataset
    
    # Set seed for reproducibility
    random.seed(random_seed)

    # Sample random indices
    indices = random.sample(range(len(dataset)), max_samples)
    return Subset(dataset, indices)

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

def train_image_caption(
    clip_model_name="openai/clip-vit-base-patch32",
    batch_size=16,
    num_epochs=1,
    learning_rate=1e-4
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load raw dataset (or subset))
    #dataset = Flickr30k(split='train')
    dataset = sample_dataset(Flickr30k(), max_samples=100)
    
    # Build train and test datasets (80% train, 20% test)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your CPU
        pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model with loaded embedding layers
    model = ViTImageCaptioningModel(clip_model_name=clip_model_name).to(device)

    # Set optimizer and loss criteria
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Typically 0 is padding, adjust if needed
    criterion = criterion.to(device)  # Move criterion to device

    # Set up Weights and Biases tracking
    wandb.init(project="image-captioning", config={
        "model": clip_model_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate
    })
    # Create directory for saving models
    os.makedirs("models", exist_ok=True)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"): 
            # Reset gradients
            optimizer.zero_grad()

            # Preproecss batch
            tokenized_images, tokenized_input_captions, tokenized_output_captions = process_batch(batch)
            if batch_idx == 0:
                print(f"tokenized_images.shape:{tokenized_images.shape}")
                print(f"tokenized_input_captions.shape:{tokenized_input_captions.shape}")
                print(f"tokenized_output_captions.shape:{tokenized_output_captions.shape}")

            # Run forward pass
            caption_output_logits = model(tokenized_images, tokenized_input_captions) # (batch_size, 76, 49408)

            # Reshape logits and targets for loss calculation
            # output_logits: [batch_size, seq_len, vocab_size] → [batch_size * seq_len, vocab_size]
            # target_captions: [batch_size, seq_len] → [batch_size * seq_len]
            reshaped_logits = caption_output_logits.reshape(-1, model.vocab_size)  # [batch_size * seq_len, vocab_size]
            reshaped_targets = tokenized_output_captions.reshape(-1) # [batch_size * seq_len]

            # Calculate loss
            loss = criterion(reshaped_logits, reshaped_targets)

            # Backprop and update gradients
            loss.backward()
            optimizer.step()

            # Track statistics
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_count += 1

            # Print stats every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {current_loss:.4f}")
                wandb.log({"batch_loss": current_loss})

        # Calculate epoch averages
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs} completed, Avg Loss: {avg_epoch_loss:.4f}")
        
        # Evaluate model after each epoch
        model.eval()
        eval_loss = 0
        eval_batch_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):  # Using same data for simplicity, should use validation set
                # Process batch
                # Preproecss batch
                tokenized_images, tokenized_input_captions, tokenized_output_captions = process_batch(batch)
                
                # Get model outputs
                caption_output_logits = model(tokenized_images, tokenized_input_captions)
                
                # Calculate loss
                reshaped_logits = caption_output_logits.reshape(-1, model.vocab_size)  # [batch_size * seq_len, vocab_size]
                reshaped_targets = tokenized_output_captions.reshape(-1) # [batch_size * seq_len]
                loss = criterion(reshaped_logits, reshaped_targets)
                
                eval_loss += loss.item()
                eval_batch_count += 1
        
        avg_eval_loss = eval_loss / eval_batch_count
        print(f"Evaluation Loss: {avg_eval_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_epoch_loss,
            "eval_loss": avg_eval_loss
        })
        
        # Save model if it's better than existing best
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            torch.save(model.state_dict(), f"models/best_image_caption_model.pt")
            print(f"New best model saved with loss: {best_loss:.4f}")
        
    # Final cleanup
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    train_image_caption()








