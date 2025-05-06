import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from torchvision.transforms.functional import to_pil_image


class Flickr30k(Dataset):
    def __init__(self, split='test'):
        """
        Memory-efficient Flickr30k dataset that loads data on demand
        
        Args:
            split: Dataset split ('train', 'test', 'validation')
            cache_dir: Optional directory to cache HuggingFace datasets
        """
        print(f"Initializing Flickr30k dataset...")
        
        # Load the dataset from HuggingFace but don't keep it all in memory
        self.dataset = load_dataset("nlphuji/flickr30k", split=split)
        
        # Prepare indices mapping (for expanded captions)
        self.indices_mapping = []
        
        # Build mapping from global index to (image_idx, caption_idx)
        for img_idx in range(len(self.dataset)):
            num_captions = len(self.dataset[img_idx]['caption'])
            for cap_idx in range(num_captions):
                self.indices_mapping.append((img_idx, cap_idx))
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        print(f"Dataset initialized with {len(self.indices_mapping)} image-caption pairs")
    
    def __len__(self):
        return len(self.indices_mapping)
    
    def __getitem__(self, idx):
        # Get the image and caption indices
        img_idx, cap_idx = self.indices_mapping[idx]
        
        # Retrieve the item from the dataset
        item = self.dataset[img_idx]
        image = item['image']
        caption = item['caption'][cap_idx]
        
        # Apply transformation to image
        image = self.transform(image)
        
        return {
            "image": image,
            "caption": caption
        }

# class Flickr30k_old(Dataset):
#     def __init__(self, sample_size=None):
#         processed_path = f"data/Flickr30K_processed.pkl"
        
#         if os.path.isfile(processed_path):
#             print("Loading Flickr30k dataset from local file...")
#             with open(processed_path, 'rb') as f:
#                 self.processed_dataset = pickle.load(f)
#         else:
#             print("Loading Flickr30k dataset from Hugging Face...")
#             self.raw_dataset = load_dataset("nlphuji/flickr30k")['test']

#             # Expand the dataset to include all image-caption pairs
#             self.processed_dataset = []
            
#             for item in self.raw_dataset:
#                 image = item['image']
#                 captions = item['caption']
                
#                 # Create one example for each caption
#                 for caption in captions:
#                     self.processed_dataset.append((image, caption))


#             print(f"Saving {len(self.processed_dataset)} processed image-caption pairs...")
#             with open(processed_path, 'wb') as f:
#                 pickle.dump(self.processed_dataset, f)
    
#     def __len__(self):
#         return len(self.processed_dataset)
    
#     def __getitem__(self, idx):
#         image, caption = self.processed_dataset[idx]
#         transform = transforms.ToTensor()
#         image = transform(image)
#         return {
#         "image": image,
#         "caption": caption
#         }


if __name__ == "__main__":    
    image_caption_dataset = Flickr30k()
    
    dataloader = DataLoader(image_caption_dataset, batch_size=32, shuffle=True)

    for idx, batch in enumerate(dataloader):
        images = batch["image"]
        captions = batch["caption"]
        print(f"\nSample {idx}:")
        print(f"Caption example: {captions[0]}")
        print(f"Image example: {images[0]}")
        print(f"Image shape example: {images[0].shape}")

        # Save the image sample
        sample_path = f"data/samples/flickr30k_sample_{idx}.png"
        to_pil_image(images[0]).save(sample_path)
        print(f"Saved image to {sample_path}")


