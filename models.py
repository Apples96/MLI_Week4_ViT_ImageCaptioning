import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F



class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, seq_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim)
        self.mask = torch.tril(torch.ones(seq_length, seq_length)) # Lower triangular matrix of ones (including the diagonal)
    def forward (self, x):
        batch_size, seq_len, _ = x.shape
         
        # Apply projections while keeping batch dimension
        query_emb = self.Wq(x) # (batch_size, seq_len, embed_dim) 
        key_emb = self.Wk(x) # ((batch_size, seq_len, embed_dim)
        value_emb = self.Wv(x) # (batch_size, seq_len, embed_dim)
        # For debugging, print the shapes
        sims = query_emb @ key_emb.transpose(1, 2) / (self.embed_dim ** 0.5) # (batch_size, seq_len, embed_dim) @ (batch_size, embed_dim, seq_len) > (batch_size, seq_len, seq_len)
        masked_sims = sims.masked_fill(self.mask == 0, -1e9)
        scaled_masked_sims = F.softmax(masked_sims, dim = 2) # (batch_size, seq_len, seq_len)
        x = scaled_masked_sims @ value_emb # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, embed_dim) > (batch_size, seq_len, embed_dim)
        return x



class ViTImageCaptioningModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", sequence_length=76, num_patches = 50, vocab_size=49408, num_decoder_layers=2, img_embed_dim = 768, text_embed_dim = 512):
        super().__init__()
        self.sequence_length = sequence_length # after preprocessing includes 76 tokens incl either BOS or EOS (we take off one of them)
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers
        self.num_patches = num_patches # includes 49 patches and 1 CLS token
        self.num_tokens = self.sequence_length + self.num_patches # 126
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        self.img_embed_dim = self.clip_model.config.vision_config.hidden_size  # Usually 768
        self.text_embed_dim = self.clip_model.config.text_config.hidden_size  # Usually 512

        # Freeze CLIP parameters (optional but recommended initially)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.layer_norm = nn.LayerNorm(self.text_embed_dim)
        self.masked_attention = MaskedSelfAttention(self.text_embed_dim, self.num_tokens)
        self.ff0 = nn.Linear(self.img_embed_dim, self.text_embed_dim) 
        self.ff1 = nn.Linear(self.text_embed_dim, self.text_embed_dim) 
        self.ff2 = nn.Linear(self.text_embed_dim, self.vocab_size) 
        
    def forward(self, tokenized_images, tokenized_input_captions):
        
        
        # Tokenize and preprocess raw images to be passes to CLIP embedding layers. Resizes images to (224, 224), converts to RGB, adds BOS and EOS tokens at start and finish of captions. 
        # Processor output keys: dict_keys(['input_ids', 'attention_mask', 'pixel_values']). 
        # Pixel values shape : [batch_size, 3, 224, 224]
        # Input IDs shape : [batch_size, seq_length]
        # Attention_mask shape : (num_patches (50), img_embed_dim (768)) > gets projected over batch dimension

        # Generate image embeddings (incl positional embeddings)and bring them down to embed_size of 512 to match text embeddings (with a FF layer)
        images_embeddings = self.clip_model.vision_model.embeddings(tokenized_images)
        # print(f"Images embeddings shape before resizing:{images_embeddings[0].shape}. Should be (batch_size, num_patches (50), img_embed_dim (768)")

        resized_images_embeddings = self.ff0(images_embeddings)
        # print(f"Images embeddings shape after resizing:{resized_images_embeddings[0].shape}. Should be (batch_size, num_patches (50), embed_dim (512)")
        
        # Generate caption embeddings (incl positional embeddings)
        input_captions_embeddings = self.clip_model.text_model.embeddings(input_ids=tokenized_input_captions)
        # print(f"Captions embeddings shape:{input_captions_embeddings.shape}. Should be (batch_size, seq_length (76), embed_dim (512))")

        # Concatenate embeddings to get intput emebddings (patches embeddings + SOS + work embeddings) and output emebddings (patches embeddings + work embeddings + EOS)
        input_embeddings = torch.cat([resized_images_embeddings, input_captions_embeddings], dim=1)
        # print(f"Input embeddings shape:{input_embeddings.shape}. Should be (batch_size, seq_length+num_patches(126), embed_dim (512))")

        x = self.layer_norm(input_embeddings)
        # print (x.shape)
        
        
        # Loop over decoder block for num_decoder_layers times
        for i in range(self.num_decoder_layers):
            residuals1 = x # (batch_size, 126, 512)
            # if i == 1:
            #     print(residuals1.shape)
            
            x = self.masked_attention(x) # (batch_size, 126, 512)
            # if i == 1:
            #     print (x.shape)
            
            x = x + residuals1 # (batch_size, 126, 512)
            # if i == 1:
            #     print (x.shape)
            
            x = self.layer_norm(x) # (batch_size, 126, 512)
            # if i == 1:
            #     print (x.shape)
            
            residuals2 = x # (batch_size, 126, 512)
            # if i == 1:
            #     print (residuals2.shape)

            x = self.ff1(x) # (batch_size, 126, 512) @ (batch_size512*512) + (batch_size, 126, 512) > (batch_size, 126, 512)
            # if i == 1:
            #     print (x.shape)
            

            x = x + residuals2 # (batch_size, 126, 512)
            # if i == 1:
            #     print (x.shape)
            
            x = self.layer_norm(x) # (batch_size, 126, 512)
            # if i == 1:
            #     print (x.shape)
        
        output_logits = self.ff2(x)  # (batch_size, 126, 512) @ (batch_size, 512, 49408) > (batch_size, 126, 49408)
        caption_output_logits = output_logits[:, 50:, :]  # Should be [batch_size, 76, 49408]
        # print (f"caption_output_logits.shape : {caption_output_logits.shape}")
        return caption_output_logits # (batch_size, 76, 49408)






