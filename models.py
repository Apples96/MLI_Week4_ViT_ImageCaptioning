import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F



class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, image_seq_length, text_seq_length, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.image_seq_length = image_seq_length
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.Wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)  # Output projection

        seq_length = image_seq_length + text_seq_length
        # Create base causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_length, seq_length))

        # Modify the mask to allow full attention to image tokens
        # This makes the first 50 columns (image token positions) all ones
        causal_mask[:, :image_seq_length] = 1

        self.register_buffer("mask", causal_mask)
    
    def forward (self, x, input_padding_mask=None):
        batch_size, seq_len, _ = x.shape
         
        # Apply projections while keeping batch dimension
        query_emb = self.Wq(x) # (batch_size, seq_len, embed_dim) 
        # print(f"query_emb.shape {query_emb.shape}")
        key_emb = self.Wk(x) # ((batch_size, seq_len, embed_dim)
        # print(f"key_emb.shape {key_emb.shape}")
        value_emb = self.Wv(x) # (batch_size, seq_len, embed_dim)
        # print(f"value_emb.shape {value_emb.shape}")

        # Reshape for multi-head attention
        # Split embed_dim into num_heads and head_dim
        query_emb = query_emb.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key_emb = key_emb.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        value_emb = value_emb.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
       


        # For debugging, print the shapes
        sims = query_emb @ key_emb.transpose(-2, -1) / (self.head_dim ** 0.5) # (batch_size, seq_len, embed_dim) @ (batch_size, embed_dim, seq_len) > (batch_size, seq_len, seq_len)
        # print(f"sims.shape {sims.shape}")
        mask = self.mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        masked_sims = sims.masked_fill(mask == 0, -1e9)
        # print(f"masked_sims.shape {masked_sims.shape}")

        # Additionally apply padding mask if provided
        if input_padding_mask is not None:
            # Create attention-compatible padding mask 
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            image_padding = torch.ones(
                (batch_size, self.image_seq_length), 
                device=input_padding_mask.device
            )
            
            # Concatenate to create full padding mask
            full_padding_mask = torch.cat([image_padding, input_padding_mask], dim=1)

            full_padding_mask = full_padding_mask.unsqueeze(1).unsqueeze(2)
            
            # Extend across attention dimension
            # [batch_size, 1, 1, seq_len] -> [batch_size, 1, seq_len, seq_len]
            extended_padding_mask = full_padding_mask.repeat(1, 1, seq_len, 1)
            
            # Combine with causal mask
            masked_sims = masked_sims.masked_fill(extended_padding_mask == 0, -1e9)

        scaled_masked_sims = F.softmax(masked_sims, dim = -1) # (batch_size, num_heads, seq_len, seq_len)
        x = scaled_masked_sims @ value_emb # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, embed_dim) > (batch_size, seq_len, embed_dim)
        # Reshape back: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # Apply output projection
        x = self.Wo(x)
       
        return x



class ViTImageCaptioningModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", text_seq_length=76, image_seq_length = 50, vocab_size=49408, num_decoder_layers=6, img_embed_dim = 768, text_embed_dim = 512, num_heads=8):
        super().__init__()
        self.text_seq_length = text_seq_length # after preprocessing includes 76 tokens incl either BOS or EOS (we take off one of them)
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers
        self.image_seq_length = image_seq_length # 50 - includes 49 patches and 1 CLS token
        self.num_tokens = self.text_seq_length + self.image_seq_length # 126
        self.num_heads = num_heads #8
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        self.img_embed_dim = self.clip_model.config.vision_config.hidden_size  # Usually 768
        self.text_embed_dim = self.clip_model.config.text_config.hidden_size  # Usually 512

        # Freeze CLIP parameters (optional but recommended initially)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.layer_norm = nn.LayerNorm(self.text_embed_dim)
        self.masked_attention = MaskedSelfAttention(self.text_embed_dim, self.image_seq_length, self.text_seq_length, num_heads=self.num_heads)
        self.ff0 = nn.Linear(self.img_embed_dim, self.text_embed_dim) 
        self.ff1 = nn.Linear(self.text_embed_dim, self.text_embed_dim) 
        self.ff2 = nn.Linear(self.text_embed_dim, self.vocab_size) 
        
    def forward(self, tokenized_images, tokenized_input_captions, input_padding_mask=None):
        
        
        # Tokenize and preprocess raw images to be passes to CLIP embedding layers. Resizes images to (224, 224), converts to RGB, adds BOS and EOS tokens at start and finish of captions. 
        # Processor output keys: dict_keys(['input_ids', 'attention_mask', 'pixel_values']). 
        # Pixel values shape : [batch_size, 3, 224, 224]
        # Input IDs shape : [batch_size, seq_length]
        # Attention_mask shape : (num_patches (50), img_embed_dim (768)) > gets projected over batch dimension

        # Generate image embeddings (incl positional embeddings)and bring them down to embed_size of 512 to match text embeddings (with a FF layer)
        vision_outputs = self.clip_model.vision_model(tokenized_images)
        images_embeddings = vision_outputs.last_hidden_state
        # print(f"Images embeddings shape before resizing:{images_embeddings.shape}. Should be (batch_size, num_patches (50), img_embed_dim (768)")

        resized_images_embeddings = self.ff0(images_embeddings)
        # print(f"Images embeddings shape after resizing:{resized_images_embeddings.shape}. Should be (batch_size, num_patches (50), embed_dim (512)")
        
        # Generate caption embeddings (incl positional embeddings)
        input_captions_embeddings = self.clip_model.text_model.embeddings(input_ids=tokenized_input_captions)
        # print(f"Captions embeddings shape:{input_captions_embeddings.shape}. Should be (batch_size, seq_length (76), embed_dim (512))")

        # Concatenate embeddings to get intput emebddings (patches embeddings + SOS + work embeddings) and output emebddings (patches embeddings + work embeddings + EOS)
        input_embeddings = torch.cat([resized_images_embeddings, input_captions_embeddings], dim=1)
        # print(f"Input embeddings shape:{input_embeddings.shape}. Should be (batch_size, seq_length+num_patches(126), embed_dim (512))")

        x = self.layer_norm(input_embeddings)
        
        
        # Loop over decoder block for num_decoder_layers times
        for i in range(self.num_decoder_layers):
            residuals1 = x # (batch_size, 126, 512)
            # if i == 0:
            #     print(f"residuals1.shape {residuals1.shape}")
            
            x = self.masked_attention(x, input_padding_mask) # (batch_size, 126, 512)
            # if i == 0:
            #     print (f"x after masked attention {x.shape}")
            
            x = x + residuals1 # (batch_size, 126, 512)
            # if i == 0:
            #     print (f"x after residual connection with residuals 1 {x.shape}")
            
            x = self.layer_norm(x) # (batch_size, 126, 512)
            # if i == 0:
            #     print (f"x after layer norm dec 1 {x.shape}")
            
            residuals2 = x # (batch_size, 126, 512)
            # if i == 0:
            #     print (f"residuals 2 {residuals2.shape}")

            x = self.ff1(x) # (batch_size, 126, 512) @ (batch_size512*512) + (batch_size, 126, 512) > (batch_size, 126, 512)
            # if i == 0:
            #     print (f"x after ff1 {x.shape}")
            

            x = x + residuals2 # (batch_size, 126, 512)
            # if i == 0:
            #     print (f"x after residual connection with residuals 2 {x.shape}")
            
            x = self.layer_norm(x) # (batch_size, 126, 512)
            # if i == 0:
            #     print (f"x after layer norm dec 2 {x.shape}")
        
        output_logits = self.ff2(x)  # (batch_size, 126, 512) @ (batch_size, 512, 49408) > (batch_size, 126, 49408)
        # print (f"output_logits.shape : {output_logits.shape}")
        caption_output_logits = output_logits[:, self.image_seq_length:, :]  # Should be [batch_size, 76, 49408]
        # print (f"outputted caption_output_logits.shape : {caption_output_logits.shape}")
        return caption_output_logits # (batch_size, 76, 49408)






