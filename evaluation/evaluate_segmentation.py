import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
from datasets import load_dataset
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from copy import deepcopy


from LayerSync.models import SiT_models
from LayerSync.transport import create_transport
from LayerSync.download import find_model_resume, find_model

try:
    from torchmetrics import JaccardIndex
except ImportError:
    print("Warning: torchmetrics is not installed. mIoU will not be calculated.")
    print("Please install it with: pip install torchmetrics")
    JaccardIndex = None


class LinearSegmentationProbe(nn.Module):
    """
    Simple linear probe for segmentation.
    Just a linear projection: features → segmentation logits.
    """
    def __init__(self, in_channels, num_classes = 21, target_size=256):
        super().__init__()
        self.target_size = target_size
        self.num_classes = num_classes
        # One linear layer to map features to class logits
        self.linear_probe = nn.Linear(in_channels, num_classes)
        
    def forward(self, features):
        B, N, C = features.shape  # [B, 256, 768]
        
        # One linear projection: [B, N, C] → [B, N, num_classes]
        logits = self.linear_probe(features)  # [B, 256, 21]
        
        # Calculate spatial size of patches
        patches_per_side = int(np.sqrt(N))  # √256 = 16
        
        # Reshape to image format: [B, num_classes, H, W]
        logits = logits.permute(0, 2, 1).reshape(B, self.num_classes, patches_per_side, patches_per_side)
        # Result: [B, 21, 16, 16]
        
        # Simple upsampling to target size
        if patches_per_side != self.target_size:
            logits = F.interpolate(
                logits, 
                size=(self.target_size, self.target_size),
                mode='bilinear', 
                align_corners=False
            )
        # Final results : [B, 21, 256, 256]
        
        return logits


class PatchToPixelSegmentationHead(nn.Module):
    """
    Segmentation head that respects the spatial correspondence of patches.
    Each patch is mapped to its correct position in the final image.
    """
    def __init__(self, in_channels, num_classes, patch_pixel_size=16):
        super().__init__()
        self.num_classes = num_classes
        self.patch_pixel_size = patch_pixel_size
        self.pixels_per_patch = patch_pixel_size * patch_pixel_size  # 16×16 = 256
        
        # Projection: each patch (768D) → pixel square (256D)
        self.patch_to_pixels = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, self.pixels_per_patch)  # 768 → 256 (16×16 pixels)
        )
        
        # Classification of each pixel individually
        self.pixel_classifier = nn.Conv2d(1, num_classes, kernel_size=1)

    def forward(self, features):
        B, N, C = features.shape  # [B, 256, 768]
        
        # Calculate patch grid (√256 = 16×16 patches)
        patches_per_side = int(np.sqrt(N))  # 16
        
        # Map each patch to its pixels
        pixels = self.patch_to_pixels(features)  # [B, 256, 256]
        
        pixels = pixels.reshape(B, patches_per_side, patches_per_side, 
                               self.patch_pixel_size, self.patch_pixel_size)
        
   
        # Permute dimensions to assemble correctly:
        # [B, patch_row, patch_col, pixel_row, pixel_col] → [B, patch_row, pixel_row, patch_col, pixel_col]
        pixels = pixels.permute(0, 1, 3, 2, 4)
        
        # Final reshape to full image
        image = pixels.reshape(B, 1, 
                              patches_per_side * self.patch_pixel_size,  # 16 * 16 = 256
                              patches_per_side * self.patch_pixel_size)  # 16 * 16 = 256
        
        # Final classification of each pixel
        segmentation = self.pixel_classifier(image)  # [B, 21, 256, 256]
        
        return segmentation


class SegmentationHead(nn.Module):
    """
    An improved segmentation head with progressive upsampling and non-linearities.
    """
    def __init__(self, in_channels, num_classes, target_size=256, activation_size=16):
        super().__init__()
        self.target_size = target_size
        self.activation_size = activation_size
        
        # Calculate upsampling more intelligently
        self.total_upsample = target_size // activation_size
        
        # If upsampling is too large, do it progressively
        if self.total_upsample >= 8:
            # Progressive upsampling: first ×4, then final interpolation
            hidden_channels = max(in_channels // 2, 128)
            
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.relu1 = nn.ReLU(inplace=True)
            
            # Deconv for upsampler ×4
            self.deconv = nn.ConvTranspose2d(hidden_channels, hidden_channels//2, 
                                           kernel_size=4, stride=4, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channels//2)
            self.relu2 = nn.ReLU(inplace=True)
            
            # Final projection
            self.final_conv = nn.Conv2d(hidden_channels//2, num_classes, kernel_size=1)
            
            # Final upsampling by interpolation
            self.final_upsample = self.total_upsample // 4
            
        else:
            # For smaller factors, simple approach
            hidden_channels = max(in_channels // 2, 64)
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.final_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
            self.deconv = None
            self.final_upsample = self.total_upsample

    def forward(self, features):
        B, N, C = features.shape
        
        # Check that N matches activation_size²
        expected_N = self.activation_size ** 2
        if N != expected_N:
            actual_size = int(np.sqrt(N))
            print(f"⚠️  Unexpected size: {actual_size}×{actual_size} instead of {self.activation_size}×{self.activation_size}")
            self.activation_size = actual_size
            self.total_upsample = self.target_size // actual_size
            self.final_upsample = self.total_upsample if self.deconv is None else self.total_upsample // 4
        
        # Reshape to 2D
        features_2d = features.permute(0, 2, 1).reshape(B, C, self.activation_size, self.activation_size)
        
        # First convolution
        x = self.conv1(features_2d)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Progressive upsampling if necessary
        if self.deconv is not None:
            x = self.deconv(x)
            x = self.bn2(x)
            x = self.relu2(x)
        
        # Final projection to classes
        logits = self.final_conv(x)
        
        # Final upsampling
        if self.final_upsample > 1:
            masks = F.interpolate(
                logits, 
                scale_factor=self.final_upsample, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            masks = logits
            
        return masks


def rgb_to_class(mask):
    """
    Convert a PASCAL VOC RGB mask to class indices.
    The Hugging Face dataset may have slightly different values due to compression/decompression.
    """
    # PASCAL VOC colormap standard
    colormap = np.array([
        [0, 0, 0],        # 0: background
        [128, 0, 0],      # 1: aeroplane
        [0, 128, 0],      # 2: bicycle
        [128, 128, 0],    # 3: bird
        [0, 0, 128],      # 4: boat
        [128, 0, 128],    # 5: bottle
        [0, 128, 128],    # 6: bus
        [128, 128, 128],  # 7: car
        [64, 0, 0],       # 8: cat
        [192, 0, 0],      # 9: chair
        [64, 128, 0],     # 10: cow
        [192, 128, 0],    # 11: diningtable
        [64, 0, 128],     # 12: dog
        [192, 0, 128],    # 13: horse
        [64, 128, 128],   # 14: motorbike
        [192, 128, 128],  # 15: person
        [0, 64, 0],       # 16: pottedplant
        [128, 64, 0],     # 17: sheep
        [0, 192, 0],      # 18: sofa
        [128, 192, 0],    # 19: train
        [0, 64, 128],     # 20: tvmonitor
        [224, 224, 192]   # 255: border/ignore (parfois présent)
    ])
    
    # Initialize with ignore_index
    class_mask = np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.uint8)
    
    # Conversion with tolerance to handle compression artifacts
    for i, color in enumerate(colormap[:-1]):  # Exclude the last color (border)
        # Use Euclidean distance with threshold for robustness
        diff = np.abs(mask.astype(np.int32) - color.astype(np.int32))
        distance = np.sqrt(np.sum(diff**2, axis=-1))
        class_mask[distance < 10] = i  # Tolerance of 10 for artifacts
    
    # Handle border pixels (very high values like [224, 224, 192])
    border_pixels = np.all(mask > 200, axis=-1)
    class_mask[border_pixels] = 255
    
    return class_mask


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, target_transform=None, image_size=256):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        mask = item['mask']
        
        # Resize the image AND the mask together to avoid misalignments
        if self.transform:
            image = self.transform(image)
        
        # Convert the mask to class indices
        mask = np.array(mask)
        
        # Check the mask format
        if mask.ndim == 3 and mask.shape[-1] == 3:
            # RGB mask -> class indices
            mask = rgb_to_class(mask)
        elif mask.ndim == 2:
            # Already in class indices, but check values
            unique_vals = np.unique(mask)
            if np.max(unique_vals) > 21:  # If values > 21, probably misinterpreted RGB indices
                print(f"Warning: mask has values > 21: {unique_vals}")
                # Try to map strange values to ignore_index
                mask = np.where(mask > 21, 255, mask)
        
        # Resize the mask BEFORE converting it to tensor
        if mask.shape != (self.image_size, self.image_size):
            # Use PIL for resizing masks (more reliable)
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask.astype(np.uint8))
            mask_pil = mask_pil.resize((self.image_size, self.image_size), PILImage.NEAREST)
            mask = np.array(mask_pil)
        
        mask = torch.as_tensor(mask, dtype=torch.long)
        
        # Final check
        if torch.max(mask) > 255:
            print(f"Warning: mask contains invalid values: {torch.unique(mask)}")
            mask = torch.clamp(mask, 0, 255)
        
        return image, mask


def main(args):
    # Configuration
    device = torch.device(args.device)
    num_classes = 21  # 20 classes + 1 background for PASCAL VOC

    # Transformations (unchanged)
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.long))
    ])
    logger = logging.getLogger(__name__)
    # Load the PASCAL VOC dataset from Hugging Face
    print("Loading the PASCAL VOC 2012 dataset from Hugging Face...")
    voc_dataset = load_dataset("nateraw/pascal-voc-2012")
    train_dataset = VOCDataset(
        voc_dataset['train'], 
        transform=image_transform, 
        image_size=args.image_size
    )
    val_dataset = VOCDataset(
        voc_dataset['val'], 
        transform=image_transform, 
        image_size=args.image_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Dataset loaded.")
    # Loading the pre-trained SiT model

    # Load the VAE and freeze it
    print("Loading the pre-trained VAE...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE loaded and frozen.")

    print(f"Loading SiT model '{args.model}'...")
    latent_size = args.image_size // 8
    if args.repa:
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            repa=True
        )
    else:
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
        )
    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # Variable to store checkpoint train_steps if loaded
    checkpoint_train_steps = None
    
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        logger.info(f"Loading checkpoint from {ckpt_path}")
        
        # Try to load as a full checkpoint first
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                # Full checkpoint with metadata
                model.load_state_dict(checkpoint["model"])
                ema.load_state_dict(checkpoint["ema"])
                
                # Get train_steps from checkpoint if available
                if "train_steps" in checkpoint:
                    checkpoint_train_steps = checkpoint["train_steps"]
                    logger.info(f"Checkpoint train_steps: {checkpoint_train_steps}")
                
                # Restore block dropping state
                if "skip_blocks" in checkpoint:
                    model.skip_blocks = set(checkpoint["skip_blocks"])
                    ema.skip_blocks = set(checkpoint["skip_blocks"])
                    logger.info(f"Restored dropped blocks: {sorted(model.skip_blocks)}")
                
                if "model_training_step" in checkpoint:
                    model.training_step = checkpoint["model_training_step"]
                    logger.info(f"Restored model training step: {model.training_step}")
                
                if "drop_block_schedule" in checkpoint:
                    model.drop_block_schedule = checkpoint["drop_block_schedule"]
                
                if "auto_drop_steps" in checkpoint:
                    model.auto_drop_steps = checkpoint["auto_drop_steps"]
                
                if "similarity_metric" in checkpoint:
                    model.similarity_metric = checkpoint["similarity_metric"]
                
                if "current_activation_indices_source" in checkpoint:
                    model.current_activation_indices_source = checkpoint["current_activation_indices_source"]
                if "current_activation_indices_target" in checkpoint:
                    model.current_activation_indices_target = checkpoint["current_activation_indices_target"]
                
                logger.info("Checkpoint loaded successfully with block dropping state")
            else:
                # Fallback: just model weights
                model.load_state_dict(checkpoint)
                ema.load_state_dict(checkpoint)
                logger.info("Loaded model weights only (no block dropping state)")
                
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            # Try old method as fallback
            state_dict = find_model(ckpt_path)
            model.load_state_dict(state_dict)
            ema.load_state_dict(state_dict)
            logger.info("Loaded using fallback method")

    # Move models to GPU
    model, ema = model.cuda(), ema.cuda()
    model.eval()  
    ema.eval()  


    transport = create_transport(
        "Linear",
        "velocity",
        None,
        None,
        None,
    )

    for param in model.parameters():
        param.requires_grad = False
    for param in ema.parameters():
        param.requires_grad = False
    print("Model loaded.")

    # Test for activations sizes
    print("DDetecting activation sizes...")

    # Determine avalable layers for segmentation probes
    num_blocks = len(ema.blocks)
    layer_names = [f'blocks.{i}' for i in range(num_blocks)]
    print(f"{len(layer_names)} layers detected: {layer_names}")
    
    # Create a list of segmentation heads
    feature_dim = ema.hidden_size
    seg_heads = nn.ModuleList([
        LinearSegmentationProbe(
            in_channels=feature_dim, 
            num_classes=num_classes, 
            target_size=args.image_size
        ) for _ in range(len(layer_names))
    ]).to(device)

 
    # Create a list of optimizers
    optimizers = [
        torch.optim.AdamW(seg_head.parameters(), lr=args.lr, weight_decay=1e-4)
        for seg_head in seg_heads
    ]
    
    # Create a list of metrics
    if JaccardIndex:
        metrics = [
            JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)
            for _ in range(len(layer_names))
        ]
    else:
        metrics = [None] * len(layer_names)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Tracking results
    best_mious = [0.0] * len(layer_names)
    
    print(f"✅ {len(seg_heads)} segmentation heads created")
    for sit_model in [ema]:
        for epoch in range(args.epochs):
            # Set all heads to training mode
            for seg_head in seg_heads:
                seg_head.train()
                
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
            # Adapt the loop to handle Hugging Face output format
            for images, targets in pbar:
                images, targets = images.to(device), targets.to(device)
                with torch.no_grad():
                    latent_dist = vae.encode(images).latent_dist
                    latents = latent_dist.sample()
                    latents = latents * 0.18215
                # Use fixed values for t and y for consistent feature extraction
                t = torch.full((images.size(0),), 0.999, device=device) # t close to 1 (data side)
                y = torch.full((images.size(0),), 1000, dtype=torch.long, device=device) # Unconditional class

                if not layer_names:
                    logger.warning("⚠️ No valid layer names for PCA visualization")
                    continue

                _ , activations_dict = sit_model(latents, t, y)

                losses = []
                for i, layer_name in enumerate(activations_dict.keys()):
                    features = activations_dict[layer_name]
                    
                    optimizers[i].zero_grad()
                    masks_pred = seg_heads[i](features)
                    loss = criterion(masks_pred, targets)
                    
                    # Check if the loss is valid
                    if torch.isfinite(loss):
                        loss.backward()
                        # Gradient clipping to prevent explosion
                        torch.nn.utils.clip_grad_norm_(seg_heads[i].parameters(), max_norm=1.0)
                        optimizers[i].step()
                        losses.append(loss.item())
                    else:
                        print(f"Warning: Invalid loss for layer {i}: {loss.item()}")
                        losses.append(0.0)
                
                # Display the average loss
                avg_loss = np.mean(losses) if losses else 0.0
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

            # Validation loop for all heads
            for seg_head in seg_heads:
                seg_head.eval()
            
            if JaccardIndex:
                for metric in metrics:
                    if metric is not None:
                        metric.reset()
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation All Layers]")
                
                for images, targets in pbar_val:
                    images, targets = images.to(device), targets.to(device)
                    
                    latent_dist = vae.encode(images).latent_dist
                    latents = latent_dist.sample() * 0.18215
                    
                    t = torch.full((images.size(0),), 0.999, device=device)
                    y = torch.full((images.size(0),), 1000, dtype=torch.long, device=device)
                    
                    # Extract all activations
                    _ , activations_dict = sit_model(latents, t, y)
                    
                    # Evaluate each head
                    for i, layer_name in enumerate(activations_dict.keys()):
                        features = activations_dict[layer_name]
                        masks_pred = seg_heads[i](features)
                        
                        if metrics[i] is not None:
                            # Check if the predictions are valid
                            pred_classes = masks_pred.argmax(dim=1)
                            
                            # Debug: check values
                            if torch.any(torch.isnan(masks_pred)) or torch.any(torch.isinf(masks_pred)):
                                print(f"Warning: Invalid predictions for layer {i}")
                                continue
                                
                            # Ensure values are within the correct range
                            pred_classes = torch.clamp(pred_classes, 0, num_classes-1)
                            targets_clamped = torch.clamp(targets, 0, 255)  # 255 = ignore_index
                            
                            metrics[i].update(pred_classes, targets_clamped)
            # Calculate and display results
            print(f"\n📊 Epoch {epoch+1} Results:")
            print("-" * 50)

            if JaccardIndex:
                for i, layer_name in enumerate(layer_names):
                    if metrics[i] is not None:
                        # Check if there is data in the metric
                        try:
                            # Access internal states to check if there is data
                            if hasattr(metrics[i], 'confmat') and torch.sum(metrics[i].confmat) > 0:
                                miou = metrics[i].compute().item()
                                layer_idx = int(layer_name.split('.')[1])
                                
                                # Check if the mIoU is valid
                                if not (torch.isnan(torch.tensor(miou)) or torch.isinf(torch.tensor(miou))):
                                    print(f"Layer {layer_idx:2d}: mIoU = {miou:.4f}")
                                    
                                    # Save the best model for each layer
                                    if miou > best_mious[i]:
                                        best_mious[i] = miou
                                        torch.save(
                                            seg_heads[i].state_dict(), 
                                            f"seg_head_layer_{layer_idx:02d}_best.pt"
                                        )
                                else:
                                    print(f"Layer {layer_idx:2d}: mIoU = NaN/Inf (skipped)")
                            else:
                                layer_idx = int(layer_name.split('.')[1])
                                print(f"Layer {layer_idx:2d}: No valid predictions (skipped)")
                        except Exception as e:
                            layer_idx = int(layer_name.split('.')[1])
                            print(f"Layer {layer_idx:2d}: Error computing mIoU: {e}")
            # Identify the best layer
            if JaccardIndex and best_mious:
                best_layer_idx = np.argmax(best_mious)
                best_miou = best_mious[best_layer_idx]
                print(f"\n🏆 Best layer so far: {best_layer_idx} (mIoU: {best_miou:.4f})")
                

    # Final summary
    print(f"\n🎯 Final Results:")
    print("=" * 60)
    for i, layer_name in enumerate(layer_names):
        layer_idx = int(layer_name.split('.')[1])
        print(f"Layer {layer_idx:2d}: Best mIoU = {best_mious[i]:.4f}")
    
    if best_mious:
        overall_best_idx = np.argmax(best_mious)
        overall_best_miou = best_mious[overall_best_idx]
        print(f"\n🏆 Overall best: Layer {overall_best_idx} with mIoU = {overall_best_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiT features evaluation on PASCAL VOC.")
    parser.add_argument("--model", type=str, default="SiT-XL/2", help="SiT model architecture.")
    parser.add_argument("--ckpt", type=str, required=False, default="")
    parser.add_argument("--data-path", type=str, default="", help="Path to store the PASCAL VOC dataset.")
    parser.add_argument("--image-size", type=int, default=256, help="Image size on which the SiT was trained.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train the segmentation head.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-map", action="store_true", help="Enable logging of feature maps.")
    parser.add_argument('--repa', action='store_true', help='Use SiT with REPA')
    args = parser.parse_args()
    main(args)


