#!/usr/bin/env python3
"""
Script to compare layer activations between a SiT model and a pre-trained encoder (DINOv2, MAE, etc.)
using alignment metrics from metrics.py.
Loads a SiT checkpoint, a hub encoder, processes n images from the dataset with noise (as in transport.py),
and computes alignment metrics between the layers of the two models.
"""

import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
from datasets import load_dataset
import sys
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from diffusers.models import AutoencoderKL

# Imports du projet
from LayerSync.models import SiT_models
from LayerSync.transport import create_transport
from LayerSync.evaluation.metrics import AlignmentMetrics


# Configuration of available encoders
ENCODER_CONFIGS = {
    'dinov2_vits14': {
        'repo': 'facebookresearch/dinov2',
        'model_name': 'dinov2_vits14',
        'patch_size': 14,
        'embed_dim': 384,
        'num_layers': 12,
        'image_size': 224,
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    },
    'dinov2_vitb14': {
        'repo': 'facebookresearch/dinov2',
        'model_name': 'dinov2_vitb14',
        'patch_size': 14,
        'embed_dim': 768,
        'num_layers': 12,
        'image_size': 224,
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    },
    'dinov2_vitl14': {
        'repo': 'facebookresearch/dinov2',
        'model_name': 'dinov2_vitl14',
        'patch_size': 14,
        'embed_dim': 1024,
        'num_layers': 24,
        'image_size': 224,
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    },
    'dinov2_vitg14': {
        'repo': 'facebookresearch/dinov2',
        'model_name': 'dinov2_vitg14',
        'patch_size': 14,
        'embed_dim': 1536,
        'num_layers': 40,
        'image_size': 224,
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    },
    'mae_vitb16': {
        'repo': 'facebookresearch/mae',
        'model_name': 'vit_base_patch16_224.mae',
        'patch_size': 16,
        'embed_dim': 768,
        'num_layers': 12,
        'image_size': 224,
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    },
    'mae_vitl16': {
        'repo': 'facebookresearch/mae',
        'model_name': 'vit_large_patch16_224.mae',
        'patch_size': 16,
        'embed_dim': 1024,
        'num_layers': 24,
        'image_size': 224,
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    },
    'mae_vith14': {
        'repo': 'facebookresearch/mae',
        'model_name': 'vit_huge_patch14_224.mae',
        'patch_size': 14,
        'embed_dim': 1280,
        'num_layers': 32,
        'image_size': 224,
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    }
}


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def load_sit_model(checkpoint_path, model_name="SiT-XL/2", image_size=256, num_classes=1000, device='cuda', repa=False):
    """
    Loads a SiT model from a checkpoint.
    """
    latent_size = image_size // 8
    if repa:
        model = SiT_models[model_name](
            input_size=latent_size,
            num_classes=num_classes,
            learn_sigma=True,
            repa=True,
        ).to(device)
    else:
        model = SiT_models[model_name](
            input_size=latent_size,
            num_classes=num_classes,
            learn_sigma=True,
        ).to(device)
    
    if os.path.isfile(checkpoint_path):
        try:
            print("Loading SiT state dict")
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("Checking content")
            if "ema" in state_dict:
                state_dict = state_dict["ema"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Error during loading: {e}")
            print("The checkpoint file might be corrupted. Check its size:")
            print(f"File size: {os.path.getsize(checkpoint_path)} bytes")
            raise
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def load_pretrained_encoder(encoder_name, device='cuda'):
    """
    Loads a pre-trained encoder from the hub.
    """
    if encoder_name not in ENCODER_CONFIGS:
        raise ValueError(f"Unsupported encoder: {encoder_name}. Available: {list(ENCODER_CONFIGS.keys())}")
    
    config = ENCODER_CONFIGS[encoder_name]
    
    try:
        if 'dinov2' in encoder_name:
            # Load DINOv2
            model = torch.hub.load(config['repo'], config['model_name'], pretrained=True)
        elif 'mae' in encoder_name:
            # Load MAE via timm
            model = timm.create_model(config['model_name'], pretrained=True)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_name}")
        
        model.eval()
        model.to(device)
        return model, config
        
    except Exception as e:
        print(f"Error during loading of encoder {encoder_name}: {e}")
        raise


class FeatureExtractor:
    """
    Feature extractor to capture intermediate activations.
    """
    def __init__(self, model, model_type='sit', final_layer_only=False):
        self.model = model
        self.model_type = model_type
        self.final_layer_only = final_layer_only
        self.features = {}
        self.hooks = []
        
    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[name] = output.detach()
        return hook
    
    def register_hooks(self):
        if self.model_type == 'sit':
            # For SiT, capture block outputs (always all layers)
            for i, block in enumerate(self.model.blocks):
                hook = block.register_forward_hook(self._hook_fn(f'block_{i}'))
                self.hooks.append(hook)
        elif self.model_type == 'dinov2':
            if self.final_layer_only:
                # Capture only the output of the last layer
                last_block_idx = len(self.model.blocks) - 1
                hook = self.model.blocks[last_block_idx].register_forward_hook(self._hook_fn('final_output'))
                self.hooks.append(hook)
            else:
                # For DINOv2, capture block outputs
                for i, block in enumerate(self.model.blocks):
                    hook = block.register_forward_hook(self._hook_fn(f'block_{i}'))
                    self.hooks.append(hook)
        elif self.model_type == 'mae':
            if self.final_layer_only:
                # Capture only the output of the last layer
                if hasattr(self.model, 'blocks'):
                    last_block_idx = len(self.model.blocks) - 1
                    hook = self.model.blocks[last_block_idx].register_forward_hook(self._hook_fn('final_output'))
                    self.hooks.append(hook)
                elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                    last_layer_idx = len(self.model.encoder.layers) - 1
                    hook = self.model.encoder.layers[last_layer_idx].register_forward_hook(self._hook_fn('final_output'))
                    self.hooks.append(hook)
            else:
                # For MAE, capture encoder block outputs
                if hasattr(self.model, 'blocks'):
                    for i, block in enumerate(self.model.blocks):
                        hook = block.register_forward_hook(self._hook_fn(f'block_{i}'))
                        self.hooks.append(hook)
                elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                    for i, layer in enumerate(self.model.encoder.layers):
                        hook = layer.register_forward_hook(self._hook_fn(f'block_{i}'))
                        self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __call__(self, x):
        self.features = {}
        with torch.no_grad():
            _ = self.model(x)
        return self.features


def flatten_features(features):
    """
    Flattens features to use them with alignment metrics.
    Returns a tensor of shape (batch_size, feature_dim).
    """
    if features.dim() == 4:  # (B, C, H, W)
        # For spatial features, average over spatial dimensions
        features = features.mean(dim=[2, 3])
    elif features.dim() == 3:  # (B, seq_len, hidden_dim) - transformer format
        # For transformer features, average over sequence dimension (except CLS token)
        if features.shape[1] > 1:  # If we have more than one token
            # Exclude first token (CLS) if averaging
            features = features[:, 1:].mean(dim=1)
        else:
            features = features.squeeze(1)
    elif features.dim() == 2:  # (B, feature_dim)
        # Already in correct format
        pass
    else:
        raise ValueError(f"Unsupported feature format: {features.shape}")
    
    return features


def add_noise_to_latents(x, transport, t):
    """
    Adds noise to latents as in transport.py
    """
    with torch.no_grad():
        # Use transport method to add noise
        noise = torch.randn_like(x)
        t, xt, ut = transport.path_sampler.plan(t, noise, x)
    return xt


def compute_metrics_for_layer_pair(features_A, features_B, metrics_list, topk=10, cca_dim=10):
    """
    Computes all requested metrics for a layer pair.
    """
    # Flatten features
    flat_A = flatten_features(features_A)
    flat_B = flatten_features(features_B)
    
    # Normalize features
    flat_A = F.normalize(flat_A, dim=-1)
    flat_B = F.normalize(flat_B, dim=-1)
    
    results = {}
    
    for metric in metrics_list:
        try:
            kwargs = {}
            if 'knn' in metric or 'cknna' in metric:
                kwargs['topk'] = min(topk, flat_A.shape[0] - 1)
            if 'svcca' in metric:
                kwargs['cca_dim'] = min(cca_dim, min(flat_A.shape[1], flat_B.shape[1], flat_A.shape[0]))
            
            score = AlignmentMetrics.measure(metric, flat_A, flat_B, **kwargs)
            results[metric] = score
        except Exception as e:
            print(f"Error during calculation of {metric}: {e}")
            results[metric] = np.nan
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compares activations between a SiT model and a pre-trained encoder with metrics.py')
    parser.add_argument('--sit_checkpoint', type=str, required=True,
                       help='Path to SiT checkpoint')
    parser.add_argument('--encoder', type=str, required=True,
                       choices=list(ENCODER_CONFIGS.keys()),
                       help='Name of the pre-trained encoder to use')
    parser.add_argument('--data_path', type=str, default='',
                       help='Path to dataset')
    parser.add_argument('--n_images', type=int, default=1000,
                       help='Number of images to process')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for SiT')
    parser.add_argument('--sit_model', type=str, default='SiT-XL/2',
                       help='SiT model name')
    parser.add_argument('--num_classes', type=int, default=1000,
                       help='Number of classes')
    parser.add_argument('--noise_level', type=float, default=0.1,
                       help='Noise level (0.0 to 1.0)')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['cka', 'mutual_knn', 'cycle_knn'],
                       choices=AlignmentMetrics.SUPPORTED_METRICS,
                       help='Metrics to compute')
    parser.add_argument('--encoder_final_only', action='store_true',
                       help='Only compare with encoder final output (recommended to save memory)')
    parser.add_argument('--output_dir', type=str, default='sit_encoder_metrics_comparison',
                       help='Output directory for results')
    parser.add_argument('--topk', type=int, default=25,
                       help='K for KNN metrics')
    parser.add_argument('--cca_dim', type=int, default=10,
                       help='Dimension for SVCCA')
    parser.add_argument('--repa', action='store_true',
                       help='Use SiT with REPA')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load SiT model
    print("Loading SiT model...")
    sit_model = load_sit_model(args.sit_checkpoint, args.sit_model, args.image_size, args.num_classes, device, args.repa)
    sit_model.eval()
    
    # Load pre-trained encoder
    print(f"Loading encoder {args.encoder}...")
    encoder_model, encoder_config = load_pretrained_encoder(args.encoder, device)
    encoder_model.eval()
    
    # Determine encoder type for feature extraction
    if 'dinov2' in args.encoder:
        encoder_type = 'dinov2'
    elif 'mae' in args.encoder:
        encoder_type = 'mae'
    else:
        encoder_type = 'other'
    
    # Create feature extractors
    print("Configuring feature extractors...")
    sit_extractor = FeatureExtractor(sit_model, 'sit')
    encoder_extractor = FeatureExtractor(encoder_model, encoder_type, final_layer_only=args.encoder_final_only)
    
    sit_extractor.register_hooks()
    encoder_extractor.register_hooks()
    
    print(f"SiT extractor: {len(sit_extractor.hooks)} hooks (all layers)")
    if args.encoder_final_only:
        print(f"Encoder extractor: {len(encoder_extractor.hooks)} hooks (final layer only)")
    else:
        print(f"Encoder extractor: {len(encoder_extractor.hooks)} hooks (all layers)")
    
    # Load VAE for SiT
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    
    # Create transport for adding noise
    transport = create_transport(
        path_type="Linear",
        prediction="velocity", 
        loss_weight=None,
        train_eps=1e-5,
        sample_eps=1e-3
    )
    
    # Prepare transformations for both models
    sit_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    encoder_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, encoder_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=encoder_config['normalize']['mean'], 
            std=encoder_config['normalize']['std']
        )
    ])

    # Prepare dataset
    try:
        hf_dataset = load_dataset(
            "imagenet-1k",
            split="train",
            streaming=False,
            cache_dir="/scratch/bvandelft/cache"
        )
        print(f"✅ Dataset cached successfully with {len(hf_dataset):,} images")              
    except Exception as e:
        raise e 

    class HFDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, sit_transform=None, encoder_transform=None):
            self.hf_dataset = hf_dataset
            self.sit_transform = sit_transform
            self.encoder_transform = encoder_transform
            
        def __len__(self):
            return len(self.hf_dataset)
            
        def __getitem__(self, idx):
            try:
                item = self.hf_dataset[idx]
                image = item['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                label = item['label']
                
                # Apply both transformations
                sit_image = self.sit_transform(image) if self.sit_transform else image
                encoder_image = self.encoder_transform(image) if self.encoder_transform else image
                
                return sit_image, encoder_image, label
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                raise e                    
            
    dataset = HFDatasetWrapper(hf_dataset, sit_transform, encoder_transform)
    print(f"Loaded HuggingFace dataset with {len(dataset):,} images")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Processing {min(args.n_images, len(dataset))} images in batches of {args.batch_size}")
    print(f"Metrics to compute: {args.metrics}")
    
    # Collect all activations
    all_sit_activations = []
    all_encoder_activations = []
    
    for batch_idx, (sit_images, encoder_images, labels) in enumerate(tqdm(loader, desc="Collecting activations")):
        if batch_idx * args.batch_size >= args.n_images:
            break
            
        sit_images = sit_images.to(device)
        encoder_images = encoder_images.to(device)
        labels = labels.to(device)
        
        # Process with SiT (with noise)
        with torch.no_grad():
            latents = vae.encode(sit_images).latent_dist.sample().mul_(0.18215)
            t = torch.full((latents.size(0),), args.noise_level, device=device)
            noisy_latents = add_noise_to_latents(latents, transport, t)
            _, sit_activations = sit_model(noisy_latents, t, labels)
        
        # Process with encoder
        encoder_activations = encoder_extractor(encoder_images)
        
        # Store activations
        all_sit_activations.append(sit_activations)
        all_encoder_activations.append(encoder_activations)
    
    # Clean up hooks
    sit_extractor.remove_hooks()
    encoder_extractor.remove_hooks()
    
    print("Computing alignment metrics...")
    
    # Get layer names
    sit_layer_names = list(all_sit_activations[0].keys())
    encoder_layer_names = list(all_encoder_activations[0].keys())
    
    print(f"SiT Layers: {sit_layer_names}")
    print(f"Encoder Layers: {encoder_layer_names}")
    
    # Compute metrics for all layer pairs
    results = {}
    for metric in args.metrics:
        results[metric] = {}
    
    for sit_layer in tqdm(sit_layer_names, desc="Processing SiT layers"):
        for encoder_layer in encoder_layer_names:
            # Concatenate activations from all batches
            sit_features = torch.cat([batch[sit_layer] for batch in all_sit_activations], dim=0)
            encoder_features = torch.cat([batch[encoder_layer] for batch in all_encoder_activations], dim=0)
            
            # Compute metrics
            layer_results = compute_metrics_for_layer_pair(
                sit_features, encoder_features, args.metrics, 
                topk=args.topk, cca_dim=args.cca_dim
            )
            
            pair_name = f"SiT_{sit_layer} -> Encoder_{encoder_layer}"
            for metric, score in layer_results.items():
                if metric not in results:
                    results[metric] = {}
                results[metric][pair_name] = score
    
    # Display and save results for each metric
    for metric in args.metrics:
        print(f"\n{'='*80}")
        print(f"RESULTS FOR METRIC: {metric.upper()}")
        print(f"{'='*80}")
        
        metric_results = results[metric]
        sorted_items = sorted(metric_results.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -float('inf'), reverse=True)
        
        print(f"Total of {len(metric_results)} pairs compared")
        print(f"\nTOP 10 pairs with best scores:")
        print("-" * 60)
        for pair_name, score in sorted_items[:10]:
            if not np.isnan(score):
                print(f"{pair_name:50} : {score:.6f}")
        
        print(f"\nTOP 10 pairs with worst scores:")
        print("-" * 60)
        for pair_name, score in sorted_items[-10:]:
            if not np.isnan(score):
                print(f"{pair_name:50} : {score:.6f}")
        
        # Save results for this metric
        results_file = os.path.join(args.output_dir, f'sit_{args.encoder}_{metric}_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Comparison between:\n")
            f.write(f"SiT Model: {args.sit_checkpoint}\n")
            f.write(f"Encoder: {args.encoder}\n")
            f.write(f"Metric: {metric}\n")
            f.write(f"Number of images: {args.n_images}\n")
            f.write(f"Noise level: {args.noise_level}\n")
            f.write(f"Total pairs compared: {len(metric_results)}\n\n")
            
            f.write("All pairs sorted by descending score:\n")
            f.write("=" * 80 + "\n")
            for pair_name, score in sorted_items:
                if not np.isnan(score):
                    f.write(f"{pair_name}: {score:.6f}\n")
        
        # Create visualization for this metric
        # Extract unique layer names from pair_names
        sit_layer_names = set()
        encoder_layer_names = set()
        
        for pair_name in metric_results.keys():
            # Format: "SiT_{sit_layer} -> Encoder_{encoder_layer}"
            parts = pair_name.split(' -> ')
            if len(parts) == 2:
                sit_part = parts[0].replace('SiT_', '')
                encoder_part = parts[1].replace('Encoder_', '')
                sit_layer_names.add(sit_part)
                encoder_layer_names.add(encoder_part)
        
        sit_layer_names = sorted(list(sit_layer_names))
        encoder_layer_names = sorted(list(encoder_layer_names))
        
        # Sort SiT layers by numerical index instead of lexicographical order
        def extract_layer_index(layer_name):
            """Extracts numerical index from a layer name"""
            try:
                if 'block_' in layer_name:
                    return int(layer_name.replace('block_', ''))
                else:
                    return int(layer_name)
            except ValueError:
                return float('inf')  # Put non-numerical names at the end
        
        sit_layers_clean = sorted(sit_layer_names, key=extract_layer_index)
        encoder_layers_clean = sorted(encoder_layer_names)
        
        # Choose visualization type based on mode
        if args.encoder_final_only and len(encoder_layers_clean) == 1:
            # Curve mode: SiT layers vs metric
            sit_data = []  # List of tuples (index, value)
            
            for sit_layer in sit_layers_clean:
                # Extract numerical index of the SiT layer 
                try:
                    if 'block_' in sit_layer:
                        layer_idx = int(sit_layer.replace('block_', ''))
                    else:
                        layer_idx = int(sit_layer)
                    
                    # Get score for this layer
                    pair_key = f"SiT_{sit_layer} -> Encoder_{encoder_layers_clean[0]}"
                    if pair_key in metric_results and not np.isnan(metric_results[pair_key]):
                        sit_data.append((layer_idx, metric_results[pair_key]))
                    else:
                        sit_data.append((layer_idx, np.nan))
                except ValueError:
                    continue
            
            # Sort by index to ensure correct order
            sit_data.sort(key=lambda x: x[0])
            
            # Separate indices and values
            sit_indices = [x[0] for x in sit_data]
            metric_values = [x[1] for x in sit_data]
            
            # Display results in correct order
            print(f"\n{metric} results in layer order:")
            print("-" * 60)
            for idx, val in sit_data:
                if not np.isnan(val):
                    print(f"SiT_{idx} -> Encoder_{encoder_layers_clean[0]}: {val:.6f}")
            
            # Create curve
            plt.figure(figsize=(12, 8))
            
            # Filter NaN values for plotting
            valid_indices = []
            valid_values = []
            for i, val in zip(sit_indices, metric_values):
                if not np.isnan(val):
                    valid_indices.append(i)
                    valid_values.append(val)
            
            if valid_indices:
                plt.plot(valid_indices, valid_values, 'o-', linewidth=2, markersize=8, 
                        label=f'{metric} vs {args.encoder}', color='blue')
                plt.scatter(valid_indices, valid_values, s=60, alpha=0.7, color='red')
                
                # Annotations for values
                for i, val in zip(valid_indices, valid_values):
                    plt.annotate(f'{val:.3f}', (i, val), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=9, alpha=0.8)
            else:
                # Create empty plot with error message
                plt.text(0.5, 0.5, 'No valid data to display', 
                        transform=plt.gca().transAxes, ha='center', va='center', 
                        fontsize=16, color='red')
            
            plt.xlabel('SiT Layer (index)', fontsize=12)
            plt.ylabel(f'{metric} Score', fontsize=12)
            plt.title(f'Evolution of {metric} score across SiT layers\n'
                     f'Compared with {args.encoder} (final layer)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Improve x-axis ticks
            if valid_indices:
                plt.xlim(min(valid_indices)-0.5, max(valid_indices)+0.5)
                plt.xticks(range(min(valid_indices), max(valid_indices)+1))
            
            plt.tight_layout()
            
            plot_file = os.path.join(args.output_dir, f'sit_{args.encoder}_{metric}_curve.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Curve saved: {plot_file}")
            
        else:
            # Classic heatmap mode for full comparisons
            distance_matrix = np.full((len(sit_layers_clean), len(encoder_layers_clean)), np.nan)
            for i, sit_layer in enumerate(sit_layers_clean):
                for j, encoder_layer in enumerate(encoder_layers_clean):
                    pair_key = f"SiT_{sit_layer} -> Encoder_{encoder_layer}"
                    if pair_key in metric_results and not np.isnan(metric_results[pair_key]):
                        distance_matrix[i, j] = metric_results[pair_key]
            
            # Reverse SiT layer order for display
            sit_layers_reversed = sit_layers_clean[::-1]
            distance_matrix_reversed = distance_matrix[::-1, :]
            
            plt.figure(figsize=(max(12, len(encoder_layers_clean)), max(12, len(sit_layers_clean))))
            sns.heatmap(distance_matrix_reversed, 
                        xticklabels=encoder_layers_clean, 
                        yticklabels=sit_layers_reversed,
                        annot=True, 
                        fmt='.4f', 
                        cmap='viridis',
                        cbar_kws={'label': f'{metric} score'})
            
            plt.xlabel(f'{args.encoder} Layers')
            plt.ylabel('SiT Layers')
            plt.title(f'{metric} scores between SiT and {args.encoder}')
            plt.tight_layout()
            
            plot_file = os.path.join(args.output_dir, f'sit_{args.encoder}_{metric}_heatmap.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Heatmap saved: {plot_file}")
        
        print(f"Results saved: {results_file}")
    
    # Create a summary file with all metrics
    summary_file = os.path.join(args.output_dir, f'sit_{args.encoder}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"COMPARISON SUMMARY\n")
        f.write(f"{'='*50}\n")
        f.write(f"SiT Model: {args.sit_checkpoint}\n")
        f.write(f"Encoder: {args.encoder}\n")
        f.write(f"Metrics computed: {args.metrics}\n")
        f.write(f"Number of images: {args.n_images}\n")
        f.write(f"Noise level: {args.noise_level}\n\n")
        
        for metric in args.metrics:
            f.write(f"\nMETRIC: {metric.upper()}\n")
            f.write("-" * 30 + "\n")
            metric_results = results[metric]
            valid_scores = [score for score in metric_results.values() if not np.isnan(score)]
            if valid_scores:
                f.write(f"Average score: {np.mean(valid_scores):.6f}\n")
                f.write(f"Median score: {np.median(valid_scores):.6f}\n")
                f.write(f"Max score: {np.max(valid_scores):.6f}\n")
                f.write(f"Min score: {np.min(valid_scores):.6f}\n")
                f.write(f"Standard deviation: {np.std(valid_scores):.6f}\n")
            else:
                f.write("No valid score calculated\n")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved in: {args.output_dir}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
