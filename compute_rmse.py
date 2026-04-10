"""
Compute RMSE for VAD-Net predictions on test sets.
Computes RMSE for Valence, Arousal, and Dominance dimensions separately.
"""

import torch
import torch.nn as nn
import numpy as np
import transforms as transforms
from fer import FER2013
from models.efficientnet_b0 import EfficientNetVAD
from models.mobilefacenet import MobileFaceNetVAD
from models.resnet_reg2 import (
    ResNet18RegressionThreeOutputs,
    ResNet50PretrainedRegressionThreeOutputs,
    ResNet50RegressionThreeOutputs,
)
import os
from PIL import Image


EVAL_INPUT_SIZE = 48


def custom_transform(crops):
    """Convert crops to tensor and normalize."""
    normalize = transforms.Normalize(
        mean=FER2013.image_mean.tolist(),
        std=FER2013.image_std.tolist()
    )
    processed = []
    for crop in crops:
        if crop.size != (EVAL_INPUT_SIZE, EVAL_INPUT_SIZE):
            crop = crop.resize((EVAL_INPUT_SIZE, EVAL_INPUT_SIZE), Image.BILINEAR)
        processed.append(normalize(transforms.ToTensor()(crop)))
    return torch.stack(processed)


def infer_checkpoint_arch(state_dict):
    keys = list(state_dict.keys())

    if any(k.startswith("regression_head") for k in keys):
        return "mobilefacenet"
    if any(k.startswith("backbone._conv_stem") for k in keys):
        return "efficientnet"
    if any(k.startswith("backbone.") for k in keys):
        return "resnet50_pretrained"
    if any("layer1.0.conv3.weight" in k for k in keys):
        return "resnet50"
    return "resnet18"


def build_model_from_state_dict(state_dict):
    arch = infer_checkpoint_arch(state_dict)
    if arch == "resnet50_pretrained":
        return ResNet50PretrainedRegressionThreeOutputs()
    if arch == "resnet50":
        return ResNet50RegressionThreeOutputs()
    if arch == "efficientnet":
        return EfficientNetVAD(dropout_rate=0.0)
    if arch == "mobilefacenet":
        return MobileFaceNetVAD(dropout_rate=0.0)
    return ResNet18RegressionThreeOutputs()


def compute_rmse(model, dataloader, dimension_names=['Valence', 'Arousal', 'Dominance'], use_cuda=False):
    """
    Compute RMSE for each dimension separately.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with test data
        dimension_names: List of names for each dimension
        use_cuda: Whether to use CUDA
    
    Returns:
        rmse_per_dim: Dictionary with RMSE for each dimension
        overall_rmse: Overall RMSE across all dimensions
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Handle 10-crop test augmentation
            if len(inputs.shape) == 5:  # (batch, 10crops, c, h, w)
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)
                
                if use_cuda:
                    inputs = inputs.cuda()
                
                outputs = model(inputs)
                # Average predictions across crops
                outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            else:
                if use_cuda:
                    inputs = inputs.cuda()
                outputs_avg = model(inputs)
            
            all_predictions.append(outputs_avg.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 3)
    targets = np.concatenate(all_targets, axis=0)  # (N, 3)

    label_mean = FER2013.label_mean.numpy()
    label_std = FER2013.label_std.numpy()
    predictions = predictions * label_std + label_mean
    targets = targets * label_std + label_mean
    
    rmse_per_dim = {}
    for i, dim_name in enumerate(dimension_names):
        mse = np.mean((predictions[:, i] - targets[:, i]) ** 2)
        rmse = np.sqrt(mse)
        rmse_per_dim[dim_name] = rmse
    
    # Overall RMSE
    overall_mse = np.mean((predictions - targets) ** 2)
    overall_rmse = np.sqrt(overall_mse)
    
    return rmse_per_dim, overall_rmse, predictions, targets


def evaluate_all_sets(model_path, use_cuda=False, cut_size=48, input_size=48, align_faces=False):
    """Evaluate model on both public and private test sets."""
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Try to load state_dict directly first, fall back to checkpoint
    try:
        state = torch.load(model_path, map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            # It's a checkpoint dict
            print(f"Loaded checkpoint format (epoch {state.get('epoch', 'unknown')})")
            state_dict = state['model']
        else:
            # Direct state dict
            print(f"Loaded direct state_dict")
            state_dict = state
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    net = build_model_from_state_dict(state_dict)

    try:
        net.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model into inferred architecture: {e}")
        return
    
    # Verify model architecture
    print(f"[OK] Model loaded correctly with inferred architecture: {infer_checkpoint_arch(state_dict)}\n")
    
    if use_cuda:
        net.cuda()

    FER2013._ensure_label_stats()
    FER2013._ensure_image_stats()

    global EVAL_INPUT_SIZE
    EVAL_INPUT_SIZE = input_size
    
    # Create datasets
    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        custom_transform,
    ])
    
    pubset = FER2013(split='PublicTest', transform=transform_test, align_faces=align_faces)
    publoader = torch.utils.data.DataLoader(pubset, batch_size=128)
    
    priset = FER2013(split='PrivateTest', transform=transform_test, align_faces=align_faces)
    priloader = torch.utils.data.DataLoader(priset, batch_size=128)
    
    # Compute RMSE
    print("\n" + "="*60)
    print("RMSE Results")
    print("="*60)
    
    pub_rmse_dict, pub_overall, _, _ = compute_rmse(net, publoader, use_cuda=use_cuda)
    pub_rmse = np.array([pub_rmse_dict['Valence'], pub_rmse_dict['Arousal'], pub_rmse_dict['Dominance']])
    print("\nPublic Test Set:")
    print(f"  Valence    : {pub_rmse[0]:.4f}")
    print(f"  Arousal    : {pub_rmse[1]:.4f}")
    print(f"  Dominance  : {pub_rmse[2]:.4f}")
    print(f"  Overall    : {pub_overall:.4f}")
    
    pri_rmse_dict, pri_overall, _, _ = compute_rmse(net, priloader, use_cuda=use_cuda)
    pri_rmse = np.array([pri_rmse_dict['Valence'], pri_rmse_dict['Arousal'], pri_rmse_dict['Dominance']])
    print("\nPrivate Test Set:")
    print(f"  Valence    : {pri_rmse[0]:.4f}")
    print(f"  Arousal    : {pri_rmse[1]:.4f}")
    print(f"  Dominance  : {pri_rmse[2]:.4f}")
    print(f"  Overall    : {pri_overall:.4f}")
    
    print("\n" + "="*70)
    print("PAPER vs YOUR RESULTS (without orthogonal regularization)")
    print("="*70)
    
    paper_results = {
        'Valence': {'public': 0.076, 'private': 0.063},
        'Arousal': {'public': 0.048, 'private': 0.094},
        'Dominance': {'public': 0.078, 'private': 0.069},
    }
    
    dim_names = ['Valence', 'Arousal', 'Dominance']
    
    print("\n{:<12} | {:>8} {:>8} | {:>8} {:>8}".format("Dimension", "Your Pub", "Paper", "Your Pri", "Paper"))
    print("-" * 70)
    
    for i, dim in enumerate(dim_names):
        your_pub = pub_rmse[i]
        your_pri = pri_rmse[i]
        paper_pub = paper_results[dim]['public']
        paper_pri = paper_results[dim]['private']
        
        print(f"{dim:<12} | {your_pub:>8.4f} {paper_pub:>8.4f} | {your_pri:>8.4f} {paper_pri:>8.4f}")
    
    print("-" * 70)
    print(f"{'Overall':<12} | {pub_overall:>8.4f}          | {pri_overall:>8.4f}")
    
    print("\n✅ SUCCESS!" if pub_overall < 0.15 else "\n[Note] Gap still exists")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate VAD-Net model')
    parser.add_argument('--model', type=str, default='FER2013_ResNet/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--cut_size', type=int, default=48, help='Spatial crop size used at evaluation')
    parser.add_argument('--input_size', type=int, default=0, help='Final model input size after crop (0 uses cut_size)')
    parser.add_argument('--align_faces', action='store_true', help='Enable OpenCV Haar-based face alignment before transforms')
    parser.add_argument('--public_csv', type=str, default='', help='Optional override CSV for PublicTest split')
    parser.add_argument('--private_csv', type=str, default='', help='Optional override CSV for PrivateTest split')
    args = parser.parse_args()

    FER2013.set_data_protocol('small_split')
    FER2013.set_split_files(
        public_file=args.public_csv or None,
        private_file=args.private_csv or None,
    )
    print('[Small Split] Using current extracted CSVs in ./data')
    
    use_cuda = args.cuda and torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    
    eval_input_size = args.input_size if args.input_size > 0 else args.cut_size
    evaluate_all_sets(args.model, use_cuda=use_cuda, cut_size=args.cut_size, input_size=eval_input_size, align_faces=args.align_faces)
