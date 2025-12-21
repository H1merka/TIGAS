"""Benchmark DataLoader vs Model speed."""
import torch
import time
from pathlib import Path

print("Setting up benchmark...")

# Import components
from tigas.models.tigas_model import TIGASModel
from tigas.data.loaders import create_dataloaders_from_csv

# Create model
model = TIGASModel(img_size=128, fast_mode=True).cuda().eval()
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Create dataloaders (with reduced workers for testing)
data_root = Path("C:/Dev/TIGAS_dataset/TIGAS")
if data_root.exists():
    print("\nCreating DataLoader with 4 workers...")
    loaders = create_dataloaders_from_csv(
        data_root=str(data_root),
        batch_size=16,
        img_size=128,
        num_workers=4,  # Reduced for testing
        augment_level='light'
    )
    train_loader = loaders['train']
    
    print("\n=== BENCHMARK ===")
    
    # Warmup DataLoader
    data_iter = iter(train_loader)
    for _ in range(3):
        _ = next(data_iter)
    
    # Measure DataLoader time
    print("\nMeasuring DataLoader speed (10 batches)...")
    torch.cuda.synchronize()
    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 10:
            break
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
    torch.cuda.synchronize()
    data_time = (time.time() - start) / 10
    print(f"DataLoader time per batch: {data_time*1000:.1f}ms")
    
    # Measure Model time
    print("\nMeasuring Model speed (10 batches)...")
    x = torch.randn(16, 3, 128, 128).cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    torch.cuda.synchronize()
    model_time = (time.time() - start) / 10
    print(f"Model time per batch: {model_time*1000:.1f}ms")
    
    # Combined
    print("\n=== RESULTS ===")
    print(f"DataLoader: {data_time*1000:.1f}ms ({data_time/(data_time+model_time)*100:.1f}%)")
    print(f"Model:      {model_time*1000:.1f}ms ({model_time/(data_time+model_time)*100:.1f}%)")
    print(f"Total:      {(data_time+model_time)*1000:.1f}ms per iteration")
    print(f"Expected speed: {1/(data_time+model_time):.1f} it/sec")
    
else:
    print(f"Dataset not found at {data_root}")
    print("Run with your dataset path")
