import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import psutil
import pandas as pd
from datetime import datetime
import subprocess
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_layers(model):
    total = 0
    for _ in model.modules():
        total += 1
    return total - 1  # Exclude the model itself

def get_gpu_count():
    return torch.cuda.device_count()

def get_data_size(dataset):
    sample, _ = dataset[0]
    single_sample_bytes = sample.element_size() * sample.nelement()
    total_bytes = single_sample_bytes * len(dataset)
    return total_bytes / (1024 * 1024)  # in MB

def get_gpu_utilizations():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        # Return a list of float values (one per GPU)
        usages = [float(x.strip()) for x in output.strip().split('\n') if x.strip()]
        return usages
    except Exception as e:
        return []

def get_system_info():
    gpu_memory_used = 0
    gpu_utilizations = get_gpu_utilizations()  # Use list of utilizations for all GPUs
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # in GB
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        'gpu_memory_used_gb': gpu_memory_used,
        'gpu_utilizations': gpu_utilizations  # Plural key now
    }

def get_efficientnet_b5(num_classes=10):
    # Load EfficientNet-B5 from torchvision (pretrained=False)
    model = models.efficientnet_b5(pretrained=False, num_classes=num_classes)
    # Modify the first convolution layer to better suit CIFAR10's 32x32 images.
    # In EfficientNet, features[0] is a ConvBNActivation block; we replace its conv layer.
    # The default stem uses a 3x3 kernel with stride 2. For CIFAR10, we set stride=1.
    orig_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=3,
        out_channels=orig_conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    return model

def train_efficientnet_b5():
    n_gpus = get_gpu_count()
    
    # Configuration updated for CIFAR10 and EfficientNet-B5
    config = {
        'model_name': 'efficientnet_b5',
        'dataset': 'cifar10',
        'batch_size': 128 * max(1, n_gpus),  # Scale batch size with number of GPUs
        'epochs': 100,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cost_per_hour': 3.91,
        'instance_type': f"gpu_node_{n_gpus}gpus" if n_gpus > 0 else "cpu_node",
        'input_dim': '32x32'
    }
    
    print(f"\nTraining Configuration:")
    print(f"Using device: {config['device']}")
    print(f"Number of GPUs: {n_gpus}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Total epochs: {config['epochs']}\n")
    
    # Initialize metrics DataFrame
    metrics_df = pd.DataFrame(columns=[
        'timestamp',
        'epoch',
        'model_name',
        'model_params',
        'model_layers',
        'dataset',
        'data_count',
        'data_size_mb',
        'data_dim',
        'instance_type',
        'device_type',
        'cpu_count',
        'gpu_count',
        'train_loss',
        'train_accuracy',
        'test_loss',
        'test_accuracy',
        'epoch_time',
        'time_so_far',
        'cpu_percent',
        'memory_used_gb',
        'gpu_memory_used_gb',
        'throughput',
        'cost_so_far'
    ])
    
    # Setup device and model
    device = torch.device(config['device'])
    model = get_efficientnet_b5(num_classes=10).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled across {n_gpus} GPUs")
    
    model_params = count_parameters(model)
    model_layers = count_layers(model)
    
    cpu_count = psutil.cpu_count()
    gpu_count = n_gpus
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    
    # Transform for CIFAR10 (RGB images with normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    
    # Use CIFAR10 dataset
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    data_count = len(train_dataset)
    data_size_mb = get_data_size(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    training_start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        time_so_far = epoch_end_time - training_start_time
        cost_so_far = (time_so_far / 3600) * config['cost_per_hour']
        throughput = len(train_dataset) / epoch_time
        
        # Get system metrics and create separate GPU utilization keys.
        sys_info = get_system_info()
        gpu_usage_dict = {}
        for i, usage in enumerate(sys_info['gpu_utilizations']):
            gpu_usage_dict[f'gpu_utilization_{i}'] = usage
        
        metrics = {
            'timestamp': datetime.now(),
            'epoch': epoch,
            'model_name': config['model_name'],
            'model_params': model_params,
            'model_layers': model_layers,
            'dataset': config['dataset'],
            'data_count': data_count,
            'data_size_mb': data_size_mb,
            'data_dim': config['input_dim'],
            'instance_type': config['instance_type'],
            'device_type': device_type,
            'cpu_count': cpu_count,
            'gpu_count': gpu_count,
            'train_loss': train_loss / len(train_loader),
            'train_accuracy': 100. * correct / total,
            'test_loss': test_loss / len(test_loader),
            'test_accuracy': 100. * test_correct / len(test_dataset),
            'epoch_time': epoch_time,
            'time_so_far': time_so_far,
            'cpu_percent': sys_info['cpu_percent'],
            'memory_used_gb': sys_info['memory_used_gb'],
            'gpu_memory_used_gb': sys_info['gpu_memory_used_gb'],
            'throughput': throughput,
            'cost_so_far': cost_so_far
        }
        metrics.update(gpu_usage_dict)
        
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        print(f"Epoch {epoch}: Loss: {metrics['train_loss']:.4f}, Acc: {metrics['test_accuracy']:.2f}%, "
              f"Time: {epoch_time:.2f}s, Total Time: {time_so_far:.2f}s, Cost: ${cost_so_far:.2f}")
        metrics_df.to_csv('log-cifar10-efficientnetB5.csv', index=False)
    
    total_time = time.time() - training_start_time
    total_cost = (total_time / 3600) * config['cost_per_hour']
    summary = {
        'timestamp': datetime.now(),
        'epoch': 'SUMMARY',
        'model_name': config['model_name'],
        'model_params': model_params,
        'model_layers': model_layers,
        'dataset': config['dataset'],
        'data_count': data_count,
        'data_size_mb': data_size_mb,
        'data_dim': config['input_dim'],
        'instance_type': config['instance_type'],
        'device_type': device_type,
        'cpu_count': cpu_count,
        'gpu_count': gpu_count,
        'train_loss': metrics_df['train_loss'].iloc[-1],
        'train_accuracy': metrics_df['train_accuracy'].iloc[-1],
        'test_loss': metrics_df['test_loss'].iloc[-1],
        'test_accuracy': metrics_df['test_accuracy'].iloc[-1],
        'epoch_time': total_time,
        'time_so_far': total_time,
        'cpu_percent': metrics_df['cpu_percent'].mean(),
        'memory_used_gb': metrics_df['memory_used_gb'].max(),
        'gpu_memory_used_gb': metrics_df['gpu_memory_used_gb'].max(),
        'throughput': metrics_df['throughput'].mean(),
        'cost_so_far': total_cost
    }
    
    # For each GPU, add a column with the average utilization over all epochs.
    for col in metrics_df.columns:
        if col.startswith('gpu_utilization_'):
            summary[col] = metrics_df[col].mean()
    
    metrics_df = pd.concat([metrics_df, pd.DataFrame([summary])], ignore_index=True)
    metrics_df.to_csv('log-cifar10-efficientnetB5.csv', index=False)
    
    # Format string for average GPU utilization per GPU
    gpu_util_str = ', '.join([f"GPU{i}: {summary.get(f'gpu_utilization_{i}', 0):.1f}%" for i in range(gpu_count)])
    
    print("\nTraining Summary:")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Final test accuracy: {summary['test_accuracy']:.2f}%")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Model parameters: {model_params:,}")
    print(f"Number of layers: {model_layers}")
    print(f"Peak GPU memory: {summary['gpu_memory_used_gb']:.2f} GB")
    print(f"Average GPU utilization: {gpu_util_str}")
    print(f"Dataset size: {data_size_mb:.2f} MB")
    print(f"Average throughput: {summary['throughput']:.2f} samples/sec")
    
    return metrics_df

if __name__ == "__main__":
    train_efficientnet_b5()
