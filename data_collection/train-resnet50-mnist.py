import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
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
    return total - 1  

def get_data_size(dataset):
    sample, _ = dataset[0]
    single_sample_bytes = sample.element_size() * sample.nelement()
    total_bytes = single_sample_bytes * len(dataset)
    return total_bytes / (1024 * 1024)  # Convert to MB

def get_gpu_utilization():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return float(output.strip())
    except:
        return 0.0

def get_system_info():
    gpu_memory_used = 0
    gpu_utilization = get_gpu_utilization()
    
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        'gpu_memory_used_gb': gpu_memory_used,
        'gpu_utilization': gpu_utilization
    }

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        
        # Initial layers
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def train_mnist():
    # Get GPU count
    n_gpus = torch.cuda.device_count()
    
    # Configuration
    config = {
        'model_name': 'resnet50',  # Changed from vgg16 to resnet50
        'dataset': 'mnist',
        'batch_size': 64 * max(1, n_gpus),
        'epochs': 100,
        'learning_rate': 0.0005,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cost_per_hour': 5.0,
        'instance_type': f"gpu_node_{n_gpus}gpus" if n_gpus > 0 else "cpu_node",
        'input_dim': '28x28'
    }
    
    print(f"\nTraining Configuration:")
    print(f"Using device: {config['device']}")
    print(f"Number of GPUs: {n_gpus}")
    print(f"Batch size per GPU: 64")
    print(f"Total batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
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
        'gpu_utilization',
        'throughput',
        'cost_so_far'
    ])
    
    # Setup
    device = torch.device(config['device'])
    model = ResNet50().to(device)  # Changed to ResNet50
    
    # Enable multi-GPU if available
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled across {n_gpus} GPUs")
    
    # Get model info
    model_params = count_parameters(model)
    model_layers = count_layers(model)
    
    # Get system info
    cpu_count = psutil.cpu_count()
    gpu_count = n_gpus
    device_type = 'GPU' if torch.cuda.is_available() else 'CPU'
    
    # Minimal transforms - just normalize to [0,1] and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Get dataset info
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
        
        # Training loop
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
        
        # Test loop
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
        
        # Calculate metrics
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        time_so_far = epoch_end_time - training_start_time
        cost_so_far = (time_so_far / 3600) * config['cost_per_hour']
        throughput = len(train_dataset) / epoch_time
        
        # Get system metrics
        sys_info = get_system_info()
        
        # Record epoch metrics
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
            **sys_info,
            'throughput': throughput,
            'cost_so_far': cost_so_far
        }
        
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], 
                             ignore_index=True)
        
        print(f"Epoch {epoch}: "
              f"Loss: {metrics['train_loss']:.4f}, "
              f"Acc: {metrics['test_accuracy']:.2f}%, "
              f"GPU Mem: {metrics['gpu_memory_used_gb']:.1f}GB, "
              f"Time: {epoch_time:.2f}s, "
              f"Cost: ${cost_so_far:.2f}")
        
        # Save metrics after each epoch
        metrics_df.to_csv('log-resnet50-mnist.csv', index=False)
    
    # Add summary row
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
        'gpu_utilization': metrics_df['gpu_utilization'].mean(),
        'throughput': metrics_df['throughput'].mean(),
        'cost_so_far': total_cost
    }
    
    metrics_df = pd.concat([metrics_df, pd.DataFrame([summary])], 
                         ignore_index=True)
    metrics_df.to_csv('log-resnet50-mnist.csv', index=False)
    
    print("\nTraining Summary:")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Final test accuracy: {summary['test_accuracy']:.2f}%")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Model parameters: {model_params:,}")
    print(f"Number of layers: {model_layers}")
    print(f"Peak GPU memory: {summary['gpu_memory_used_gb']:.2f} GB")
    print(f"Average GPU utilization: {summary['gpu_utilization']:.1f}%")
    print(f"Dataset size: {data_size_mb:.2f} MB")
    print(f"Average throughput: {summary['throughput']:.2f} samples/sec")
    
    return metrics_df

if __name__ == "__main__":
    train_mnist()
