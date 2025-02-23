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
	return total - 1  # Subtract 1 to exclude the model itself

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

class SimplifiedVGG16(nn.Module):
	def __init__(self, num_classes=10):
		super(SimplifiedVGG16, self).__init__()
		# Reduced number of filters for MNIST
		self.features = nn.Sequential(
			# Block 1: input 28x28x1 -> 28x28x32
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),  # -> 14x14x32
			
			# Block 2: 14x14x32 -> 14x14x64
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),  # -> 7x7x64
			
			# Block 3: 7x7x64 -> 7x7x128
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),  # -> 4x4x128
			
			# Block 4: 4x4x128 -> 4x4x256
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),  # -> 2x2x256
		)
		
		# Adjusted classifier for 2x2x256 input
		# Change this in SimplifiedVGG16 class:
		self.classifier = nn.Sequential(
			nn.Linear(256, 1024),  # Changed from 1024x1024 to 1024 input features
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(1024, 512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512, num_classes)
		)

		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x
	
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

def train_mnist():
	# Get GPU count
	n_gpus = torch.cuda.device_count()
	# n_gpus = get_gpu_count()
	
	# Configuration
	config = {
		'model_name': 'simplified_vgg16',
		'dataset': 'mnist',
		'batch_size': 64 * max(1, n_gpus),  # 64 per GPU
		'epochs': 100,
		'learning_rate': 0.0005,  # Reduced learning rate
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
	model = SimplifiedVGG16().to(device)
	
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
		metrics_df.to_csv('log-vgg16-mnist.csv', index=False)
	
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
	metrics_df.to_csv('log_mnist.csv', index=False)
	
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
