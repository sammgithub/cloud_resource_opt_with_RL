import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import time
import psutil
import pandas as pd
from datetime import datetime
import subprocess
import os
import numpy as np
from PIL import Image


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


# Let's fix the GWaveNet class, particularly the feature dimensions:

class GWaveNet(nn.Module):
	def __init__(self):
		super(GWaveNet, self).__init__()
		
		# First custom conv layer remains same
		k1 = np.array([
			[1, 0, 1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0, 1, 0],
			[1, 0, 1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0, 1, 0],
			[1, 0, 1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0, 1, 0],
			[1, 0, 1, 0, 1, 0, 1]
		])
		self.custom_kernel = torch.from_numpy(k1.reshape(1, 1, 7, 7)).float()
		
		self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding='same')
		with torch.no_grad():
			self.conv1.weight.copy_(self.custom_kernel)
			
		# Feature extraction
		self.features = nn.Sequential(
			nn.ReLU(),
			nn.MaxPool2d(2, 2, padding=1),  # 200 -> 100
			
			nn.Conv2d(1, 512, kernel_size=7, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),  # 100 -> 50
			
			nn.Conv2d(512, 256, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),  # 50 -> 25
			
			nn.Conv2d(256, 256, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),  # 25 -> 13
			
			nn.Conv2d(256, 128, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),  # 13 -> 7
			
			nn.Conv2d(128, 64, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),  # 7 -> 4
			
			nn.Conv2d(64, 32, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),  # 4 -> 2
		)
		
		# Calculate final feature size: 32 channels × 2 × 2 = 128
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32, 512),
			nn.ReLU(),
			nn.Linear(512, 1),
			nn.Sigmoid()
		)
		
		self.l2_reg = 1e-4

	# def forward(self, x):
	#     print("Input shape:", x.shape)
	#     x = self.conv1(x)
	#     print("After conv1 shape:", x.shape)
	#     x = self.features(x)
	#     print("After features shape:", x.shape)
	#     x = x.view(x.size(0), -1)  # This flattens to 32 features
	#     print("After flatten shape:", x.shape)
	#     # At this point x is [32, 32]
	#     x = self.classifier(x)
	#     print("Final shape:", x.shape)
	#     return x

	def forward(self, x):
		# Add print statements to debug dimensions
		x = self.conv1(x)
		x = self.features(x)
		x = self.classifier(x)
		return x
		
class GWDataset(Dataset):
	def __init__(self, data_dir, transform=None):
		self.data_dir = data_dir
		self.transform = transform
		self.samples = []
		
		# Load gw class
		gw_dir = os.path.join(data_dir, 'gw')
		for img_name in os.listdir(gw_dir):
			if img_name.endswith('.jpg') or img_name.endswith('.png'):
				self.samples.append((os.path.join(gw_dir, img_name), 1))
				
		# Load ngw class
		ngw_dir = os.path.join(data_dir, 'ngw')
		for img_name in os.listdir(ngw_dir):
			if img_name.endswith('.jpg') or img_name.endswith('.png'):
				self.samples.append((os.path.join(ngw_dir, img_name), 0))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		img_path, label = self.samples[idx]
		image = Image.open(img_path).convert('L')  # Convert to grayscale
		
		if self.transform:
			image = self.transform(image)
			
		return image, torch.tensor(label, dtype=torch.float32)

def train_gwavenet():
	# Get GPU count
	n_gpus = torch.cuda.device_count()
	
	# Configuration
	config = {
		'model_name': 'gwavenet',
		'dataset': 'gw_dataset',
		'batch_size': 32 * max(1, n_gpus),  # Reduced batch size for larger images
		'epochs': 100,
		'learning_rate': 0.01,  # SGD learning rate
		'device': 'cuda' if torch.cuda.is_available() else 'cpu',
		'cost_per_hour': 5.0,
		'instance_type': f"gpu_node_{n_gpus}gpus" if n_gpus > 0 else "cpu_node",
		'input_dim': '200x200'
	}
	
	print(f"\nTraining Configuration:")
	print(f"Using device: {config['device']}")
	print(f"Number of GPUs: {n_gpus}")
	print(f"Batch size per GPU: 32")
	print(f"Total batch size: {config['batch_size']}")
	print(f"Learning rate: {config['learning_rate']}")
	print(f"Total epochs: {config['epochs']}\n")
	
	# Initialize metrics DataFrame
	metrics_df = pd.DataFrame(columns=[
		'timestamp', 'epoch', 'model_name', 'model_params', 'model_layers',
		'dataset', 'data_count', 'data_size_mb', 'data_dim', 'instance_type',
		'device_type', 'cpu_count', 'gpu_count', 'train_loss', 'train_accuracy',
		'test_loss', 'test_accuracy', 'epoch_time', 'time_so_far', 'cpu_percent',
		'memory_used_gb', 'gpu_memory_used_gb', 'gpu_utilization', 'throughput',
		'cost_so_far', 'auc'
	])
	
	# Setup
	device = torch.device(config['device'])
	model = GWaveNet().to(device)
	
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
	
	# Data transforms
	transform = transforms.Compose([
		transforms.Resize((200, 200)),
		transforms.ToTensor(),
	])
	
	# Load datasets
	train_dataset = GWDataset('/home/ec2-user/data_augmented/train', transform=transform)
	test_dataset = GWDataset('/home/ec2-user/data_augmented/test', transform=transform)
	val_dataset = GWDataset('/home/ec2-user/data_augmented/validation', transform=transform)
	
	# Get dataset info
	data_count = len(train_dataset)
	data_size_mb = get_data_size(train_dataset)
	
	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
							shuffle=True, num_workers=4, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
						   shuffle=False, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
						  shuffle=False, num_workers=4, pin_memory=True)
	
	optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
	criterion = nn.BCELoss()
	
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
			loss = criterion(output.squeeze(), target)
			
			# Add L2 regularization
			l2_reg = torch.tensor(0.).to(device)
			for param in model.parameters():
				l2_reg += torch.norm(param)
			loss += model.module.l2_reg * l2_reg
			
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item()
			pred = (output.squeeze() > 0.5).float()
			correct += pred.eq(target).sum().item()
			total += target.size(0)
		
		# Test loop
		model.eval()
		test_loss = 0
		test_correct = 0
		all_preds = []
		all_targets = []
		
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				test_loss += criterion(output.squeeze(), target).item()
				pred = (output.squeeze() > 0.5).float()
				test_correct += pred.eq(target).sum().item()
				
				all_preds.extend(output.squeeze().cpu().numpy())
				all_targets.extend(target.cpu().numpy())
		
		# Calculate AUC
		from sklearn.metrics import roc_auc_score
		auc_score = roc_auc_score(all_targets, all_preds)
		
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
			'cost_so_far': cost_so_far,
			'auc': auc_score
		}
		
		metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], 
							 ignore_index=True)
		
		print(f"Epoch {epoch}: "
			  f"Loss: {metrics['train_loss']:.4f}, "
			  f"Acc: {metrics['test_accuracy']:.2f}%, "
			  f"AUC: {auc_score:.3f}, "
			  f"Time: {epoch_time:.2f}s, "
			  f"Cost: ${cost_so_far:.2f}")
		
		# Save metrics after each epoch
		metrics_df.to_csv('log-gwavent-gwave.csv', index=False)
	
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
		'cost_so_far': total_cost,
		'auc': metrics_df['auc'].iloc[-1]
	}
	
	metrics_df = pd.concat([metrics_df, pd.DataFrame([summary])], 
						 ignore_index=True)
	metrics_df.to_csv('log-gwavent-gwave.csv', index=False)
	
	print("\nTraining Summary:")
	print(f"Total training time: {total_time/3600:.2f} hours")
	print(f"Final test accuracy: {summary['test_accuracy']:.2f}%")
	print(f"Final AUC: {summary['auc']:.3f}")
	print(f"Total cost: ${total_cost:.2f}")
	print(f"Model parameters: {model_params:,}")
	print(f"Number of layers: {model_layers}")
	print(f"Peak GPU memory: {summary['gpu_memory_used_gb']:.2f} GB")
	print(f"Average GPU utilization: {summary['gpu_utilization']:.1f}%")
	print(f"Dataset size: {data_size_mb:.2f} MB")
	print(f"Average throughput: {summary['throughput']:.2f} samples/sec")
	
	return metrics_df

if __name__ == "__main__":
	train_gwavenet()
