#!/usr/bin/env python3
# Cloud Resource Optimizer with Reinforcement Learning
# This script optimizes cloud resources for ML training workloads

#################################################################
########################## SERAJ MOSTAFA ########################
######################## Dept. of IS, UMBC ######################
#################################################################

import os
import sys
import time
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque, defaultdict
import subprocess
import importlib.util
import psutil
import signal
import argparse
import glob
import logging
from typing import Dict, List, Tuple, Union, Any, Optional
from dataclasses import dataclass, field
import warnings
import multiprocessing

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.reshape.concat")

# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler("cloud_optimizer.log"),
		logging.StreamHandler()
	]
)
logger = logging.getLogger("cloud_optimizer")

# ==================== Configuration Classes ====================

@dataclass
class SLAConfig:
	time_max: float = None      # Maximum training time in seconds
	cost_max: float = None      # Maximum cost in dollars
	throughput_min: float = None  # Minimum samples/second
	memory_max: float = None    # Maximum memory usage in GB
	gpu_util_min: float = None  # Minimum GPU utilization percentage
	cpu_util_min: float = None  # Minimum CPU utilization percentage

	def __post_init__(self):
		# Set default values as fallback
		if self.time_max is None:
			self.time_max = 600.0  
		if self.cost_max is None:
			self.cost_max = 0.30  
		if self.throughput_min is None:
			self.throughput_min = 1800.0  
		if self.memory_max is None:
			self.memory_max = psutil.virtual_memory().total / (1024**3) * 0.5  
		if self.gpu_util_min is None:
			# Check current utilization
			current_util = self._get_current_gpu_utilization()
			if current_util is not None and current_util < 10.0:
				# Set a more realistic target for low-utilization scenarios
				self.gpu_util_min = min(current_util * 2.0, 20.0)
			else:
				self.gpu_util_min = 80.0
		if self.cpu_util_min is None:
			self.cpu_util_min = 70.0

	def _get_current_gpu_utilization(self) -> float:
		"""Get current average GPU utilization if available"""
		try:
			import subprocess
			output = subprocess.check_output(
				['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
				encoding='utf-8'
			)
			usages = [float(x.strip()) for x in output.strip().split('\n') if x.strip()]
			if usages:
				return sum(usages) / len(usages)
		except:
			pass
		return None

	@staticmethod
	def forecast_sla_values(baseline_history: list) -> dict:
		"""
		Forecast SLA thresholds based on a list of baseline metrics.
		Each entry in baseline_history is a dict with keys:
		  'time_so_far' (training time in seconds),
		  'cost_so_far' (cost in dollars), and
		  'throughput' (samples/sec).
		Returns a dict with forecasted 'time_max', 'cost_max', and 'throughput_min'.
		"""
		# import numpy as np

		# Gather lists of values from the history
		times = [entry['time_so_far'] for entry in baseline_history if 'time_so_far' in entry]
		costs = [entry['cost_so_far'] for entry in baseline_history if 'cost_so_far' in entry]
		throughputs = [entry['throughput'] for entry in baseline_history if 'throughput' in entry]

		# Compute the mean values; use fallback defaults if no values are available
		forecast_time = np.mean(times) if times else 600.0
		forecast_cost = np.mean(costs) if costs else 0.30
		forecast_throughput = np.mean(throughputs) if throughputs else 1800.0

		# Apply margins:
		# Allow 10% extra time and cost above the baseline
		# And require throughput to be at least 90% of the baseline performance
		forecast_time *= 1.1
		forecast_cost *= 1.1
		forecast_throughput *= 0.9

		return {
			'time_max': forecast_time,
			'cost_max': forecast_cost,
			'throughput_min': forecast_throughput
		}

	def update_from_baseline_history(self, baseline_history: list):
		"""
		Update the SLA thresholds using a history of baseline metrics.
		"""
		forecast = self.forecast_sla_values(baseline_history)
		self.time_max = forecast['time_max']
		self.cost_max = forecast['cost_max']
		self.throughput_min = forecast['throughput_min']

@dataclass
class ResourceConfig:
	"""Dynamic Resource Configuration"""
	# These will be populated automatically
	gpu_count_max: int = None  # Maximum number of GPUs
	gpu_count_min: int = None  # Minimum number of GPUs
	cpu_count_max: int = None  # Maximum number of CPUs
	cpu_count_min: int = None  # Minimum number of CPUs
	memory_max: int = None  # Maximum memory in GB
	memory_min: int = None  # Minimum memory in GB
	cost_per_gpu_hour: float = 10.0  # Cost per GPU hour
	cost_per_cpu_hour: float = 3.0  # Cost per CPU hour
	cost_per_gb_hour: float = 2.0  # Cost per GB hour

	def __post_init__(self):
		# Automatically detect system resources
		if self.gpu_count_max is None:
			self.gpu_count_max = torch.cuda.device_count() if torch.cuda.is_available() else 0
		if self.gpu_count_min is None:
			self.gpu_count_min = 1 if torch.cuda.is_available() else 0
		if self.cpu_count_max is None:
			self.cpu_count_max = psutil.cpu_count()
		if self.cpu_count_min is None:
			self.cpu_count_min = 1
		if self.memory_max is None:
			self.memory_max = int(psutil.virtual_memory().total / (1024**3))  # Convert to GB
		if self.memory_min is None:
			self.memory_min = 4  # 4GB minimum

@dataclass
class RLConfig:
	"""Reinforcement Learning Configuration"""
	# Notice we don't specify state_dim as it will be calculated dynamically
	hidden_dim: int = 256     # Hidden dimension for neural networks
	gamma: float = 0.99       # Discount factor
	tau: float = 0.005        # Target network update rate
	lr_actor: float = 3e-4    # Learning rate for actor
	lr_critic: float = 3e-4   # Learning rate for critic
	batch_size: int = 64      # Batch size for training
	buffer_size: int = 100000  # Replay buffer size
	exploration_noise: float = 0.1  # Exploration noise
	train_freq: int = 10      # Train every n steps
	reset_interval: int = 200  # Check for critic reset every n steps
	reset_probability: float = 0.3  # Probability of critic reset

@dataclass
class MultiObjectiveConfig:
	"""Multi-Objective Configuration"""
	num_policies: int = 4     # Number of policies (Pareto front approximation)
	# Preference vectors [cost_weight, time_weight, resource_weight]
	preference_vectors: List[List[float]] = field(default_factory=list)
	
	def __post_init__(self):
		if not self.preference_vectors:
			self.preference_vectors = [
				[0.33, 0.33, 0.34],  # Balanced
				[0.60, 0.20, 0.20],  # Cost-focused
				[0.20, 0.60, 0.20],  # Time-focused
				[0.20, 0.20, 0.60]   # Resource-focused
			]


# ==================== Offline Data Analysis ====================

class OfflineDataAnalyzer:
	"""Analyzes offline training logs to predict resource needs"""
	
	def __init__(self):
		self.dataframes = []
		self.combined_data = None
		self.model_characteristics = {}
		

	def add_log(self, log_path: str) -> bool:
		"""Add a CSV log file to the analyzer"""
		try:
			df = pd.read_csv(log_path)
			# Extract model name from the log
			model_name = df['model_name'].iloc[0] if 'model_name' in df.columns else os.path.basename(log_path)
			logger.info(f"Loaded log for model {model_name} with {len(df)} entries")
			
			# Store model characteristics for later prediction
			if 'model_params' in df.columns and 'model_layers' in df.columns:
				self.model_characteristics[model_name] = {
					'model_params': df['model_params'].iloc[0],
					'model_layers': df['model_layers'].iloc[0],
					'dataset': df['dataset'].iloc[0] if 'dataset' in df.columns else 'unknown',
					'data_count': df['data_count'].iloc[0] if 'data_count' in df.columns else 0,
					'data_size_mb': df['data_size_mb'].iloc[0] if 'data_size_mb' in df.columns else 0
				}
			
			self.dataframes.append(df)
			return True
		except Exception as e:
			logger.error(f"Failed to load log file {log_path}: {str(e)}")
			return False
	
	def analyze_all_logs(self):
		"""Analyze all loaded logs to find patterns"""
		if not self.dataframes:
			logger.warning("No logs to analyze")
			return
		
		# Combine all dataframes for analysis
		self.combined_data = pd.concat(self.dataframes, ignore_index=True)
		
		# Basic statistics about all training runs
		logger.info(f"Analyzed {len(self.dataframes)} logs with {len(self.combined_data)} total entries")
		logger.info(f"Average training time: {self.combined_data['time_so_far'].mean()/60:.2f} minutes")
		logger.info(f"Average cost: ${self.combined_data['cost_so_far'].mean():.2f}")
		
		# Find best configurations per model
		model_groups = self.combined_data.groupby('model_name')
		for model_name, group in model_groups:
			# Find best throughput configuration
			best_throughput_idx = group['throughput'].idxmax()
			best_throughput_config = group.loc[best_throughput_idx]
			
			# Find most cost-efficient configuration
			group['cost_efficiency'] = group['throughput'] / (group['cost_so_far'] + 0.001)
			best_cost_idx = group['cost_efficiency'].idxmax()
			best_cost_config = group.loc[best_cost_idx]
			
			logger.info(f"Model {model_name}:")
			logger.info(f"  Best throughput: {best_throughput_config['throughput']:.2f} samples/sec with "
					  f"{best_throughput_config['gpu_count']} GPUs, {best_throughput_config['cpu_count']} CPUs")
			logger.info(f"  Best cost efficiency: {best_cost_config['cost_efficiency']:.2f} throughput/$ with "
					  f"{best_cost_config['gpu_count']} GPUs, {best_cost_config['cpu_count']} CPUs")
	
	def predict_resources(self, model_info: Dict) -> Dict:
		"""Predict optimal resources based on model and dataset characteristics"""
		# Simple prediction for now, could be enhanced with ML models
		if not self.model_characteristics:
			# No historical data, use basic heuristics
			return self._default_prediction(model_info)
		
		# Find similar models based on parameter count and layer count
		similarities = []
		
		for model_name, chars in self.model_characteristics.items():
			# Calculate similarity based on model size and dataset size
			model_size_similarity = min(
				model_info.get('model_params', 0) / max(1, chars['model_params']),
				chars['model_params'] / max(1, model_info.get('model_params', 0))
			)
			
			layer_similarity = min(
				model_info.get('model_layers', 0) / max(1, chars['model_layers']),
				chars['model_layers'] / max(1, model_info.get('model_layers', 0))
			)
			
			data_size_similarity = 1.0
			if 'data_size_mb' in model_info and chars['data_size_mb'] > 0:
				data_size_similarity = min(
					model_info['data_size_mb'] / chars['data_size_mb'],
					chars['data_size_mb'] / model_info['data_size_mb']
				)
			
			# Overall similarity score (higher is better)
			similarity = (0.5 * model_size_similarity + 
						  0.3 * layer_similarity + 
						  0.2 * data_size_similarity)
			
			similarities.append((model_name, similarity))
		
		# Sort by similarity (highest first)
		similarities.sort(key=lambda x: x[1], reverse=True)
		
		if similarities and similarities[0][1] > 0.7:
			# Found a similar model, use its best configuration
			similar_model = similarities[0][0]
			similar_model_data = self.combined_data[self.combined_data['model_name'] == similar_model]
			
			# Get best balanced configuration (optimize for throughput/cost)
			similar_model_data['efficiency'] = similar_model_data['throughput'] / (similar_model_data['cost_so_far'] + 0.1)
			best_config_idx = similar_model_data['efficiency'].idxmax()
			best_config = similar_model_data.loc[best_config_idx]
			
			logger.info(f"Found similar model: {similar_model} with similarity {similarities[0][1]:.2f}")
			logger.info(f"Using its best configuration as starting point")
			
			return {
				'gpu_count': int(best_config['gpu_count']),
				'cpu_count': int(best_config['cpu_count']),
				'memory_gb': int(best_config.get('memory_used_gb', 16) * 1.2),  # Add 20% buffer
				'batch_size': int(best_config.get('batch_size', 64)),
				'estimated_time': best_config['time_so_far'],
				'estimated_cost': best_config['cost_so_far'],
				'similarity_score': similarities[0][1]
			}
		else:
			# No similar model found, use default prediction
			return self._default_prediction(model_info)
	
	def _default_prediction(self, model_info: Dict) -> Dict:
		"""Make a default prediction when no similar models are found"""
		model_params = model_info.get('model_params', 0)
		model_layers = model_info.get('model_layers', 0)
		data_size_mb = model_info.get('data_size_mb', 0)
		
		# Basic heuristics for resource allocation
		if model_params < 10_000_000:  # Small model
			gpu_count = 1
			cpu_count = 4
			memory_gb = 8
		elif model_params < 50_000_000:  # Medium model
			gpu_count = 2
			cpu_count = 8
			memory_gb = 16
		else:  # Large model
			gpu_count = 4
			cpu_count = 16
			memory_gb = 32
		
		# Adjust based on dataset size
		if data_size_mb > 1000:
			gpu_count = min(gpu_count + 1, 8)
			memory_gb += 8
		
		# Batch size heuristic
		batch_size = 32 * gpu_count
		
		# Rough estimates for time and cost
		estimated_time = (model_params / 1_000_000) * (data_size_mb / 100) * 600  # in seconds
		estimated_cost = estimated_time / 3600 * (gpu_count * 2.5 + cpu_count * 0.1 + memory_gb * 0.02)
		
		return {
			'gpu_count': gpu_count,
			'cpu_count': cpu_count,
			'memory_gb': memory_gb,
			'batch_size': batch_size,
			'estimated_time': estimated_time,
			'estimated_cost': estimated_cost,
			'similarity_score': 0.0
		}

	def update_with_result(self, result: Dict):
		"""Update the analyzer with a new training result"""
		# This would be called after each training run to incorporate new knowledge
		pass

# ==================== CSV Monitor ====================

class CSVMonitor:
	"""Monitors and processes CSV files in real-time"""
	
	def __init__(self, csv_path: str, update_interval: float = 1.0):
		self.csv_path = csv_path
		self.update_interval = update_interval
		self.last_modified_time = 0
		self.last_row_count = 0
		self.df = None
		self.extended_columns = [
			'rl_policy_used', 'rl_action_taken', 'rl_reward', 
			'rl_sla_time_met', 'rl_sla_cost_met', 'rl_sla_resource_met',
			'rl_critic_reset'
		]
	
	def check_for_updates(self) -> bool:
		"""Check if the CSV file has been updated"""
		if not os.path.exists(self.csv_path):
			return False
		
		current_mtime = os.path.getmtime(self.csv_path)
		if current_mtime > self.last_modified_time:
			self.last_modified_time = current_mtime
			return True
		
		return False
	
	def load_latest_data(self) -> pd.DataFrame:
		"""Load the latest data from the CSV file"""
		try:
			if not os.path.exists(self.csv_path):
				# CSV doesn't exist yet, create an empty dataframe
				self.df = pd.DataFrame()
				return self.df.copy()
			
			self.df = pd.read_csv(self.csv_path)
			self.last_row_count = len(self.df)
			return self.df.copy()
		except Exception as e:
			logger.error(f"Error loading CSV file: {str(e)}")
			return pd.DataFrame()
	
	def get_new_rows(self) -> pd.DataFrame:
		"""Get only the newly added rows from the CSV file"""
		current_df = self.load_latest_data()
		if len(current_df) > self.last_row_count:
			new_rows = current_df.iloc[self.last_row_count:]
			self.last_row_count = len(current_df)
			return new_rows
		return pd.DataFrame()
	
	def append_rl_metrics(self, epoch: int, metrics: Dict) -> bool:
		"""Append RL metrics to the CSV file"""
		try:
			df = self.load_latest_data()
			if 'epoch' not in df.columns or len(df) == 0:
				logger.warning("Cannot append RL metrics: CSV file empty or missing 'epoch' column")
				return False
			
			# Find row with matching epoch
			mask = df['epoch'] == epoch
			if not mask.any():
				logger.warning(f"Epoch {epoch} not found in CSV")
				return False
			
			# Add RL columns if they don't exist
			for col in self.extended_columns:
				if col not in df.columns:
					df[col] = None
			
			# Update row with RL metrics
			for key, value in metrics.items():
				col_name = f"rl_{key}" if not key.startswith('rl_') else key
				if col_name in df.columns:
					df.loc[mask, col_name] = value
			
			# Save back to CSV
			df.to_csv(self.csv_path, index=False)
			return True
		except Exception as e:
			logger.error(f"Error appending RL metrics: {str(e)}")
			return False
	
	def create_extended_csv(self, original_csv: str, rl_metrics: pd.DataFrame) -> str:
		"""Create an extended CSV with original data plus RL metrics"""
		try:
			# Load original data
			df_original = pd.read_csv(original_csv)
			
			# Merge with RL metrics on epoch column
			df_extended = pd.merge(df_original, rl_metrics, on='epoch', how='left')
			
			# Generate output filename
			base_name = os.path.splitext(original_csv)[0]
			output_path = f"{base_name}_rl_extended.csv"
			
			# Save to new file
			df_extended.to_csv(output_path, index=False)
			logger.info(f"Created extended CSV: {output_path}")
			
			return output_path
		except Exception as e:
			logger.error(f"Error creating extended CSV: {str(e)}")
			return ""

# ==================== Training Module Management ====================

class TrainingModuleManager:
	"""Manages loading and running external training modules"""
	
	def __init__(self):
		self.module = None
		self.module_path = None
		self.training_process = None
		self.csv_monitor = None
		self.output_csv_path = None
		self.resource_config = ResourceConfig()  # Added this line
	
	
	def load_module(self, module_path: str) -> bool:
		"""Load a Python module from a file path"""
		try:
			module_name = os.path.splitext(os.path.basename(module_path))[0]
			spec = importlib.util.spec_from_file_location(module_name, module_path)
			module = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(module)

			self.module = module
			self.module_path = module_path

			# ✅ Dynamic detection first
			train_funcs = [name for name in dir(module) if name.startswith('train_') and callable(getattr(module, name))]
			if train_funcs:
				logger.info(f"Found training function: {train_funcs[0]} in {module_path}")
				return True

			# Optional fallback (legacy static names)
			for func_name in ['train', 'train_model', 'main']:
				if hasattr(module, func_name) and callable(getattr(module, func_name)):
					logger.info(f"Found training function: {func_name} in {module_path}")
					return True

			logger.warning(f"Could not find training function in {module_path}")
			return False
		except Exception as e:
			logger.error(f"Error loading module {module_path}: {str(e)}")
			return False


	def analyze_module(self) -> Dict:
		"""Analyze the loaded module to extract model and dataset information"""
		if not self.module:
			logger.error("No module loaded")
			return {}
		
		info = {
			'module_name': os.path.basename(self.module_path),
			'has_gpu_support': False,
			'model_type': 'unknown',
			'dataset_type': 'unknown'
		}
		
		# Scan the source code to extract information
		with open(self.module_path, 'r') as f:
			source = f.read()
			
			# Check for GPU usage
			if 'cuda' in source or 'gpu' in source.lower():
				info['has_gpu_support'] = True
			
			# Try to identify model type
			for model_type in ['resnet', 'vgg', 'inception', 'transformer', 'bert', 'gpt']:
				if model_type in source.lower():
					info['model_type'] = model_type
					break
			
			# Try to identify dataset
			for dataset in ['cifar', 'imagenet', 'mnist', 'coco', 'pascal']:
				if dataset in source.lower():
					info['dataset_type'] = dataset
					break
		
		return info
	

	def run_training(self, gpu_count: int = None, cpu_count: int = None, batch_size: int = None) -> bool:
		"""Run the training module in a separate process with controlled resources"""
		if not self.module:
			logger.error("No module loaded")
			return False
			
		# Set environment variables for data loading optimization
		if gpu_count is not None and gpu_count > 0:
			# Check GPU utilization first
			try:
				import subprocess
				output = subprocess.check_output(
					['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
					encoding='utf-8'
				)
				usages = [float(x.strip()) for x in output.strip().split('\n') if x.strip()]
				avg_util = sum(usages) / len(usages) if usages else 0
				
				# For low GPU utilization, optimize data loading
				if avg_util < 10.0:
					os.environ['OPTIMIZER_NUM_WORKERS'] = str(min(8, os.cpu_count() or 2))
					os.environ['OPTIMIZER_PIN_MEMORY'] = 'true'
					logger.info(f"Setting optimized data loading for low GPU utilization ({avg_util:.1f}%)")
				else:
					os.environ['OPTIMIZER_NUM_WORKERS'] = str(min(4, os.cpu_count() or 1))
					os.environ['OPTIMIZER_PIN_MEMORY'] = 'true'
			except:
				# Default values if we can't check utilization
				os.environ['OPTIMIZER_NUM_WORKERS'] = str(min(4, os.cpu_count() or 1))
				os.environ['OPTIMIZER_PIN_MEMORY'] = 'true'
		
		# Before calling train_func, ALWAYS set these environment variables:
		os.environ['COST_PER_GPU_HOUR'] = str(self.resource_config.cost_per_gpu_hour)
		os.environ['COST_PER_CPU_HOUR'] = str(self.resource_config.cost_per_cpu_hour)
		os.environ['COST_PER_GB_HOUR'] = str(self.resource_config.cost_per_gb_hour)
		
		train_func = None

		# Static fallback for known names
		for func_name in ['train', 'train_model', 'train_resnet50', 'main']:
			if hasattr(self.module, func_name) and callable(getattr(self.module, func_name)):
				train_func = getattr(self.module, func_name)
				break

		# 🔥 Dynamic fallback: any function starting with train_
		if not train_func:
			train_funcs = [name for name in dir(self.module) if name.startswith('train_') and callable(getattr(self.module, name))]
			if train_funcs:
				train_func = getattr(self.module, train_funcs[0])
				logger.info(f"Detected training function dynamically: {train_funcs[0]}")

		if not train_func:
			logger.error("Could not find training function")
			return False

		# Set environment variables for resource constraints
		original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
		original_omp_threads = os.environ.get('OMP_NUM_THREADS', None)
		original_mkl_threads = os.environ.get('MKL_NUM_THREADS', None)
		
		try:
			# Restrict GPU visibility - this actually controls which GPUs are used
			if gpu_count is not None:
				if gpu_count == 0:
					# No GPU access
					os.environ['CUDA_VISIBLE_DEVICES'] = ""
					logger.info("Setting CUDA_VISIBLE_DEVICES to empty string (no GPU access)")
				else:
					available_gpus = list(range(torch.cuda.device_count()))
					if available_gpus:
						# Use only the first gpu_count GPUs
						visible_gpus = available_gpus[:gpu_count]
						os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, visible_gpus))
						logger.info(f"Setting CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")
			
			# Control CPU threads
			if cpu_count is not None:
				os.environ['OMP_NUM_THREADS'] = str(cpu_count)
				os.environ['MKL_NUM_THREADS'] = str(cpu_count)
				os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
				os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
				logger.info(f"Setting thread environment variables to {cpu_count}")
			
			# Control batch size
			if batch_size is not None:
				# Set environment variable
				os.environ['OPTIMIZER_BATCH_SIZE'] = str(batch_size)
				
				# Try to modify batch_size directly in the module
				if hasattr(self.module, 'batch_size'):
					try:
						self.module.batch_size = batch_size
						logger.info(f"Set module batch_size attribute to {batch_size}")
					except:
						pass
				
				# Sometimes batch_size might be inside a config dictionary
				if hasattr(self.module, 'config') and isinstance(self.module.config, dict):
					if 'batch_size' in self.module.config:
						self.module.config['batch_size'] = batch_size
						logger.info(f"Set config batch_size to {batch_size}")
			
			# Get initial CSV files before training
			initial_csv_files = set(glob.glob("log-*.csv"))
			
			# Run the training function
			logger.info(f"Starting training with gpu_count={gpu_count}, cpu_count={cpu_count}, batch_size={batch_size}")
			result = train_func()
			
			# Find newly created CSV files (or modified ones)
			after_csv_files = set(glob.glob("log-*.csv"))
			new_csv_files = after_csv_files - initial_csv_files
			
			if new_csv_files:
				# Use the most recently modified new CSV
				self.output_csv_path = max(new_csv_files, key=os.path.getmtime)
				logger.info(f"Training completed, output CSV: {self.output_csv_path}")
			elif after_csv_files:
				# If no new files were created, use the most recently modified existing CSV
				recent_csv = max(after_csv_files, key=os.path.getmtime)
				# Check if the file was modified during this training run
				if os.path.getmtime(recent_csv) > time.time() - 300:  # Modified in the last 5 minutes
					self.output_csv_path = recent_csv
					logger.info(f"Training completed, using modified CSV: {self.output_csv_path}")
				else:
					logger.warning("Training completed but no new or recently modified CSV files found")
					return False
			else:
				# Try to find any CSV files
				csv_files = glob.glob("*.csv")
				if csv_files:
					# Use the most recently modified CSV
					self.output_csv_path = max(csv_files, key=os.path.getmtime)
					logger.info(f"Training completed, found CSV: {self.output_csv_path}")
				else:
					logger.warning("Training completed but no output CSV found")
					return False
			
			# Create a timestamped copy of the output CSV to prevent overwriting
			if self.output_csv_path and os.path.exists(self.output_csv_path):
				# Create a timestamped copy of the output CSV
				timestamp = int(time.time())
				if gpu_count is not None:  # This is the optimized run
					run_type = "optimized"
				else:
					run_type = "baseline"
				
				# Create new path with timestamp and run type
				file_base, file_ext = os.path.splitext(self.output_csv_path)
				new_path = f"{file_base}_{run_type}_{timestamp}{file_ext}"
				
				# Copy the file
				import shutil
				shutil.copy2(self.output_csv_path, new_path)
				logger.info(f"Created preserved copy of {run_type} metrics: {new_path}")
				
				# Update the path to use the timestamped version
				self.output_csv_path = new_path
			
			return True
		
		except Exception as e:
			logger.error(f"Error running training: {str(e)}")
			return False
		
		finally:
			# Restore original environment variables
			if original_cuda_visible_devices is not None:
				os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
			elif 'CUDA_VISIBLE_DEVICES' in os.environ:
				del os.environ['CUDA_VISIBLE_DEVICES']
				
			if original_omp_threads is not None:
				os.environ['OMP_NUM_THREADS'] = original_omp_threads
			elif 'OMP_NUM_THREADS' in os.environ:
				del os.environ['OMP_NUM_THREADS']
				
			if original_mkl_threads is not None:
				os.environ['MKL_NUM_THREADS'] = original_mkl_threads
			elif 'MKL_NUM_THREADS' in os.environ:
				del os.environ['MKL_NUM_THREADS']


	def get_model_info(self, csv_path: str = None) -> Dict:
		"""Extract model and dataset information from CSV file"""
		if csv_path is None:
			csv_path = self.output_csv_path
		
		if not csv_path or not os.path.exists(csv_path):
			logger.warning("No CSV file available to extract model info")
			return {}
		
		try:
			df = pd.read_csv(csv_path)
			if len(df) == 0:
				return {}
			
			# Extract information from first row
			info = {}
			for col in ['model_name', 'model_params', 'model_layers', 'dataset', 
						'data_count', 'data_size_mb', 'data_dim']:
				if col in df.columns:
					info[col] = df[col].iloc[0]
			
			return info
		except Exception as e:
			logger.error(f"Error extracting model info from CSV: {str(e)}")
			return {}
	
	def generate_summary(self) -> Dict:
		"""Generate a summary of the training run"""
		if not self.output_csv_path or not os.path.exists(self.output_csv_path):
			logger.warning("No output CSV file to generate summary")
			return {}
		
		try:
			df = pd.read_csv(self.output_csv_path)
			if len(df) == 0:
				return {}
			
			# Find summary row or use the last row
			if 'epoch' in df.columns and 'SUMMARY' in df['epoch'].values:
				# summary_row = df[df['epoch'] == 'SUMMARY'].iloc[0] ## changed to the next line as summary removed from train.py
				summary_row = df.iloc[-1]  # Use last real epoch

			else:
				summary_row = df.iloc[-1]
			
			# Extract key metrics
			summary = {}
			for col in ['model_name', 'dataset', 'time_so_far', 'cost_so_far', 
					  'throughput', 'cpu_percent', 'memory_used_gb', 'gpu_memory_used_gb']:
				if col in summary_row.index:
					summary[col] = summary_row[col]
			
			# Make sure we correctly capture GPU count - use maximum detected during training
			if 'gpu_count' in summary_row.index:
				summary['gpu_count'] = summary_row['gpu_count']
			else:
				# Count how many GPU utilization columns exist with non-zero values
				gpu_util_columns = [col for col in summary_row.index if col.startswith('gpu_utilization_')]
				active_gpus = 0
				for col in gpu_util_columns:
					if not pd.isna(summary_row[col]) and summary_row[col] > 0:
						active_gpus += 1
				
				# If no active GPUs detected but GPU memory used, set to at least 1
				if active_gpus == 0 and 'gpu_memory_used_gb' in summary_row.index and summary_row['gpu_memory_used_gb'] > 0:
					active_gpus = 1
					
				summary['gpu_count'] = active_gpus
			
			# Ensure CPU count is captured
			if 'cpu_count' in summary_row.index:
				summary['cpu_count'] = summary_row['cpu_count']
			else:
				# Default to system CPU count if not specified
				summary['cpu_count'] = psutil.cpu_count(logical=False) or 4
			
			# Add GPU utilization metrics
			gpu_util_columns = [col for col in summary_row.index if col.startswith('gpu_utilization_')]
			gpu_utils = {}
			for col in gpu_util_columns:
				if not pd.isna(summary_row[col]):
					gpu_utils[col] = summary_row[col]
			
			if gpu_utils:
				summary['gpu_utilizations'] = gpu_utils
			
			return summary
		except Exception as e:
			logger.error(f"Error generating summary: {str(e)}")
			return {}


# ==================== Core MDP Components ====================

class CloudEnvironmentState:
	"""State representation for the cloud environment"""
	
	def __init__(self, config: Dict):
		self.resource_config = ResourceConfig(**config.get('resource', {}))
		self.sla_config = SLAConfig(**config.get('sla', {}))
		
		# Define all state components
		self.state_components = {
			# Resource allocation
			'gpu_count': 0,
			'cpu_count': 0,
			'memory_gb': 0,
			
			# Resource utilization
			'gpu_util': 0.0,
			'cpu_util': 0.0,
			'memory_util': 0.0,
			
			# Training progress
			'current_epoch': 0,
			'total_epochs': 0,
			'epoch_progress': 0.0,
			'throughput': 0.0,
			
			# Cost and time
			'time_elapsed': 0.0,
			'time_remaining': 0.0,
			'cost_so_far': 0.0,
			'cost_per_epoch': 0.0,
			
			# SLA status (binary indicators)
			'sla_time_met': 1,
			'sla_cost_met': 1,
			'sla_throughput_met': 1,
			'sla_memory_met': 1,
			'sla_gpu_util_met': 1,
			'sla_cpu_util_met': 1,
			
			# SLA violation severity (normalized)
			'sla_time_severity': 0.0,
			'sla_cost_severity': 0.0,
			'sla_throughput_severity': 0.0,
			'sla_memory_severity': 0.0,
			'sla_gpu_util_severity': 0.0,
			'sla_cpu_util_severity': 0.0
		}
		
		# Calculate state dimension
		self.state_dim = len(self.state_components)
	
	def update_from_metrics(self, metrics: Dict):
		"""Update state from training metrics"""
		# Update resource allocation
		if 'gpu_count' in metrics:
			self.state_components['gpu_count'] = metrics['gpu_count']
		if 'cpu_count' in metrics:
			self.state_components['cpu_count'] = metrics['cpu_count']
		if 'memory_used_gb' in metrics:
			self.state_components['memory_gb'] = metrics['memory_used_gb']
		
		# Update resource utilization
		if 'cpu_percent' in metrics:
			self.state_components['cpu_util'] = metrics['cpu_percent'] / 100.0
		
		# Handle GPU utilization - may have multiple GPUs
		gpu_util_sum = 0.0
		gpu_util_count = 0
		for key in metrics:
			if key.startswith('gpu_utilization_'):
				if not pd.isna(metrics[key]):
					gpu_util_sum += metrics[key]
					gpu_util_count += 1
		
		if gpu_util_count > 0:
			self.state_components['gpu_util'] = gpu_util_sum / (100.0 * gpu_util_count)
		
		# Update memory utilization
		if 'memory_used_gb' in metrics and 'memory_gb' in self.state_components and self.state_components['memory_gb'] > 0:
			self.state_components['memory_util'] = metrics['memory_used_gb'] / self.state_components['memory_gb']
		
		# Update training progress
		if 'epoch' in metrics:
			try:
				self.state_components['current_epoch'] = int(metrics['epoch'])
			except:
				# Handle case where epoch might be 'SUMMARY' or other non-integer
				pass
				
		if 'total_epochs' in metrics:
			self.state_components['total_epochs'] = metrics['total_epochs']
			
		if self.state_components['total_epochs'] > 0:
			self.state_components['epoch_progress'] = self.state_components['current_epoch'] / self.state_components['total_epochs']
			
		if 'throughput' in metrics:
			self.state_components['throughput'] = metrics['throughput'] / 1000.0  # Normalize to thousands
		
		# Update cost and time
		if 'time_so_far' in metrics:
			self.state_components['time_elapsed'] = metrics['time_so_far']
			
		if 'cost_so_far' in metrics:
			self.state_components['cost_so_far'] = metrics['cost_so_far']
			
		if 'epoch_time' in metrics and self.state_components['current_epoch'] > 0:
			avg_time_per_epoch = metrics['time_so_far'] / self.state_components['current_epoch']
			epochs_remaining = self.state_components['total_epochs'] - self.state_components['current_epoch']
			self.state_components['time_remaining'] = avg_time_per_epoch * epochs_remaining
		
		# Update the SLA status section in CloudEnvironmentState.update_from_metrics method

		# Time SLA - would require a stricter time limit
		if 'time_so_far' in metrics:
			self.state_components['sla_time_met'] = 1 if metrics['time_so_far'] <= self.sla_config.time_max else 0
			if not self.state_components['sla_time_met']:
				severity = (metrics['time_so_far'] - self.sla_config.time_max) / self.sla_config.time_max
				self.state_components['sla_time_severity'] = min(1.0, severity)

		# Cost SLA - would require a stricter cost limit
		if 'cost_so_far' in metrics:
			self.state_components['sla_cost_met'] = 1 if metrics['cost_so_far'] <= self.sla_config.cost_max else 0
			if not self.state_components['sla_cost_met']:
				severity = (metrics['cost_so_far'] - self.sla_config.cost_max) / self.sla_config.cost_max
				self.state_components['sla_cost_severity'] = min(1.0, severity)

		# Throughput SLA - require higher throughput
		if 'throughput' in metrics:
			self.state_components['sla_throughput_met'] = 1 if metrics['throughput'] >= self.sla_config.throughput_min else 0
			if not self.state_components['sla_throughput_met']:
				severity = (self.sla_config.throughput_min - metrics['throughput']) / self.sla_config.throughput_min
				self.state_components['sla_throughput_severity'] = min(1.0, severity)

		
		# Update SLA status
		# Time SLA
		if 'time_so_far' in metrics:
			self.state_components['sla_time_met'] = 1 if metrics['time_so_far'] <= self.sla_config.time_max else 0
			if not self.state_components['sla_time_met']:
				severity = (metrics['time_so_far'] - self.sla_config.time_max) / self.sla_config.time_max
				self.state_components['sla_time_severity'] = min(1.0, severity)
		
		# Cost SLA
		if 'cost_so_far' in metrics:
			self.state_components['sla_cost_met'] = 1 if metrics['cost_so_far'] <= self.sla_config.cost_max else 0
			if not self.state_components['sla_cost_met']:
				severity = (metrics['cost_so_far'] - self.sla_config.cost_max) / self.sla_config.cost_max
				self.state_components['sla_cost_severity'] = min(1.0, severity)
		
		# Throughput SLA
		if 'throughput' in metrics:
			self.state_components['sla_throughput_met'] = 1 if metrics['throughput'] >= self.sla_config.throughput_min else 0
			if not self.state_components['sla_throughput_met']:
				severity = (self.sla_config.throughput_min - metrics['throughput']) / self.sla_config.throughput_min
				self.state_components['sla_throughput_severity'] = min(1.0, severity)
		
		# Memory SLA
		if 'memory_used_gb' in metrics:
			self.state_components['sla_memory_met'] = 1 if metrics['memory_used_gb'] <= self.sla_config.memory_max else 0
			if not self.state_components['sla_memory_met']:
				severity = (metrics['memory_used_gb'] - self.sla_config.memory_max) / self.sla_config.memory_max
				self.state_components['sla_memory_severity'] = min(1.0, severity)
		
		# GPU Utilization SLA - require higher utilization
		if 'gpu_util' in self.state_components:
			gpu_util_pct = self.state_components['gpu_util'] * 100.0
			self.state_components['sla_gpu_util_met'] = 1 if gpu_util_pct >= self.sla_config.gpu_util_min else 0
			if not self.state_components['sla_gpu_util_met']:
				severity = (self.sla_config.gpu_util_min - gpu_util_pct) / self.sla_config.gpu_util_min
				self.state_components['sla_gpu_util_severity'] = min(1.0, severity)

	
		# CPU Utilization SLA
		if 'cpu_percent' in metrics:
			self.state_components['sla_cpu_util_met'] = 1 if metrics['cpu_percent'] >= self.sla_config.cpu_util_min else 0
			if not self.state_components['sla_cpu_util_met']:
				severity = (self.sla_config.cpu_util_min - metrics['cpu_percent']) / self.sla_config.cpu_util_min
				self.state_components['sla_cpu_util_severity'] = min(1.0, severity)
	
	def get_vector(self) -> np.ndarray:
		"""Get the state as a normalized vector for the RL agent"""
		# Ensure consistent ordering and fixed-length vector
		return np.array([
			self.state_components.get('gpu_count', 0) / max(1, self.resource_config.gpu_count_max),
			self.state_components.get('cpu_count', 0) / max(1, self.resource_config.cpu_count_max),
			self.state_components.get('memory_gb', 0) / max(1, self.resource_config.memory_max),

			self.state_components.get('gpu_util', 0.0),
			self.state_components.get('cpu_util', 0.0),
			self.state_components.get('memory_util', 0.0),

			self.state_components.get('epoch_progress', 0.0),
			min(1.0, self.state_components.get('throughput', 0.0)),

			min(1.0, self.state_components.get('time_elapsed', 0.0) / self.sla_config.time_max),
			min(1.0, self.state_components.get('cost_so_far', 0.0) / self.sla_config.cost_max),

			self.state_components.get('sla_time_met', 0),
			self.state_components.get('sla_cost_met', 0),
			self.state_components.get('sla_throughput_met', 0),
			self.state_components.get('sla_memory_met', 0),
			self.state_components.get('sla_gpu_util_met', 0),
			self.state_components.get('sla_cpu_util_met', 0),

			self.state_components.get('sla_time_severity', 0.0),
			self.state_components.get('sla_cost_severity', 0.0),
			self.state_components.get('sla_throughput_severity', 0.0),
			self.state_components.get('sla_memory_severity', 0.0),
			self.state_components.get('sla_gpu_util_severity', 0.0),
			self.state_components.get('sla_cpu_util_severity', 0.0),
			
			# Pad to match expected 26-dim if needed
			0.0, 0.0, 0.0, 0.0
		], dtype=np.float32)

	
	def get_sla_status(self) -> Dict:
		"""Get the current SLA status"""
		return {
			'time': {
				'met': bool(self.state_components['sla_time_met']),
				'severity': self.state_components['sla_time_severity']
			},
			'cost': {
				'met': bool(self.state_components['sla_cost_met']),
				'severity': self.state_components['sla_cost_severity']
			},
			'throughput': {
				'met': bool(self.state_components['sla_throughput_met']),
				'severity': self.state_components['sla_throughput_severity']
			},
			'memory': {
				'met': bool(self.state_components['sla_memory_met']),
				'severity': self.state_components['sla_memory_severity']
			},
			'gpu_util': {
				'met': bool(self.state_components['sla_gpu_util_met']),
				'severity': self.state_components['sla_gpu_util_severity']
			},
			'cpu_util': {
				'met': bool(self.state_components['sla_cpu_util_met']),
				'severity': self.state_components['sla_cpu_util_severity']
			}
		}
	
	def get_overall_sla_compliance(self) -> float:
		"""Get the overall SLA compliance rate (0.0 to 1.0)"""
		sla_statuses = [
			self.state_components['sla_time_met'],
			self.state_components['sla_cost_met'],
			self.state_components['sla_throughput_met'],
			self.state_components['sla_memory_met'],
			self.state_components['sla_gpu_util_met'],
			self.state_components['sla_cpu_util_met']
		]
		
		return sum(sla_statuses) / len(sla_statuses)

# Resource allocation action space
class ResourceAction:
	"""Resource allocation action representation"""
	
	def __init__(self, resource_config: ResourceConfig):
		self.resource_config = resource_config
		
		# Define discrete actions for each resource
		self.gpu_actions = [-1, 0, 1]  # Decrease, no change, increase
		self.cpu_actions = [-1, 0, 1]  # Decrease, no change, increase  
		self.memory_actions = [-1, 0, 1]  # Decrease, no change, increase
		
		# Total number of possible actions
		self.action_dim = len(self.gpu_actions) * len(self.cpu_actions) * len(self.memory_actions)
	
	def action_to_changes(self, action_idx: int) -> Tuple[int, int, int]:
		"""Convert action index to resource changes"""
		# Action index encoding: (3 gpu actions * 3 cpu actions * 3 memory actions = 27 total actions)
		gpu_idx = action_idx // 9  # 0, 1, 2
		cpu_idx = (action_idx % 9) // 3  # 0, 1, 2
		memory_idx = action_idx % 3  # 0, 1, 2
		
		return self.gpu_actions[gpu_idx], self.cpu_actions[cpu_idx], self.memory_actions[memory_idx]
		
		return self.gpu_actions[gpu_idx], self.cpu_actions[cpu_idx], self.memory_actions[memory_idx]
	
	def compute_new_allocation(self, action_idx: int, current_allocation: Dict) -> Dict:
		"""Compute new allocation based on action"""
		gpu_action, cpu_action, memory_action = self.action_to_changes(action_idx)
		
		# Get current allocation
		current_gpu = current_allocation.get('gpu_count', 0)
		current_cpu = current_allocation.get('cpu_count', 0)
		current_memory = current_allocation.get('memory_gb', 0)
		
		# Calculate new allocation with bounds
		new_gpu = max(self.resource_config.gpu_count_min, 
					  min(current_gpu + gpu_action, self.resource_config.gpu_count_max))
		
		new_cpu = max(self.resource_config.cpu_count_min,
					  min(current_cpu + cpu_action * 4,  # Change in steps of 4 cores
						  self.resource_config.cpu_count_max))
		
		new_memory = max(self.resource_config.memory_min,
						min(current_memory + memory_action * 8,  # Change in steps of 8 GB
							self.resource_config.memory_max))
		
		return {
			'gpu_count': new_gpu,
			'cpu_count': new_cpu,
			'memory_gb': new_memory
		}
	
	def get_action_description(self, action_idx: int) -> str:
		"""Get a human-readable description of an action"""
		gpu_action, cpu_action, memory_action = self.action_to_changes(action_idx)
		
		# Create description strings
		gpu_str = "No change to GPUs" if gpu_action == 0 else f"{'Increase' if gpu_action > 0 else 'Decrease'} GPUs by {abs(gpu_action)}"
		cpu_str = "No change to CPUs" if cpu_action == 0 else f"{'Increase' if cpu_action > 0 else 'Decrease'} CPUs by {abs(cpu_action) * 4}"
		memory_str = "No change to memory" if memory_action == 0 else f"{'Increase' if memory_action > 0 else 'Decrease'} memory by {abs(memory_action) * 8} GB"
		
		return f"{gpu_str}, {cpu_str}, {memory_str}"

# CloudEnvironment (MDP formulation)
class CloudEnvironment:
	"""Cloud Environment for reinforcement learning (MDP formulation)"""
	
	def __init__(self, config: Dict):
		self.config = config
		self.resource_config = ResourceConfig(**config.get('resource', {}))
		self.sla_config = SLAConfig(**config.get('sla', {}))
		
		# Initialize state
		self.state = CloudEnvironmentState(config)
		
		# Initialize action space
		self.action_space = ResourceAction(self.resource_config)
		
		# State and action dimensions
		self.state_dim = self.state.state_dim
		self.action_dim = self.action_space.action_dim

		self.epoch_counter = 0
		# self.epoch_adjustment_frequency = 1  # Adjust resources every 10 epochs
		
		# Current allocation
		self.current_allocation = {
			'gpu_count': 1,
			'cpu_count': 4,
			'memory_gb': 8
		}
		
		# Metrics history
		self.metrics_history = []
		
		# Episode stats
		self.episode_reward = 0.0
		self.episode_steps = 0
		self.episode_start_time = time.time()
		
		# Set default user preference
		self.user_preference = 'balanced'

		logger.info(f"Initialized CloudEnvironment with state_dim={self.state_dim}, action_dim={self.action_dim}")
	
	def reset(self) -> np.ndarray:
		"""Reset the environment to initial state"""
		self.state = CloudEnvironmentState(self.config)
		self.metrics_history = []
		self.episode_reward = 0.0
		self.episode_steps = 0
		self.episode_start_time = time.time()
		
		# Reset to minimum allocation with guaranteed minimum values
		self.current_allocation = {
			'gpu_count': max(1, self.resource_config.gpu_count_min),
			'cpu_count': max(1, self.resource_config.cpu_count_min),
			'memory_gb': max(4, self.resource_config.memory_min)
		}
		
		return self.state.get_vector()
	
	
	def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
		"""Take an action in the environment and observe outcome"""
		
		# Increment counter
		self.epoch_counter += 1

		# Compute new allocation from action
		new_allocation = self.action_space.compute_new_allocation(action_idx, self.current_allocation)

		# 🔁 Auto-reduce GPUs with low utilization (<10%), keep at least 1
		gpu_utils = self.state.state_components.get('gpu_util_list', [])
		if len(gpu_utils) > 1:
			if all(util < 5 for util in gpu_utils):
				# All GPUs are idle → reduce to 1
				new_allocation['gpu_count'] = 1
			else:
				# Remove severely underutilized ones
				active_gpus = [util for util in gpu_utils if util >= 10]
				new_allocation['gpu_count'] = max(1, len(active_gpus))

			logger.info(f"[Auto GPU Trim] Adjusted GPU count to {new_allocation['gpu_count']} based on utilization {gpu_utils}")

		# Check if allocation actually changed
		allocation_changed = (
			new_allocation['gpu_count'] != self.current_allocation['gpu_count'] or
			new_allocation['cpu_count'] != self.current_allocation['cpu_count'] or
			new_allocation['memory_gb'] != self.current_allocation['memory_gb']
		)

		# Save previous allocation for reward
		prev_allocation = copy.deepcopy(self.current_allocation)

		# Apply new allocation
		self.current_allocation = new_allocation

		# Update state with latest metrics and allocation
		if self.metrics_history:
			latest_metrics = self.metrics_history[-1].copy()
			latest_metrics.update(self.current_allocation)
			self.state.update_from_metrics(latest_metrics)

		# Calculate reward
		reward = self._calculate_reward(action_idx, prev_allocation, allocation_changed)

		# Update episode stats
		self.episode_reward += reward
		self.episode_steps += 1

		# Check for done
		done = self._check_done()

		# Return current state and metadata
		info = {
			'current_allocation': self.current_allocation,
			'allocation_changed': allocation_changed,
			'action_description': self.action_space.get_action_description(action_idx),
			'episode_reward': self.episode_reward,
			'episode_steps': self.episode_steps,
			'sla_compliance': self.state.get_overall_sla_compliance(),
			'prev_allocation': prev_allocation
		}

		return self.state.get_vector(), reward, done, info

	def update_metrics(self, metrics: Dict) -> np.ndarray:
		"""Update environment with new metrics from training"""
		# Increment epoch counter if this is a new epoch
		if 'epoch' in metrics and isinstance(metrics['epoch'], (int, float)):
			current_epoch = int(metrics['epoch'])
			if len(self.metrics_history) == 0 or current_epoch > self.metrics_history[-1].get('epoch', -1):
				self.epoch_counter += 1

	#   # Add current allocation to metrics
		metrics_with_allocation = metrics.copy()
		metrics_with_allocation.update(self.current_allocation)
		
		# Update state
		self.state.update_from_metrics(metrics_with_allocation)
		
		# Store metrics
		self.metrics_history.append(metrics_with_allocation)
		
		return self.state.get_vector()
	
	# updated reward fucntion to comply with adaptivity
	def _calculate_reward(self, action_idx: int, prev_allocation: Dict, allocation_changed: bool) -> float:
		"""Calculate reward based on action and state with improved reward tracking"""
		# Debug logging to confirm function is being called
		print(f"REWARD CALCULATION CALLED FOR ACTION {action_idx}")
		print(f"Previous allocation: {prev_allocation}")
		print(f"Current allocation: {self.current_allocation}")
		
		# Base reward components
		cost_reward = 0.0
		performance_reward = 0.0
		efficiency_reward = 0.0
		sla_reward = 0.0
		
		# Get current resource allocation (with safe defaults)
		current_gpu_count = self.current_allocation.get('gpu_count', 1)
		current_cpu_count = self.current_allocation.get('cpu_count', 1)
		current_memory_gb = self.current_allocation.get('memory_gb', 4)
		
		# Previous resource allocation (with safe defaults)
		prev_gpu_count = prev_allocation.get('gpu_count', 1)
		prev_cpu_count = prev_allocation.get('cpu_count', 1)
		prev_memory_gb = prev_allocation.get('memory_gb', 4)
		
		# Log the resource values to verify they're being retrieved correctly
		print(f"Resource values - Current: {current_gpu_count} GPUs, {current_cpu_count} CPUs, {current_memory_gb}GB")
		print(f"Resource values - Previous: {prev_gpu_count} GPUs, {prev_cpu_count} CPUs, {prev_memory_gb}GB")
		
		# Calculate cost component
		# Amplify the cost values to make them more significant
		cost_amplifier = 20.0
		
		# Calculate hourly cost for both allocations
		current_hourly_cost = (
			current_gpu_count * self.resource_config.cost_per_gpu_hour +
			current_cpu_count * self.resource_config.cost_per_cpu_hour +
			current_memory_gb * self.resource_config.cost_per_gb_hour
		)

		previous_hourly_cost = (
			prev_gpu_count * self.resource_config.cost_per_gpu_hour +
			prev_cpu_count * self.resource_config.cost_per_cpu_hour +
			prev_memory_gb * self.resource_config.cost_per_gb_hour
		)
		
		# Cost savings (positive means cost decreased)
		cost_savings = previous_hourly_cost - current_hourly_cost
		cost_reward = cost_savings * cost_amplifier
		
		# Calculate performance reward
		performance_amplifier = 10.0
		if len(self.metrics_history) >= 2:
			current_throughput = self.metrics_history[-1].get('throughput', 0)
			previous_throughput = self.metrics_history[-2].get('throughput', 0)
			
			# Calculate throughput change
			throughput_change = current_throughput - previous_throughput
			
			# Normalize by baseline throughput to get percentage change
			if previous_throughput > 0:
				throughput_pct_change = throughput_change / previous_throughput
				performance_reward = throughput_pct_change * performance_amplifier
			else:
				performance_reward = 0.0
		
		# Resource efficiency calculation
		efficiency_amplifier = 10.0
		gpu_util = self.state.state_components.get('gpu_util', 0.1)
		cpu_util = self.state.state_components.get('cpu_util', 0.1)
		
		# Target is 80% utilization - reward approaches closer to target
		gpu_efficiency = (1.0 - abs(gpu_util - 0.8)) * efficiency_amplifier
		cpu_efficiency = (1.0 - abs(cpu_util - 0.8)) * efficiency_amplifier
		
		# Heavy penalty for very low utilization when using multiple GPUs
		if current_gpu_count > 1 and gpu_util < 0.3:
			gpu_efficiency -= 5.0 * (current_gpu_count - 1)
		
		efficiency_reward = (gpu_efficiency + cpu_efficiency) / 2.0
		
		# SLA reward component
		sla_amplifier = 15.0
		sla_status = self.state.get_sla_status()
		
		# Count SLA compliance
		sla_met_count = sum(1 for status in sla_status.values() if status.get('met', False))
		total_slas = len(sla_status)
		
		# Calculate compliance percentage
		sla_compliance = sla_met_count / max(1, total_slas)
		
		# Base reward based on compliance percentage
		sla_reward = (sla_compliance * 2.0 - 0.5) * sla_amplifier
		
		# Apply weights based on user preference
		if self.user_preference == 'cost':
			weights = {'cost': 0.6, 'performance': 0.1, 'efficiency': 0.1, 'sla': 0.2}
		elif self.user_preference == 'time':
			weights = {'cost': 0.1, 'performance': 0.6, 'efficiency': 0.1, 'sla': 0.2}
		elif self.user_preference == 'resource':
			weights = {'cost': 0.1, 'performance': 0.1, 'efficiency': 0.6, 'sla': 0.2}
		else:  # balanced
			weights = {'cost': 0.25, 'performance': 0.25, 'efficiency': 0.25, 'sla': 0.25}
		
		# Small penalty for unnecessary transitions
		transition_penalty = 2.0 if allocation_changed else 0.0
		
		# Calculate final reward
		total_reward = (
			weights['cost'] * cost_reward +
			weights['performance'] * performance_reward +
			weights['efficiency'] * efficiency_reward +
			weights['sla'] * sla_reward -
			transition_penalty
		)
		
		# Ensure we never return exactly zero reward
		if abs(total_reward) < 0.01:
			total_reward = 0.5  # Small positive value
		
		# Force a minimum reward value for debugging
		# Remove this line once you confirm the function is working
		# debugging_reward = 10.0
		# total_reward = debugging_reward
		
		# Store components for logging
		self.last_reward_components = {
			'cost_reward': cost_reward,
			'performance_reward': performance_reward,
			'efficiency_reward': efficiency_reward,
			'sla_reward': sla_reward,
			'total_reward': total_reward
		}
		
		# Debug logging of the final reward
		print(f"REWARD COMPONENTS: cost={cost_reward:.2f}, perf={performance_reward:.2f}, "
			  f"efficiency={efficiency_reward:.2f}, sla={sla_reward:.2f}")
		print(f"RETURNING REWARD: {total_reward:.2f}")
		
		return total_reward

	
	def _check_done(self) -> bool:
		"""Check if episode should terminate"""
		# Episode is done if training is complete or SLAs are severely violated
		
		# Check if training is complete (current_epoch >= total_epochs)
		if (self.state.state_components['current_epoch'] >= self.state.state_components['total_epochs'] and
			self.state.state_components['total_epochs'] > 0):
			return True
		
		# Check for severe SLA violations
		sla_status = self.state.get_sla_status()
		severe_violations = sum(1 for status in sla_status.values() 
							   if not status['met'] and status['severity'] > 0.5)
		
		if severe_violations >= 3:
			logger.warning("Episode terminated due to multiple severe SLA violations")
			return True
		
		# Episode time limit
		episode_time = time.time() - self.episode_start_time
		if episode_time > 3600:  # 1 hour limit
			logger.warning("Episode terminated due to time limit")
			return True
		
		return False
	
	def estimate_optimal_resources(self) -> List[Dict]:
		"""Estimate optimal resource allocations for different priorities"""
		if not self.metrics_history:
			return []
		
		# Get latest metrics
		latest_metrics = self.metrics_history[-1]
		
		# Variables to consider
		model_params = latest_metrics.get('model_params', 0)
		data_size = latest_metrics.get('data_size_mb', 0)
		throughput = latest_metrics.get('throughput', 0)
		current_gpu_count = self.current_allocation['gpu_count']
		current_cpu_count = self.current_allocation['cpu_count']
		current_memory_gb = self.current_allocation['memory_gb']
		
		# GPU utilization
		gpu_utilizations = {}
		for key, value in latest_metrics.items():
			if key.startswith('gpu_utilization_') and not pd.isna(value):
				gpu_utilizations[key] = value
		
		avg_gpu_util = sum(gpu_utilizations.values()) / max(1, len(gpu_utilizations))
		
		# Calculate options for different priorities
		options = []
		
		# 1. Time-critical option: maximize performance
		if avg_gpu_util > 70 and current_gpu_count < self.resource_config.gpu_count_max:
			# High utilization suggests adding GPUs would help
			time_gpu_count = min(current_gpu_count + 2, self.resource_config.gpu_count_max)
			time_cpu_count = min(current_cpu_count + 8, self.resource_config.cpu_count_max)
			time_memory_gb = min(current_memory_gb + 16, self.resource_config.memory_max)
		else:
			# GPUs not fully utilized, focus on optimizing current resources
			time_gpu_count = current_gpu_count
			time_cpu_count = min(current_cpu_count + 4, self.resource_config.cpu_count_max)
			time_memory_gb = current_memory_gb
		
		time_option = {
			'priority': 'time',
			'name': 'Time-Critical',
			'gpu_count': time_gpu_count,
			'cpu_count': time_cpu_count,
			'memory_gb': time_memory_gb,
			'estimated_speedup': time_gpu_count / max(1, current_gpu_count) * 0.8,
			'description': 'Maximize training speed at higher cost'
		}
		options.append(time_option)
		
		# 2. Cost-critical option: minimize cost while maintaining performance
		# Modified for very low GPU utilization
		if avg_gpu_util < 5.0 and current_gpu_count > 1:
			# Very low utilization - drastically reduce GPUs
			cost_gpu_count = 1  # Reduce to single GPU
			# For extreme cases, consider CPU-only
			if avg_gpu_util < 1.0 and self.resource_config.cpu_count_max >= 8:
				cost_gpu_count = 0  # CPU-only option
		elif avg_gpu_util < 50 and current_gpu_count > 1:
			# Moderate low utilization - reduce by one
			cost_gpu_count = current_gpu_count - 1
		else:
			cost_gpu_count = current_gpu_count
	
		if throughput > self.sla_config.throughput_min * 1.5:
			# High throughput suggests we can reduce CPUs
			cost_cpu_count = max(current_cpu_count - 4, self.resource_config.cpu_count_min)
			cost_memory_gb = max(current_memory_gb - 8, self.resource_config.memory_min)
		else:
			cost_cpu_count = current_cpu_count
			cost_memory_gb = current_memory_gb
		
		cost_option = {
			'priority': 'cost',
			'name': 'Cost-Critical',
			'gpu_count': cost_gpu_count,
			'cpu_count': cost_cpu_count,
			'memory_gb': cost_memory_gb,
			'estimated_savings': 1.0 - (
				(cost_gpu_count * self.resource_config.cost_per_gpu_hour +
				 cost_cpu_count * self.resource_config.cost_per_cpu_hour +
				 cost_memory_gb * self.resource_config.cost_per_gb_hour) /
				(current_gpu_count * self.resource_config.cost_per_gpu_hour +
				 current_cpu_count * self.resource_config.cost_per_cpu_hour +
				 current_memory_gb * self.resource_config.cost_per_gb_hour)
			),
			'description': 'Minimize cost while maintaining acceptable performance'
		}
		options.append(cost_option)
		
		# Modified for very low GPU utilization
		if avg_gpu_util < 5.0 and current_gpu_count > 1:
			# Extremely low utilization - reduce to minimum
			resource_gpu_count = 1
			# For near-zero utilization, consider removing GPUs entirely
			if avg_gpu_util < 1.0:
				resource_gpu_count = 0
		elif avg_gpu_util < 40 and current_gpu_count > 1:
			# Low utilization - reduce by one
			resource_gpu_count = current_gpu_count - 1

		elif avg_gpu_util > 90:
			# Very high utilization suggests adding a GPU to avoid bottlenecks
			resource_gpu_count = min(current_gpu_count + 1, self.resource_config.gpu_count_max)
		else:
			resource_gpu_count = current_gpu_count
		
		# Adjust CPU and memory to match GPU needs
		resource_cpu_count = max(resource_gpu_count * 4, self.resource_config.cpu_count_min)
		resource_memory_gb = max(resource_gpu_count * 8, self.resource_config.memory_min)
		
		resource_option = {
			'priority': 'resource',
			'name': 'Resource-Efficient',
			'gpu_count': resource_gpu_count,
			'cpu_count': resource_cpu_count,
			'memory_gb': resource_memory_gb,
			'estimated_utilization_improvement': (90 - avg_gpu_util) / 100.0 if avg_gpu_util < 90 else 0,
			'description': 'Optimize resource utilization and efficiency'
		}
		options.append(resource_option)
		
		# 4. Balanced option (current allocation if optimized, or slightly adjusted)
		balanced_option = {
			'priority': 'balanced',
			'name': 'Balanced',
			'gpu_count': current_gpu_count,
			'cpu_count': current_cpu_count,
			'memory_gb': current_memory_gb,
			'description': 'Current allocation (balanced between cost and performance)'
		}
		options.append(balanced_option)
		
		# Calculate expected cost and performance for each option
		hourly_costs = {}
		for option in options:
			hourly_cost = (
				option['gpu_count'] * self.resource_config.cost_per_gpu_hour +
				option['cpu_count'] * self.resource_config.cost_per_cpu_hour +
				option['memory_gb'] * self.resource_config.cost_per_gb_hour
			)
			option['hourly_cost'] = hourly_cost
			hourly_costs[option['priority']] = hourly_cost
		
		# Return all options, sorted by priority
		return options

	def log_adjustment_details(self, action_idx: int, prev_allocation: Dict, new_allocation: Dict, reward: float):
		"""Log detailed information about a resource adjustment"""
		# Get current SLA status
		sla_status = self.state.get_sla_status()

		# Format SLA information
		sla_info = []
		for sla_name, status in sla_status.items():
			status_str = "✓ Met" if status['met'] else f"✗ Violated (severity: {status['severity']:.2f})"
			sla_info.append(f"{sla_name}: {status_str}")

		# Get action description
		action_desc = self.action_space.get_action_description(action_idx)

		# Safe key access for allocations
		gpu_old = prev_allocation.get('gpu_count', 0)
		gpu_new = new_allocation.get('gpu_count', 0)
		cpu_old = prev_allocation.get('cpu_count', 0)
		cpu_new = new_allocation.get('cpu_count', 0)
		mem_old = prev_allocation.get('memory_gb', 0.0)
		mem_new = new_allocation.get('memory_gb', 0.0)

		# Calculate deltas
		gpu_change = gpu_new - gpu_old
		cpu_change = cpu_new - cpu_old
		mem_change = mem_new - mem_old

		# Format allocation changes
		allocation_changes = []
		if gpu_change != 0:
			allocation_changes.append(f"GPUs: {gpu_old} → {gpu_new} ({gpu_change:+d})")
		if cpu_change != 0:
			allocation_changes.append(f"CPUs: {cpu_old} → {cpu_new} ({cpu_change:+d})")
		if mem_change != 0:
			allocation_changes.append(f"Memory: {mem_old:.1f}GB → {mem_new:.1f}GB ({mem_change:+.1f}GB)")

		# Get reward breakdown
		reward_components = getattr(self, "last_reward_components", {})

		# Log everything
		logger.info(f"\n=== Resource Adjustment at Epoch {self.epoch_counter} ===")
		logger.info(f"Action: {action_desc}")
		logger.info(f"Changes: {', '.join(allocation_changes) if allocation_changes else 'No changes'}")
		logger.info(f"Reward: {reward:.4f}")
		logger.info(f"Reward components: {reward_components}")
		logger.info(f"SLA Status:\n  " + "\n  ".join(sla_info))
		logger.info(f"Current allocation: GPUs={gpu_new}, CPUs={cpu_new}, Memory={mem_new:.1f}GB")
		logger.info("=====================================\n")

		# Return all details
		return {
			'epoch': self.epoch_counter,
			'action': action_idx,
			'action_description': action_desc,
			'reward': reward,
			'gpu_count': gpu_new,
			'cpu_count': cpu_new,
			'memory_gb': mem_new,
			'gpu_change': gpu_change,
			'cpu_change': cpu_change,
			'memory_change': mem_change,
			'reward_cost': reward_components.get('cost_reward', 0),
			'reward_performance': reward_components.get('performance_reward', 0),
			'reward_efficiency': reward_components.get('efficiency_reward', 0),
			'reward_sla': reward_components.get('sla_reward', 0),
			'sla_time_met': int(sla_status['time']['met']),
			'sla_cost_met': int(sla_status['cost']['met']),
			'sla_throughput_met': int(sla_status['throughput']['met']),
			'sla_memory_met': int(sla_status['memory']['met']),
			'sla_gpu_util_met': int(sla_status['gpu_util']['met']),
			'sla_cpu_util_met': int(sla_status['cpu_util']['met']),
			'sla_time_severity': sla_status['time']['severity'],
			'sla_cost_severity': sla_status['cost']['severity'],
			'sla_throughput_severity': sla_status['throughput']['severity'],
			'sla_memory_severity': sla_status['memory']['severity'],
			'sla_gpu_util_severity': sla_status['gpu_util']['severity'],
			'sla_cpu_util_severity': sla_status['cpu_util']['severity'],
		}

	# Method to get reward components
	def _get_reward_components(self, prev_allocation: Dict) -> Dict:
		"""Get the components of the reward for the last action"""
		if not hasattr(self, 'last_reward_components'):
			# Calculate if not already done
			self._calculate_reward(0, prev_allocation, False)
		return self.last_reward_components

# ==================== Neural Network Models with dropout to handle uncertainty ====================

class Actor(nn.Module):
	"""Actor network for policy approximation with dropout for uncertainty estimation"""
	
	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.1):
		super(Actor, self).__init__()
		
		self.net = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(p=dropout_rate),  # Add dropout after first hidden layer
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(p=dropout_rate),  # Add dropout after second hidden layer
			nn.Linear(hidden_dim, action_dim)
		)
		
		# Store dropout rate for reference
		self.dropout_rate = dropout_rate
		# Flag to enable/disable dropout during inference
		self.enable_dropout = False
		
	def forward(self, state: torch.Tensor) -> torch.Tensor:
		# Set dropout mode based on flag
		if self.enable_dropout:
			self.train()  # Enable dropout
		else:
			self.eval()   # Disable dropout (normal behavior)
			
		return F.softmax(self.net(state), dim=-1)

class Critic(nn.Module):
	"""Critic network for value function approximation with dropout for uncertainty estimation"""
	
	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.1):
		super(Critic, self).__init__()
		
		# Q-value architecture with dropout
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.drop1 = nn.Dropout(p=dropout_rate)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.drop2 = nn.Dropout(p=dropout_rate)
		self.l3 = nn.Linear(hidden_dim, 1)
		
		# Store dropout rate for reference
		self.dropout_rate = dropout_rate
		# Flag to enable/disable dropout during inference
		self.enable_dropout = False
		
	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		# Set dropout mode based on flag
		if self.enable_dropout:
			self.train()  # Enable dropout
		else:
			self.eval()   # Disable dropout (normal behavior)
			
		# Forward pass with explicit dropout layers
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		q1 = self.drop1(q1)
		q1 = F.relu(self.l2(q1))
		q1 = self.drop2(q1)
		q1 = self.l3(q1)
		
		return q1

class SimplePolicy:
	"""A simple policy gradient agent for resource optimization"""

	def __init__(self, config, state_dim, action_dim):
		self.config = config
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Extract RL configuration
		self.rl_config = RLConfig(**config.get('rl', {}))

		# Initialize policy network
		self.policy_network = nn.Sequential(
			nn.Linear(state_dim, self.rl_config.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.rl_config.hidden_dim, self.rl_config.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.rl_config.hidden_dim, action_dim)
		).to(self.device)
		
		# Initialize optimizer
		self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.rl_config.lr_actor)
		
		# Experience buffer
		self.states = []
		self.actions = []
		self.rewards = []
		self.log_probs = []
		
		# Training stats
		self.train_steps = 0
		
		logger.info(f"Initialized Simple Policy Gradient agent")

	def select_action(self, state, explore=True):
		"""Select action using the policy network"""
		# state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		state_tensor.requires_grad_()  # 🔥 ensures backprop tracks this

		
		with torch.no_grad():
			logits = self.policy_network(state_tensor)
			action_probs = F.softmax(logits, dim=-1)
			
			if explore:
				# Add exploration noise
				noise = torch.FloatTensor(action_probs.shape).data.normal_(
					0, self.rl_config.exploration_noise).to(self.device)
				action_probs = F.softmax(action_probs + noise, dim=-1)
			
			# Sample action from distribution or take greedy action
			if explore and random.random() < 0.2:  # Simple exploration strategy
				action_idx = random.randint(0, self.action_dim - 1)
			else:
				action_idx = torch.argmax(action_probs).item()
			
			# Create one-hot action for later
			action_one_hot = torch.zeros(self.action_dim)
			action_one_hot[action_idx] = 1.0
			
			return action_idx, action_one_hot, 0.0  # Return 0 for uncertainty (compatibility)

	def add_experience(self, state, action, reward, next_state, done):
		"""Add experience to buffer for batch update"""
		self.states.append(torch.FloatTensor(state).to(self.device))
		self.actions.append(action.to(self.device))
		self.rewards.append(reward)
		
		# Compute log probability of the action
		with torch.no_grad():
			logits = self.policy_network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
			action_probs = F.softmax(logits, dim=-1)
			log_prob = torch.log(action_probs.squeeze(0)[torch.argmax(action)])
			self.log_probs.append(log_prob)

	def train(self):
		"""Train the policy network using collected experiences"""
		self.train_steps += 1
		
		# Only train every few steps and if enough experience is collected
		if self.train_steps % self.rl_config.train_freq != 0 or len(self.states) < self.rl_config.batch_size:
			return
		
		# Calculate returns (discounted rewards)
		returns = []
		R = 0
		for r in reversed(self.rewards):
			R = r + self.rl_config.gamma * R
			returns.insert(0, R)
			
		# Convert returns to tensor and normalize
		returns = torch.FloatTensor(returns).to(self.device)
		returns = (returns - returns.mean()) / (returns.std() + 1e-5)
		
		# Convert log probs to tensor
		# log_probs = torch.stack(self.log_probs)
		log_probs = [lp for lp in self.log_probs if lp.requires_grad]
		if not log_probs:
			logger.warning("No valid log_probs with gradients found — skipping training step.")
			return
		log_probs = torch.stack(log_probs)

		
		# Calculate policy loss
		policy_loss = -(log_probs * returns).mean()
		
		# Update policy network
		self.optimizer.zero_grad()
		policy_loss.backward()
		self.optimizer.step()
		
		# Clear experience buffer
		self.states = []
		self.actions = []
		self.rewards = []
		self.log_probs = []
		
	# For compatibility with old code
	def check_adaptive_reset(self, sla_status):
		"""Placeholder for compatibility with EnsembleActorCritic"""
		pass
		
	# Added for compatibility
	reset_counters = [0]

	def save(self, path='saved_policy'):
		"""Save the policy network"""
		os.makedirs(path, exist_ok=True)
		torch.save({
			'policy_network': self.policy_network.state_dict(),
			'optimizer': self.optimizer.state_dict(),
		}, f"{path}/policy.pt")
		logger.info(f"Saved policy to {path}")

	def load(self, path='saved_policy'):
		"""Load the policy network"""
		try:
			checkpoint = torch.load(f"{path}/policy.pt")
			self.policy_network.load_state_dict(checkpoint['policy_network'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			logger.info(f"Loaded policy from {path}")
		except Exception as e:
			logger.error(f"Error loading policy: {str(e)}")

	def reset(self):
		"""Reset the agent's state for a new episode"""
		# Reset training stats
		self.train_steps = 0
		
		# Clear experience buffers
		self.states = []
		self.actions = []
		self.rewards = []
		self.log_probs = []
		
		print("SimplePolicy agent reset")

		# def reset(self):
		# pass  # Optional: no-op for now


class MultiObjectiveRLAgent:
	"""Multi-objective RL agent with multiple policies for different objectives"""
	
	def __init__(self, config, state_dim, action_dim):
		self.config = config
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Extract configuration
		self.rl_config = RLConfig(**config.get('rl', {}))
		self.multi_objective_config = MultiObjectiveConfig(**config.get('multi_objective', {}))
		
		# Initialize multiple policies
		self.actors = []
		self.critics = []
		self.target_actors = []
		self.target_critics = []
		self.actor_optimizers = []
		self.critic_optimizers = []
		
		# Create policies for each preference vector
		for i in range(self.multi_objective_config.num_policies):
			# Actor and critic
			actor = Actor(state_dim, action_dim, self.rl_config.hidden_dim).to(self.device)
			critic = Critic(state_dim, action_dim, self.rl_config.hidden_dim).to(self.device)
			
			# Target networks
			target_actor = Actor(state_dim, action_dim, self.rl_config.hidden_dim).to(self.device)
			target_critic = Critic(state_dim, action_dim, self.rl_config.hidden_dim).to(self.device)
			
			# Initialize targets with same weights
			target_actor.load_state_dict(actor.state_dict())
			target_critic.load_state_dict(critic.state_dict())
			
			# Optimizers
			actor_optimizer = optim.Adam(actor.parameters(), lr=self.rl_config.lr_actor)
			critic_optimizer = optim.Adam(critic.parameters(), lr=self.rl_config.lr_critic)
			
			# Add to lists
			self.actors.append(actor)
			self.critics.append(critic)
			self.target_actors.append(target_actor)
			self.target_critics.append(target_critic)
			self.actor_optimizers.append(actor_optimizer)
			self.critic_optimizers.append(critic_optimizer)
		
		# Replay buffers for each policy
		self.replay_buffers = [
			ReplayBuffer(state_dim, action_dim, self.rl_config.buffer_size)
			for _ in range(self.multi_objective_config.num_policies)
		]
		
		# Training stats
		self.train_steps = 0
		self.reset_counters = [0] * self.multi_objective_config.num_policies
		
		logger.info(f"Initialized MultiObjectiveRLAgent with {self.multi_objective_config.num_policies} policies")
	
	def select_policy(self, sla_status, user_preference):
		"""Select appropriate policy based on SLA status and user preference"""
		# Count SLA violations
		violations = sum(1 for status in sla_status.values() if not status['met'])
		
		if violations > 2:
			# Multiple SLA violations - use balanced policy (index 0)
			return 0
		
		# Map user preference to policy index
		preference_map = {
			'balanced': 0,
			'cost': 1,
			'time': 2,
			'resource': 3
		}
		
		policy_idx = preference_map.get(user_preference, 0)
		return policy_idx
	
	def select_action(self, state, policy_idx=0, explore=True):
		"""Select action using the specified policy"""
		# Convert state to tensor
		state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		
		with torch.no_grad():
			# Get action probabilities
			action_probs = self.actors[policy_idx](state_tensor)
			
			# For exploration, add some noise
			if explore:
				noise = torch.FloatTensor(action_probs.shape).data.normal_(
					0, self.rl_config.exploration_noise).to(self.device)
				action_probs = action_probs + noise
				action_probs = F.softmax(action_probs, dim=-1)
			
			# Get action index with highest probability
			action_idx = torch.argmax(action_probs).item()
			
			# Convert to one-hot for critic
			action_one_hot = torch.zeros(self.action_dim)
			action_one_hot[action_idx] = 1.0
			
			return action_idx, action_one_hot
	
	def add_experience(self, state, action, reward, next_state, done, policy_idx=0):
		"""Add experience to replay buffer"""
		self.replay_buffers[policy_idx].add(state, action, reward, next_state, done)
	
	def train(self, policy_idx=0):
		"""Train the specified policy"""
		buffer = self.replay_buffers[policy_idx]
		
		if len(buffer) < self.rl_config.batch_size:
			return
		
		self.train_steps += 1
		
		# Only train every few steps
		if self.train_steps % self.rl_config.train_freq != 0:
			return
		
		# Sample batch
		state, action, reward, next_state, done = buffer.sample(self.rl_config.batch_size)
		
		# Train critic
		with torch.no_grad():
			# Target actions
			next_action_probs = self.target_actors[policy_idx](next_state)
			
			# Target Q-values
			target_q = self.target_critics[policy_idx](next_state, next_action_probs)
			target_q = reward + (1 - done) * self.rl_config.gamma * target_q
		
		# Current Q-values
		current_q = self.critics[policy_idx](state, action)
		
		# Critic loss
		critic_loss = F.mse_loss(current_q, target_q)
		
		# Optimize critic
		self.critic_optimizers[policy_idx].zero_grad()
		critic_loss.backward()
		self.critic_optimizers[policy_idx].step()
		
		# Train actor
		action_probs = self.actors[policy_idx](state)
		actor_loss = -self.critics[policy_idx](state, action_probs).mean()
		
		# Optimize actor
		self.actor_optimizers[policy_idx].zero_grad()
		actor_loss.backward()
		self.actor_optimizers[policy_idx].step()
		
		# Update target networks
		for param, target_param in zip(self.critics[policy_idx].parameters(), 
									  self.target_critics[policy_idx].parameters()):
			target_param.data.copy_(
				self.rl_config.tau * param.data + (1 - self.rl_config.tau) * target_param.data)
		
		for param, target_param in zip(self.actors[policy_idx].parameters(), 
									  self.target_actors[policy_idx].parameters()):
			target_param.data.copy_(
				self.rl_config.tau * param.data + (1 - self.rl_config.tau) * target_param.data)
	
	def check_adaptive_reset(self, sla_status):
		"""Check if critic should be reset adaptively based on SLA status"""
		if self.train_steps % self.rl_config.reset_interval == 0:
			# Count severe violations
			severe_violations = sum(1 for status in sla_status.values() 
								  if not status['met'] and status['severity'] > 0.3)
			
			# Increase reset probability with more violations
			reset_prob = self.rl_config.reset_probability * (1 + severe_violations * 0.2)
			
			for i in range(self.multi_objective_config.num_policies):
				if random.random() < reset_prob:
					# Reset critic for this policy
					logger.info(f"Adaptively resetting critic for policy {i}")
					self.reset_counters[i] += 1
					
					# Initialize new critic
					new_critic = Critic(self.state_dim, self.action_dim, 
									   self.rl_config.hidden_dim).to(self.device)
					new_target = Critic(self.state_dim, self.action_dim, 
									  self.rl_config.hidden_dim).to(self.device)
					
					# Copy weights from current actor to make initial Q-values reasonable
					self.critics[i] = new_critic
					self.target_critics[i] = new_target
					self.target_critics[i].load_state_dict(self.critics[i].state_dict())
					
					# New optimizer
					self.critic_optimizers[i] = optim.Adam(
						self.critics[i].parameters(), lr=self.rl_config.lr_critic)
	
	def save(self, path='saved_agent'):
		"""Save all policies"""
		os.makedirs(path, exist_ok=True)
		
		for i in range(self.multi_objective_config.num_policies):
			torch.save({
				'actor': self.actors[i].state_dict(),
				'critic': self.critics[i].state_dict(),
				'target_actor': self.target_actors[i].state_dict(),
				'target_critic': self.target_critics[i].state_dict(),
				'actor_optimizer': self.actor_optimizers[i].state_dict(),
				'critic_optimizer': self.critic_optimizers[i].state_dict(),
			}, f"{path}/policy_{i}.pt")
		
		logger.info(f"Saved agent to {path}")
	
	def load(self, path='saved_agent'):
		"""Load all policies"""
		for i in range(self.multi_objective_config.num_policies):
			try:
				checkpoint = torch.load(f"{path}/policy_{i}.pt")
				
				self.actors[i].load_state_dict(checkpoint['actor'])
				self.critics[i].load_state_dict(checkpoint['critic'])
				self.target_actors[i].load_state_dict(checkpoint['target_actor'])
				self.target_critics[i].load_state_dict(checkpoint['target_critic'])
				self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizer'])
				self.critic_optimizers[i].load_state_dict(checkpoint['critic_optimizer'])
			except Exception as e:
				logger.error(f"Error loading policy {i}: {str(e)}")

class ReplayBuffer:
	"""Experience replay buffer for RL agent"""
	
	def __init__(self, state_dim: int, action_dim: int, max_size: int = 100000):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.done = np.zeros((max_size, 1))
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
			next_state: np.ndarray, done: bool):
		"""Add a transition to the buffer"""
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.done[self.ptr] = done
		
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
	
	def sample(self, batch_size: int) -> Tuple:
		"""Sample a batch of transitions"""
		ind = np.random.randint(0, self.size, size=batch_size)
		
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)
	
	def __len__(self) -> int:
		return self.size

class RLMetricsTracker:
	"""Tracks and stores metrics for RL optimization for later plotting"""
	
	def __init__(self, output_dir="./rl_metrics"):
		self.output_dir = output_dir
		os.makedirs(output_dir, exist_ok=True)
		
		# Initialize DataFrames for different metrics
		self.allocation_history = pd.DataFrame(columns=[
			'epoch', 'gpu_count', 'cpu_count', 'memory_gb', 
			'action', 'action_description', 'reward'
		])
		
		self.reward_components = pd.DataFrame(columns=[
			'epoch', 'reward', 'reward_cost', 'reward_performance', 
			'reward_efficiency', 'reward_sla'
		])
		
		self.sla_metrics = pd.DataFrame(columns=[
			'epoch', 'sla_time_met', 'sla_cost_met', 'sla_throughput_met',
			'sla_memory_met', 'sla_gpu_util_met', 'sla_cpu_util_met',
			'sla_time_severity', 'sla_cost_severity', 'sla_throughput_severity',
			'sla_memory_severity', 'sla_gpu_util_severity', 'sla_cpu_util_severity'
		])
		
		self.pareto_front = pd.DataFrame(columns=[
			'epoch', 'preference', 'estimated_cost', 'estimated_time', 
			'gpu_count', 'cpu_count', 'memory_gb'
		])
		
		self.performance_metrics = pd.DataFrame(columns=[
			'epoch', 'throughput', 'time_so_far', 'cost_so_far',
			'gpu_util', 'cpu_util', 'memory_util'
		])
	
	def add_adjustment_metrics(self, metrics):
		"""Add metrics from a resource adjustment"""
		# Extract metrics for each DataFrame
		
		# Allocation history
		allocation = {
			'epoch': metrics['epoch'],
			'gpu_count': metrics['gpu_count'],
			'cpu_count': metrics['cpu_count'],
			'memory_gb': metrics['memory_gb'],
			'action': metrics['action'],
			'action_description': metrics['action_description'],
			'reward': metrics['reward']
		}
		self.allocation_history = pd.concat([
			self.allocation_history, 
			pd.DataFrame([allocation])
		], ignore_index=True)
		
		# Reward components
		rewards = {
			'epoch': metrics['epoch'],
			'reward': metrics['reward'],
			'reward_cost': metrics['reward_cost'],
			'reward_performance': metrics['reward_performance'],
			'reward_efficiency': metrics['reward_efficiency'],
			'reward_sla': metrics['reward_sla']
		}
		self.reward_components = pd.concat([
			self.reward_components, 
			pd.DataFrame([rewards])
		], ignore_index=True)
		
		# SLA metrics
		sla = {k: metrics[k] for k in self.sla_metrics.columns if k in metrics}
		if 'epoch' in sla:
			self.sla_metrics = pd.concat([
				self.sla_metrics, 
				pd.DataFrame([sla])
			], ignore_index=True)
	
	def add_performance_metrics(self, metrics):
		"""Add general training performance metrics"""
		perf = {k: metrics[k] for k in self.performance_metrics.columns if k in metrics}
		if 'epoch' in perf:
			self.performance_metrics = pd.concat([
				self.performance_metrics, 
				pd.DataFrame([perf])
			], ignore_index=True)
	
	def add_pareto_front(self, epoch, pareto_options):
		"""Add current Pareto front of optimization options"""
		for option in pareto_options:
			option_data = {
				'epoch': epoch,
				'preference': option.get('priority', 'unknown'),
				'estimated_cost': option.get('estimated_cost', 0),
				'estimated_time': option.get('estimated_time', 0),
				'gpu_count': option.get('gpu_count', 0),
				'cpu_count': option.get('cpu_count', 0),
				'memory_gb': option.get('memory_gb', 0)
			}
			self.pareto_front = pd.concat([
				self.pareto_front, 
				pd.DataFrame([option_data])
			], ignore_index=True)
	
	def save_all_metrics(self, run_timestamp=None, use_timestamp=True):
		"""Save all metrics to CSV files, optionally without timestamps"""
		# Generate timestamp if needed and not provided
		if use_timestamp and run_timestamp is None:
			run_timestamp = int(time.time())
		
		# Determine file suffix based on whether to use timestamp
		file_suffix = f"_{run_timestamp}" if use_timestamp else ""
		
		# Define output paths
		files = {}
		
		# Save each DataFrame
		file_path = f"{self.output_dir}/allocation_history{file_suffix}.csv"
		self.allocation_history.to_csv(file_path, index=False)
		files['allocation_history'] = file_path
		
		file_path = f"{self.output_dir}/reward_components{file_suffix}.csv"
		self.reward_components.to_csv(file_path, index=False)
		files['reward_components'] = file_path
		
		file_path = f"{self.output_dir}/sla_metrics{file_suffix}.csv"
		self.sla_metrics.to_csv(file_path, index=False)
		files['sla_metrics'] = file_path
		
		file_path = f"{self.output_dir}/pareto_front{file_suffix}.csv"
		self.pareto_front.to_csv(file_path, index=False)
		files['pareto_front'] = file_path
		
		file_path = f"{self.output_dir}/performance_metrics{file_suffix}.csv"
		self.performance_metrics.to_csv(file_path, index=False)
		files['performance_metrics'] = file_path
		
		# Create combined metrics file with essential columns
		combined = pd.merge(
			self.allocation_history[['epoch', 'gpu_count', 'cpu_count', 'memory_gb', 'reward']],
			self.reward_components[['epoch', 'reward_cost', 'reward_performance', 'reward_efficiency', 'reward_sla']],
			on='epoch', how='outer'
		)
		
		# Add SLA metrics
		if not self.sla_metrics.empty:
			combined = pd.merge(
				combined,
				self.sla_metrics[['epoch', 'sla_time_met', 'sla_cost_met', 'sla_throughput_met']],
				on='epoch', how='outer'
			)
		
		# Add performance metrics
		if not self.performance_metrics.empty:
			combined = pd.merge(
				combined,
				self.performance_metrics[['epoch', 'throughput', 'time_so_far', 'cost_so_far']],
				on='epoch', how='outer'
			)
		
		file_path = f"{self.output_dir}/combined_metrics{file_suffix}.csv"
		combined.to_csv(file_path, index=False)
		files['combined_metrics'] = file_path
		
		# Add RL metrics if available
		if hasattr(self, 'rl_metrics') and not self.rl_metrics.empty:
			file_path = f"{self.output_dir}/rl_action_log{file_suffix}.csv"
			self.rl_metrics.to_csv(file_path, index=False)
			files['rl_action_log'] = file_path
			logger.info(f"Saved RL action log to: {file_path}")
		
		return files
	

class CloudResourceOptimizer:
	"""Main class for cloud resource optimization using RL"""
	
	def __init__(self, config_path: str = None, config_dict: Dict = None, offline_data: OfflineDataAnalyzer = None):
		# Load configuration
		if config_path and os.path.exists(config_path):
			with open(config_path, 'r') as f:
				self.config = yaml.safe_load(f)
		elif config_dict:
			self.config = config_dict
		else:
			# Default config
			self.config = {
				'resource': ResourceConfig().__dict__,
				'sla': SLAConfig().__dict__,
				'rl': RLConfig().__dict__,
				'multi_objective': MultiObjectiveConfig().__dict__
			}

		# Offline data analyzer
		self.offline_analyzer = offline_data or OfflineDataAnalyzer()
		
		# Training module manager
		self.training_manager = TrainingModuleManager()
		self.training_manager.resource_config = ResourceConfig(**self.config.get('resource', {}))

		# Environment and agent will be initialized later when we know the state dimension
		self.cloud_env = None
		self.agent = None
		
		# Current preference
		self.user_preference = 'balanced'
		
		# Metrics collection
		self.rl_metrics = pd.DataFrame(columns=[
			'epoch', 'rl_policy_used', 'rl_action_taken', 'rl_action_description',
			'rl_reward', 'rl_gpu_count', 'rl_cpu_count', 'rl_memory_gb',
			'rl_sla_time_met', 'rl_sla_cost_met', 'rl_sla_throughput_met',
			'rl_sla_memory_met', 'rl_sla_gpu_util_met', 'rl_sla_cpu_util_met',
			'rl_critic_reset', 'rl_cumulative_reward'
		])

		self.metrics_tracker = RLMetricsTracker()  # Initialize the metrics tracker
		
		logger.info("Initialized CloudResourceOptimizer")
	
	def save_all_metrics(self, use_timestamp=False):
		"""Save all metrics collected during training"""
		try:
			# First, add the RL metrics to the metrics tracker
			if len(self.rl_metrics) > 0:
				# If metrics tracker has rl_metrics attribute, update it
				if hasattr(self.metrics_tracker, 'rl_metrics'):
					self.metrics_tracker.rl_metrics = self.rl_metrics
				# Otherwise, set it
				else:
					setattr(self.metrics_tracker, 'rl_metrics', self.rl_metrics)
				
				# Save all metrics through the metrics tracker
				files = self.metrics_tracker.save_all_metrics(use_timestamp=use_timestamp)
				logger.info(f"All metrics saved: {list(files.keys())}")
				return files
			else:
				# Save metrics tracker data even if no RL metrics
				if hasattr(self, 'metrics_tracker'):
					files = self.metrics_tracker.save_all_metrics(use_timestamp=use_timestamp)
					logger.info(f"Metrics tracker data saved: {list(files.keys())}")
					return files
				else:
					logger.warning("No RL metrics to save and no metrics tracker available.")
					return {}
		except Exception as e:
			logger.error(f"Error saving RL metrics: {e}")
			return {}

	def validate_gpu_configuration(self):
		"""Validate GPU configuration and detect potential issues"""
		import torch
		import os
		
		# Check CUDA availability
		cuda_available = torch.cuda.is_available()
		visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
		actual_visible_count = len(visible_devices.split(',')) if visible_devices else 0
		
		# Try to count devices
		try:
			cuda_device_count = torch.cuda.device_count()
		except:
			cuda_device_count = 0
			
		# Log findings
		logger.info(f"GPU Validation: CUDA Available: {cuda_available}")
		logger.info(f"GPU Validation: CUDA_VISIBLE_DEVICES: '{visible_devices}'")
		logger.info(f"GPU Validation: Visible Device Count: {actual_visible_count}")
		logger.info(f"GPU Validation: CUDA Device Count: {cuda_device_count}")
		
		# Detect issues
		if cuda_available and cuda_device_count == 0:
			logger.warning("GPU Validation: CUDA is available but no devices detected - possible misconfiguration")
		
		if visible_devices and cuda_device_count == 0:
			logger.warning("GPU Validation: GPUs specified in CUDA_VISIBLE_DEVICES but none detected by PyTorch")
			
		if not visible_devices and cuda_device_count > 0:
			logger.warning("GPU Validation: PyTorch detected GPUs but CUDA_VISIBLE_DEVICES is empty")
		
		return {
			'cuda_available': cuda_available,
			'visible_devices': visible_devices,
			'visible_count': actual_visible_count,
			'cuda_device_count': cuda_device_count,
			'possible_issue': cuda_available and (actual_visible_count != cuda_device_count)
		}


	def monitor_cost_time_efficiency(self, epoch, baseline_metrics, current_metrics):
		"""Monitor cost, time, and efficiency during training"""
		
		if not baseline_metrics or not current_metrics:
			return {}
		
		# Get base metrics for comparison
		base_cost = baseline_metrics.get('cost_so_far', 0)
		base_time = baseline_metrics.get('time_so_far', 0)
		base_throughput = baseline_metrics.get('throughput', 0)
		
		# Get current metrics
		current_cost = current_metrics.get('cost_so_far', 0)
		current_time = current_metrics.get('time_so_far', 0)
		current_throughput = current_metrics.get('throughput', 0)
		
		# Calculate changes
		cost_change_pct = ((current_cost - base_cost) / max(0.001, base_cost)) * 100 if base_cost else 0
		time_change_pct = ((current_time - base_time) / max(0.001, base_time)) * 100 if base_time else 0
		throughput_change_pct = ((current_throughput - base_throughput) / max(0.001, base_throughput)) * 100 if base_throughput else 0
		
		# Check for problematic conditions
		issues = []
		
		# Problem: Both cost and time increasing
		if cost_change_pct > 5 and time_change_pct > 5:
			issues.append(f"WARNING: Both cost (+{cost_change_pct:.1f}%) and time (+{time_change_pct:.1f}%) are increasing")
		
		# Problem: Cost increasing but throughput decreasing
		if cost_change_pct > 5 and throughput_change_pct < -5:
			issues.append(f"WARNING: Cost increasing (+{cost_change_pct:.1f}%) while throughput decreasing ({throughput_change_pct:.1f}%)")
		
		# Problem: Time increasing but resource decreasing (could be expected in cost-focused mode)
		if time_change_pct > 10 and self.user_preference != 'cost':
			issues.append(f"WARNING: Time increasing significantly (+{time_change_pct:.1f}%) in {self.user_preference} mode")
		
		# Problem: Cost increasing but resource decreasing (could be expected in time-focused mode)
		if cost_change_pct > 10 and self.user_preference != 'time':
			issues.append(f"WARNING: Cost increasing significantly (+{cost_change_pct:.1f}%) in {self.user_preference} mode")
		
		# Log findings
		for issue in issues:
			logger.warning(f"Epoch {epoch}: {issue}")
		
		metrics = {
			'cost_change_pct': cost_change_pct,
			'time_change_pct': time_change_pct,
			'throughput_change_pct': throughput_change_pct,
			'issues': issues
		}
		
		return metrics

	def set_user_preference(self, preference: str):
		"""Set user preference for optimization strategy"""
		valid_preferences = ['balanced', 'cost', 'time', 'resource']
		if preference in valid_preferences:
			self.user_preference = preference
			logger.info(f"Set user preference to {preference}")
			return True
		else:
			logger.warning(f"Invalid preference: {preference}. Using 'balanced'")
			self.user_preference = 'balanced'
			return False
	
	def optimize_training(self, training_file_path: str, dataset_path: str = None,
						 offline_logs: List[str] = None, 
						 num_baseline_samples: int = 4, #<< change it here for steps
						 baseline_sample_ratio: float = 0.2) -> Dict:

		# Start timing
		start_time = time.time()

		# Ensure RL metrics directory exists
		os.makedirs("rl_metrics", exist_ok=True)

		# Load offline logs if provided
		if offline_logs:
			for log_path in offline_logs:
				self.offline_analyzer.add_log(log_path)
			self.offline_analyzer.analyze_all_logs()

		# Load training module
		logger.info(f"Loading training module: {training_file_path}")
		if not self.training_manager.load_module(training_file_path):
			logger.error(f"Failed to load training module: {training_file_path}")
			return {'status': 'failed', 'reason': 'Failed to load training module'}

		# Analyze module
		module_info = self.training_manager.analyze_module()
		logger.info(f"Module analysis: {module_info}")

		# Dynamically determine available GPUs and CPUs for baseline use
		available_gpus = torch.cuda.device_count()
		available_cpus = multiprocessing.cpu_count()

		logger.info(f"Detected {available_gpus} GPUs and {available_cpus} CPUs for baseline sampling")

		# Run multiple baseline training samples for better estimation
		logger.info(f"Starting {num_baseline_samples} baseline training samples (each {baseline_sample_ratio*100:.0f}% of total)")

		self.config['sla']['throughput_min'] = 2000.0
		self.config['sla']['gpu_util_min'] = 95.0
		self.config['sla']['cpu_util_min'] = 90.0

		# List to store all baseline metrics
		baseline_sample_metrics = []
		baseline_sample_summaries = []

		# Calculate the number of epochs for each sample
		if hasattr(self.training_manager.module, 'epochs'):
			total_epochs = self.training_manager.module.epochs
			sample_epochs = max(1, int(total_epochs * baseline_sample_ratio))
			logger.info(f"Total epochs: {total_epochs}, sample epochs: {sample_epochs}")
		else:
			# If we can't determine total epochs, use a default sample size
			sample_epochs = 15 # << change epochs here
			logger.info(f"Could not determine total epochs, using {sample_epochs} epochs per sample")

		# Set environment variable for sample epochs
		original_epochs = os.environ.get('OPTIMIZER_EPOCHS', None)
		os.environ['OPTIMIZER_EPOCHS'] = str(sample_epochs)

		# Run multiple baseline samples
		for sample_idx in range(num_baseline_samples):
			logger.info(f"Running baseline sample {sample_idx+1}/{num_baseline_samples}")
			
			# Set a different random seed for each sample to get different data subsets
			seed = int(time.time()) + sample_idx
			os.environ['OPTIMIZER_RANDOM_SEED'] = str(seed)
			
			if not self.training_manager.run_training(
				gpu_count=available_gpus,
				cpu_count=available_cpus,
				batch_size=None  # Let training script decide dynamically
			):
				logger.error(f"Baseline sample {sample_idx+1} failed")
				continue

			# Get CSV metrics for this sample
			sample_csv_path = self.training_manager.output_csv_path
			if not sample_csv_path or not os.path.exists(sample_csv_path):
				logger.error(f"No CSV found for baseline sample {sample_idx+1}")
				continue
			
			sample_metrics = pd.read_csv(sample_csv_path)
			sample_summary = self.training_manager.generate_summary()
			
			baseline_sample_metrics.append(sample_metrics)
			baseline_sample_summaries.append(sample_summary)
			
			logger.info(f"Baseline sample {sample_idx+1} completed: "
					   f"{sample_summary.get('throughput', 0):.2f} samples/sec, "
					   f"cost: ${sample_summary.get('cost_so_far', 0):.2f}")

		# Restore original epochs setting if it existed
		if original_epochs:
			os.environ['OPTIMIZER_EPOCHS'] = original_epochs
		else:
			del os.environ['OPTIMIZER_EPOCHS']

		# Check if we have at least one successful baseline sample
		if not baseline_sample_metrics:
			logger.error("All baseline samples failed")
			return {'status': 'failed', 'reason': 'All baseline samples failed'}

		# Combine baseline metrics and extrapolate to full training
		baseline_csv_path = baseline_sample_metrics[0].iloc[0]['model_name'] + "_combined_baseline.csv"
		baseline_metrics = pd.concat(baseline_sample_metrics, ignore_index=True)
		baseline_metrics.to_csv(baseline_csv_path, index=False)

		# Calculate average baseline summary by averaging the samples and scaling up
		baseline_summary = {}
		for key in ['time_so_far', 'cost_so_far', 'throughput']:
			values = [summary.get(key, 0) for summary in baseline_sample_summaries]
			avg_value = sum(values) / max(1, len(values))
			# Scale up time and cost to full training (not throughput)
			if key in ['time_so_far', 'cost_so_far'] and baseline_sample_ratio > 0:
				baseline_summary[key] = avg_value / baseline_sample_ratio
			else:
				baseline_summary[key] = avg_value

		# Copy other metrics from first sample summary
		for key, value in baseline_sample_summaries[0].items():
			if key not in baseline_summary:
				baseline_summary[key] = value

		logger.info(f"Averaged baseline summary (extrapolated to full training): "
				   f"time: {baseline_summary.get('time_so_far', 0)/60:.2f} min, "
				   f"cost: ${baseline_summary.get('cost_so_far', 0):.2f}, "
				   f"throughput: {baseline_summary.get('throughput', 0):.2f} samples/sec")

		
		# Get model info from CSV
		model_info = self.training_manager.get_model_info()
		if not model_info:
			logger.error("Failed to extract model info")
			return {'status': 'failed', 'reason': 'Failed to extract model info'}

		logger.info(f"Model info: {model_info}")

		# Get baseline metrics
		baseline_csv_path = self.training_manager.output_csv_path
		if not baseline_csv_path:
			logger.error("No baseline CSV found")
			return {'status': 'failed', 'reason': 'No baseline CSV found'}

		baseline_metrics = pd.read_csv(baseline_csv_path)
		baseline_summary = self.training_manager.generate_summary()
		logger.info(f"Baseline summary: {baseline_summary}")

		# Store baseline information
		baseline_info = {
			'csv_path': baseline_csv_path,
			'summary': baseline_summary,
			'gpu_count': baseline_summary.get('gpu_count', 0),
			'cpu_count': baseline_summary.get('cpu_count', 0),
			'memory_gb': baseline_summary.get('memory_used_gb', 0),
			'throughput': baseline_summary.get('throughput', 0),
			'training_time': baseline_summary.get('time_so_far', 0),
			'cost': baseline_summary.get('cost_so_far', 0)
		}

		# Initialize environment and agent
		self.cloud_env = CloudEnvironment(self.config)

		# Set the user preference for the environment
		self.cloud_env.user_preference = self.user_preference

		# Initialize training manager's resource config
		self.training_manager.resource_config = ResourceConfig(**self.config.get('resource', {}))

		# Initialize agent with the simple policy gradient approach
		self.agent = SimplePolicy(
			self.config,
			self.cloud_env.state_dim,
			self.cloud_env.action_dim
		)

		# Reset environment
		state = self.cloud_env.reset()

		# Initialize the metrics tracker
		self.metrics_tracker = RLMetricsTracker()

		num_rl_episodes = 5  # 🔁 Repeat training to give RL more chances to learn

		for episode in range(num_rl_episodes):
			logger.info(f"=== Starting RL Episode {episode + 1} ===")

			self.agent.reset()
			self.episode_reward = 0
			self.episode_steps = 0
			self.epoch_counter = 0

			epoch = None 
			
			previous_state_vector = None  # Initialize previous state for change detection # >> this block updated to make it adaptive <<
			# previous_state_vector = None  # Initialize previous state for change detection 
			for i, row in baseline_metrics.iterrows():
				try:
					epoch = int(float(row['epoch']))
				except (ValueError, TypeError):
					continue  # Skip any invalid rows

				state = self.cloud_env.update_metrics(row.to_dict())

				# Track performance
				self.metrics_tracker.add_performance_metrics({
					'epoch': epoch,
					'throughput': row.get('throughput', 0),
					'time_so_far': row.get('time_so_far', 0),
					'cost_so_far': row.get('cost_so_far', 0),
					'gpu_util': self.cloud_env.state.state_components['gpu_util'],
					'cpu_util': self.cloud_env.state.state_components['cpu_util'],
					'memory_util': self.cloud_env.state.state_components['memory_util']
				})

				logger.info(f"[epoch {epoch}] Metrics stored, SLA met: {self.cloud_env.state.get_overall_sla_compliance()}")

				# Get current state vector from environment (not from 'state' variable)
				current_state_vector = self.cloud_env.state.get_vector()
				
				# Detect if significant change or first state
				if previous_state_vector is None or self._detect_significant_change(current_state_vector, previous_state_vector):
					# Store state for next comparison
					previous_state_vector = current_state_vector.copy()
					
					logger.info(f"Processing epoch {epoch} in RL loop - significant change detected")
					
					action_idx, action_one_hot, _ = self.agent.select_action(state, explore=True)
					next_state, reward, done, info = self.cloud_env.step(action_idx)

					adjustment_metrics = self.cloud_env.log_adjustment_details(
						action_idx,
						info.get('prev_allocation', {}),
						self.cloud_env.current_allocation,
						reward
					)

					self.metrics_tracker.add_adjustment_metrics(adjustment_metrics)
					self.save_all_metrics()
				else:
					# Always update the previous state vector
					previous_state_vector = current_state_vector.copy()
					logger.info(f"[epoch {epoch}] No RL action taken - no significant change detected.")

			# # commneting out following block due to having the upper block to ensure adaptivness << >> starts here
			# for i, row in baseline_metrics.iterrows():
			#   try:
			#       epoch = int(float(row['epoch']))
			#   except (ValueError, TypeError):
			#       continue  # Skip any invalid rows

			#   state = self.cloud_env.update_metrics(row.to_dict())

			#   # Track performance
			#   self.metrics_tracker.add_performance_metrics({
			#       'epoch': epoch,
			#       'throughput': row.get('throughput', 0),
			#       'time_so_far': row.get('time_so_far', 0),
			#       'cost_so_far': row.get('cost_so_far', 0),
			#       'gpu_util': self.cloud_env.state.state_components['gpu_util'],
			#       'cpu_util': self.cloud_env.state.state_components['cpu_util'],
			#       'memory_util': self.cloud_env.state.state_components['memory_util']
			#   })

			#   logger.info(f"[epoch {epoch}] Metrics stored, SLA met: {self.cloud_env.state.get_overall_sla_compliance()}")

			#   current_state_vector = state.copy()  # Get current state vector
			#   # Detect if significant change or first state
			#   if previous_state_vector is None or self._detect_significant_change(current_state_vector, previous_state_vector):
			#       # Store state for next comparison
			#       previous_state_vector = current_state_vector.copy()         
			#   # if epoch % 5 == 0:  # Trigger RL action every 10 epochs now << change it here to tranck after every 10 epochs
			#       logger.info(f"Processing epoch {epoch} in RL loop")

			#       action_idx, action_one_hot, _ = self.agent.select_action(state, explore=True)
			#       next_state, reward, done, info = self.cloud_env.step(action_idx)

			#       adjustment_metrics = self.cloud_env.log_adjustment_details(
			#           action_idx,
			#           info.get('prev_allocation', {}),
			#           self.cloud_env.current_allocation,
			#           reward
			#       )

			#       self.metrics_tracker.add_adjustment_metrics(adjustment_metrics)
			#       self.save_all_metrics()
			#       # commneting out following block due to having the upper block to ensure adaptivness << >> ends here

					# Pareto tracking
					pareto_options = self.cloud_env.estimate_optimal_resources()
					self.metrics_tracker.add_pareto_front(epoch, pareto_options)

					# Train agent
					self.agent.add_experience(state, action_one_hot, reward, next_state, done)
					if i > 10:
						self.agent.train()
					else:
						logger.info(f"[epoch {epoch}] No RL action taken this epoch.")


			# After the loop, ensure epoch was defined at least once
			if epoch is not None:
				self.metrics_tracker.save_all_metrics()
			else:
				logger.warning("No epochs processed, skipping metrics save.")
				
		
		# ✅ Track Pareto
		pareto_options = self.cloud_env.estimate_optimal_resources()
		self.metrics_tracker.add_pareto_front(epoch, pareto_options)


		# Save all collected metrics from RL phase
		self.metrics_tracker.save_all_metrics()

		# Generate optimization options based on RL learning
		initial_options = self.cloud_env.estimate_optimal_resources()
		logger.info(f"Generated initial optimization options: {initial_options}")

		# Filter to 3 distinct options: time-focused, cost-focused, and balanced
		options_by_priority = {}
		for option in initial_options:
			options_by_priority[option['priority']] = option

		# Select options to actually test with another training run
		if not initial_options:
			# No options generated, create options based on user preference
			logger.warning("No optimization options generated, creating default options")
			
			# Create an appropriate test option based on user preference
			if self.user_preference == 'cost':
				# For cost optimization - reduce GPUs while maintaining CPU:GPU ratio
				gpu_utils = []
				for col in baseline_metrics.columns:
					if col.startswith('gpu_utilization_'):
						mean_util = baseline_metrics[col].mean()
						if not pd.isna(mean_util):
							gpu_utils.append(mean_util)

				avg_gpu_util = sum(gpu_utils) / max(1, len(gpu_utils)) if gpu_utils else 0

				# Make reduction based on utilization and preference
				if avg_gpu_util < 30:  # Very low utilization
					reduction_factor = 0.75  # Reduce by 75%
				elif avg_gpu_util < 50:
					reduction_factor = 0.5   # Reduce by 50%
				elif avg_gpu_util < 70:
					reduction_factor = 0.25  # Reduce by 25%
				else:
					reduction_factor = 0.1   # Reduce by 10%

				new_gpu_count = max(1, int(baseline_info.get('gpu_count', 8) * (1 - reduction_factor)))
				
				# Maintain a reasonable CPU:GPU ratio (8:1)
				new_cpu_count = max(8, min(new_gpu_count * 8, baseline_info.get('cpu_count', 4)))

				test_option = {
					'priority': 'cost',
					'name': 'Cost-Critical',
					'gpu_count': new_gpu_count,
					'cpu_count': new_cpu_count,
					'memory_gb': baseline_info.get('memory_gb', 8),
					'description': f'Minimize cost while maintaining acceptable performance (reduced GPUs by {int(reduction_factor*100)}%)'
				}

			elif self.user_preference == 'time':
				# For time optimization - increase GPUs if possible
				gpu_utils = []
				for col in baseline_metrics.columns:
					if col.startswith('gpu_utilization_'):
						mean_util = baseline_metrics[col].mean()
						if not pd.isna(mean_util):
							gpu_utils.append(mean_util)

				avg_gpu_util = sum(gpu_utils) / max(1, len(gpu_utils)) if gpu_utils else 0

				# If GPUs are severely underutilized but multiple are used, we might get better performance
				# with fewer, more fully utilized GPUs
				if avg_gpu_util < 20 and baseline_info.get('gpu_count', 1) > 4:
					# Very low utilization but many GPUs - try reducing to improve efficiency
					new_gpu_count = max(2, int(baseline_info.get('gpu_count', 8) * 0.5))
					desc = "Optimize for speed by improving GPU utilization efficiency"
				else:
					# Standard time optimization - increase resources if possible
					new_gpu_count = min(baseline_info.get('gpu_count', 1) + 1,
									   self.cloud_env.resource_config.gpu_count_max)
					desc = "Maximize training speed at higher cost"

				# Adjust CPU count to maintain proportional resources
				new_cpu_count = min(baseline_info.get('cpu_count', 4) + 4,
								   self.cloud_env.resource_config.cpu_count_max)

				test_option = {
					'priority': 'time',
					'name': 'Time-Critical',
					'gpu_count': new_gpu_count,
					'cpu_count': new_cpu_count,
					'memory_gb': baseline_info.get('memory_gb', 8),
					'description': desc
				}

			elif self.user_preference == 'resource':
				# For resource efficiency - optimize GPU utilization
				gpu_utils = []
				for col in baseline_metrics.columns:
					if col.startswith('gpu_utilization_'):
						mean_util = baseline_metrics[col].mean()
						if not pd.isna(mean_util):
							gpu_utils.append(mean_util)

				avg_gpu_util = sum(gpu_utils) / max(1, len(gpu_utils)) if gpu_utils else 0

				# Adjust GPU count based on utilization
				if avg_gpu_util < 40 and baseline_info.get('gpu_count', 1) > 1:
					# Low utilization - reduce GPUs
					new_gpu_count = max(1, baseline_info.get('gpu_count', 1) - 1)
					desc = "Optimize resource utilization by reducing underutilized GPUs"
				elif avg_gpu_util > 90:
					# High utilization - increase GPUs to avoid bottlenecks
					new_gpu_count = min(baseline_info.get('gpu_count', 1) + 1,
									   self.cloud_env.resource_config.gpu_count_max)
					desc = "Optimize resource utilization by adding GPUs to reduce bottlenecks"
				else:
					# Good utilization - maintain GPU count
					new_gpu_count = baseline_info.get('gpu_count', 1)
					desc = "Maintain optimal resource utilization"

				# Adjust CPU count for proper ratio
				new_cpu_count = new_gpu_count * 6  # 6 CPUs per GPU as a reasonable ratio

				test_option = {
					'priority': 'resource',
					'name': 'Resource-Efficient',
					'gpu_count': new_gpu_count,
					'cpu_count': new_cpu_count,
					'memory_gb': baseline_info.get('memory_gb', 8),
					'description': desc
				}

			else:  # balanced
				# For balanced - optimize based on utilization
				gpu_utils = []
				for col in baseline_metrics.columns:
					if col.startswith('gpu_utilization_'):
						mean_util = baseline_metrics[col].mean()
						if not pd.isna(mean_util):
							gpu_utils.append(mean_util)

				avg_gpu_util = sum(gpu_utils) / max(1, len(gpu_utils)) if gpu_utils else 0

				# Adjust GPU count based on utilization
				if avg_gpu_util < 30:
					# Very low utilization - reduce GPUs
					new_gpu_count = max(1, int(baseline_info.get('gpu_count', 1) * 0.75))
					desc = "Balanced configuration optimized for efficiency (reduced GPUs)"
				elif avg_gpu_util < 60:
					# Moderately low utilization - slight reduction
					new_gpu_count = max(1, int(baseline_info.get('gpu_count', 1) * 0.9))
					desc = "Balanced configuration with improved resource efficiency"
				elif avg_gpu_util > 90:
					# High utilization - slight increase
					new_gpu_count = min(baseline_info.get('gpu_count', 1) + 1,
									   self.cloud_env.resource_config.gpu_count_max)
					desc = "Balanced configuration with enough resources to prevent bottlenecks"
				else:
					# Good utilization - maintain GPU count
					new_gpu_count = baseline_info.get('gpu_count', 1)
					desc = "Current allocation (balanced between cost and performance)"

				# Adjust CPU count for proper ratio
				new_cpu_count = new_gpu_count * 6  # 6 CPUs per GPU as a reasonable ratio

				test_option = {
					'priority': 'balanced',
					'name': 'Balanced',
					'gpu_count': new_gpu_count,
					'cpu_count': new_cpu_count,
					'memory_gb': baseline_info.get('memory_gb', 8),
					'description': desc
				}

		else:
			# Use the generated options based on user preference
			if self.user_preference == 'balanced':
				test_option = options_by_priority.get('balanced', initial_options[0])
			elif self.user_preference == 'cost':
				test_option = options_by_priority.get('cost', initial_options[0])
			elif self.user_preference == 'time':
				test_option = options_by_priority.get('time', initial_options[0])
			else:
				test_option = options_by_priority.get('resource', initial_options[0])

			# Add this safety check
			if test_option is None:
				# If somehow we still don't have a test_option, create a safe default
				logger.warning("Failed to determine optimization option, using baseline configuration")
				test_option = {
					'priority': self.user_preference,
					'name': 'Baseline',
					'gpu_count': baseline_info.get('gpu_count', 1),
					'cpu_count': baseline_info.get('cpu_count', 2),
					'memory_gb': baseline_info.get('memory_gb', 8),
					'description': 'Using baseline configuration as fallback'
				}

			# Extract GPU utilization for batch size decisions
			gpu_utils = []
			for col in baseline_metrics.columns:
				if col.startswith('gpu_utilization_'):
					mean_util = baseline_metrics[col].mean()
					if not pd.isna(mean_util):
						gpu_utils.append(mean_util)

			avg_gpu_util = sum(gpu_utils) / max(1, len(gpu_utils)) if gpu_utils else 0

			# Run optimized training with the selected option
			logger.info(f"Running optimized training with {test_option['priority']} option: "
					   f"gpu_count={test_option['gpu_count']}, cpu_count={test_option['cpu_count']}")

			self.config['sla']['throughput_min'] = 500.0
			self.config['sla']['gpu_util_min'] = 40.0
			self.config['sla']['cpu_util_min'] = 40.0

			# For time optimization, maximize batch size
			if self.user_preference == 'time':
				# For time optimization, use larger batch size for maximum throughput
				optimized_batch_size = 128 * max(1, test_option['gpu_count'])
			
			else:
				# Extract baseline batch size
				baseline_batch_size = 128  # Default if not found
				if 'batch_size' in baseline_metrics.columns:
					baseline_batch_size = baseline_metrics['batch_size'].iloc[0]
				elif hasattr(self.training_manager.module, 'batch_size'):
					baseline_batch_size = self.training_manager.module.batch_size
				
				# Calculate per-GPU batch size from baseline
				if baseline_info['gpu_count'] > 0:
					baseline_per_gpu = baseline_batch_size / baseline_info['gpu_count']
				else:
					baseline_per_gpu = 128  # Default per GPU if we can't calculate
				
				# If GPU utilization is very low, don't reduce batch size as aggressively
				if avg_gpu_util < 15.0:  # Increased threshold from 5.0 to 15.0
					# More conservative approach - don't go below 75% of baseline batch size
					min_batch_size = int(0.75 * baseline_batch_size)
					new_batch_size = int(baseline_per_gpu * test_option['gpu_count'])
					optimized_batch_size = max(min_batch_size, new_batch_size)
					logger.info(f"Low GPU utilization ({avg_gpu_util:.2f}%) - using conservative batch size: {optimized_batch_size}")
				else:
					# Standard scaling based on GPU count
					optimized_batch_size = int(baseline_per_gpu * test_option['gpu_count'])
				
				# Ensure minimum reasonable batch size
				optimized_batch_size = max(32, optimized_batch_size)

			# Debug print: log the batch size decisions
			logger.info(f"Baseline GPU count: {baseline_info['gpu_count']}, Optimized GPU count: {test_option['gpu_count']}, " 
					  f"GPU Utilization: {avg_gpu_util:.2f}%, Computed Batch Size: {optimized_batch_size}")

			# # Set the environment variable so the training module uses the computed batch size
			# os.environ['OPTIMIZER_BATCH_SIZE'] = str(optimized_batch_size)

		# --- Ensure optimized_batch_size is defined ---
		if 'optimized_batch_size' not in locals():
			# Fallback: extract a default per-GPU batch size from baseline metrics or module settings.
			default_per_gpu_batch_size = 64  # Absolute fallback
			if 'batch_size' in baseline_metrics.columns:
				base_batch = baseline_metrics['batch_size'].iloc[0]
				if baseline_info['gpu_count'] > 0:
					default_per_gpu_batch_size = base_batch / baseline_info['gpu_count']
			elif hasattr(self.training_manager.module, 'batch_size'):
				base_batch = self.training_manager.module.batch_size
				if baseline_info['gpu_count'] > 0:
					default_per_gpu_batch_size = base_batch / baseline_info['gpu_count']
			logger.warning("Batch size not determined, calculating from baseline")
			optimized_batch_size = int(default_per_gpu_batch_size * max(1, test_option['gpu_count']))

		# --- Set the environment variable for the batch size ---
		os.environ['OPTIMIZER_BATCH_SIZE'] = str(optimized_batch_size)
		logger.info(f"Using optimized batch size: {optimized_batch_size}")

		# --- Apply centralized resource configuration ---
		ResourceManager.apply_configuration(
			gpu_count=test_option['gpu_count'],
			cpu_count=test_option['cpu_count'],
			batch_size=optimized_batch_size,
			resource_config=self.training_manager.resource_config.__dict__
		)
		logger.info(f"Applied ResourceManager configuration: GPU={test_option['gpu_count']}, CPU={test_option['cpu_count']}, Batch size={optimized_batch_size}")

		# Ensure the optimized run uses all epochs (remove any epoch restriction)
		if 'OPTIMIZER_EPOCHS' in os.environ:
			del os.environ['OPTIMIZER_EPOCHS']
		logger.info("Running optimized training with full epochs")

		# --- Run the optimized training ---
		if not self.training_manager.run_training(
				gpu_count=test_option['gpu_count'],
				cpu_count=test_option['cpu_count'],
				batch_size=optimized_batch_size
		):
			logger.error("Optimized training failed")
			optimized_metrics = None
			optimized_summary = None
		else:
			optimized_csv_path = self.training_manager.output_csv_path
			if not optimized_csv_path:
				logger.error("No optimized CSV found")
				optimized_metrics = None
				optimized_summary = None
			else:
				optimized_metrics = pd.read_csv(optimized_csv_path)
				optimized_summary = self.training_manager.generate_summary()
				logger.info(f"Optimized summary: {optimized_summary}")
				
				# Add RL processing for optimized phase
				if optimized_metrics is not None:
					# Reset environment for optimized phase
					self.cloud_env.reset()
					previous_state_vector = None
					
					# Process each epoch of optimized metrics
					for i, row in optimized_metrics.iterrows():
						try:
							epoch = int(float(row['epoch']))
						except (ValueError, TypeError):
							continue  # Skip any invalid rows
							
						# Update environment with optimized metrics
						state = self.cloud_env.update_metrics(row.to_dict())
						
						# Get current state vector
						current_state_vector = self.cloud_env.state.get_vector()
						
						# Log basic metrics
						logger.info(f"[OPTIMIZED - epoch {epoch}] Metrics stored, SLA met: {self.cloud_env.state.get_overall_sla_compliance()}")
						
						# Detect if significant change occurred
						if previous_state_vector is None or self._detect_significant_change(current_state_vector, previous_state_vector):
							previous_state_vector = current_state_vector.copy()
							
							# Select action based on learned policy
							action_idx, action_one_hot, _ = self.agent.select_action(state, explore=False)
							
							# Calculate reward and update state
							next_state, reward, done, info = self.cloud_env.step(action_idx)
							
							# Get adjustment details
							adjustment_metrics = self.cloud_env.log_adjustment_details(
								action_idx,
								info.get('prev_allocation', {}),
								self.cloud_env.current_allocation,
								reward
							)
							
							# Save metrics for reporting
							self.metrics_tracker.add_adjustment_metrics(adjustment_metrics)
							
							# Log detailed RL adjustment summary
							logger.info(f"\n========== OPTIMIZED RL Adjustment — Epoch {epoch} ==========")
							logger.info(f"🎯 Action Taken       : {adjustment_metrics['action_description']}")
							logger.info(f"🧠 Selected by Agent  : Action #{adjustment_metrics['action']} → {adjustment_metrics['action_description']}")
							logger.info(f"🔧 Resource Allocation: GPUs {info.get('prev_allocation', {}).get('gpu_count', 0)} → {adjustment_metrics['gpu_count']}, "
									  f"CPUs {info.get('prev_allocation', {}).get('cpu_count', 0)} → {adjustment_metrics['cpu_count']}, "
									  f"Memory {info.get('prev_allocation', {}).get('memory_gb', 0):.1f}GB → {adjustment_metrics['memory_gb']:.1f}GB")
							logger.info(f"💰 Reward: {reward:.2f} | Cost: {adjustment_metrics['reward_cost']:.2f} | SLA: {adjustment_metrics['reward_sla']:.2f}")
							logger.info("==========================================================\n")
						else:
							# Just update the state vector for comparison
							previous_state_vector = current_state_vector.copy()
							logger.info(f"[OPTIMIZED - epoch {epoch}] No action taken - no significant change detected")

		# Store optimized information
		optimized_info = {}
		if optimized_summary:
			optimized_info = {
				'csv_path': self.training_manager.output_csv_path,
				'summary': optimized_summary,
				'gpu_count': test_option['gpu_count'],
				'cpu_count': test_option['cpu_count'],
				'memory_gb': optimized_summary.get('memory_used_gb', 0),
				'throughput': optimized_summary.get('throughput', 0),
				'training_time': optimized_summary.get('time_so_far', 0),
				'cost': optimized_summary.get('cost_so_far', 0),
				'option_used': test_option['priority']
			}

		# Generate additional optimization options for the report
		final_options = []
		for priority in ['time', 'cost', 'balanced']:
			if priority in options_by_priority:
				option = options_by_priority[priority]

				# Estimate training time based on baseline and throughput relationship
				if baseline_info['throughput'] > 0:
					# Time is inversely proportional to throughput, adjusted by GPU count
					est_throughput_factor = 1.0
					if option['gpu_count'] > baseline_info['gpu_count']:
						# More GPUs generally mean more throughput, but not linearly
						est_throughput_factor = (option['gpu_count'] / baseline_info['gpu_count']) ** 0.7
					elif option['gpu_count'] < baseline_info['gpu_count']:
						est_throughput_factor = (option['gpu_count'] / baseline_info['gpu_count']) ** 0.9

					estimated_throughput = baseline_info['throughput'] * est_throughput_factor
					estimated_time = baseline_info['training_time'] * (baseline_info['throughput'] / estimated_throughput)

					# IMPORTANT: Calculate cost based on resources and time
					resource_hourly_cost = (
						option['gpu_count'] * self.cloud_env.resource_config.cost_per_gpu_hour +
						option['cpu_count'] * self.cloud_env.resource_config.cost_per_cpu_hour +
						option['memory_gb'] * self.cloud_env.resource_config.cost_per_gb_hour
					)
					estimated_cost = (estimated_time / 3600) * resource_hourly_cost

					# Adjust if we have actual results for this option
					if optimized_info and optimized_info['option_used'] == option['priority']:
						estimated_time = optimized_info['training_time']
						estimated_cost = optimized_info['cost']

					# Enhance option with estimates
					option['estimated_time'] = estimated_time
					option['estimated_cost'] = estimated_cost
					option['estimated_throughput'] = estimated_throughput

					# Add improvement percentages
					if baseline_info['training_time'] > 0:
						option['time_change_pct'] = ((estimated_time - baseline_info['training_time']) /
												  baseline_info['training_time']) * 100
					if baseline_info['cost'] > 0:
						option['cost_change_pct'] = ((estimated_cost - baseline_info['cost']) /
												 baseline_info['cost']) * 100

				final_options.append(option)

		# Ensure we have distinct options
		if len(final_options) < 3:
			# Add more options if needed
			if not any(o['priority'] == 'time' for o in final_options):
				time_option = {
					'priority': 'time',
					'name': 'Time-Critical',
					'gpu_count': min(baseline_info['gpu_count'] + 2, self.cloud_env.resource_config.gpu_count_max),
					'cpu_count': min(baseline_info['cpu_count'] + 4, self.cloud_env.resource_config.cpu_count_max),
					'memory_gb': baseline_info['memory_gb'],
					'estimated_time': baseline_info['training_time'] * 0.6,  # 40% faster
					'estimated_cost': baseline_info['cost'] * 1.3,  # 30% more expensive
					'estimated_throughput': baseline_info['throughput'] * 1.4,
					'time_change_pct': -40.0,
					'cost_change_pct': 30.0,
					'description': 'Maximize training speed at higher cost'
				}
				final_options.append(time_option)

			if not any(o['priority'] == 'cost' for o in final_options):
				cost_option = {
					'priority': 'cost',
					'name': 'Cost-Critical',
					'gpu_count': max(baseline_info['gpu_count'] - 1, 1),
					'cpu_count': max(baseline_info['cpu_count'] - 2, 2),
					'memory_gb': baseline_info['memory_gb'],
					'estimated_time': baseline_info['training_time'] * 1.3,  # 30% slower
					'estimated_cost': baseline_info['cost'] * 0.7,  # 30% cheaper
					'estimated_throughput': baseline_info['throughput'] * 0.7,
					'time_change_pct': 30.0,
					'cost_change_pct': -30.0,
					'description': 'Minimize cost while maintaining acceptable performance'
				}
				final_options.append(cost_option)

			if not any(o['priority'] == 'balanced' for o in final_options):
				balanced_option = {
					'priority': 'balanced',
					'name': 'Balanced',
					'gpu_count': baseline_info['gpu_count'],
					'cpu_count': baseline_info['cpu_count'],
					'memory_gb': baseline_info['memory_gb'],
					'estimated_time': baseline_info['training_time'],
					'estimated_cost': baseline_info['cost'],
					'estimated_throughput': baseline_info['throughput'],
					'time_change_pct': 0.0,
					'cost_change_pct': 0.0,
					'description': 'Current allocation (balanced between cost and performance)'
				}
				final_options.append(balanced_option)

		# Create extended CSV with RL metrics if we have baseline metrics
		extended_csv_path = ''
		if len(self.rl_metrics) > 0 and baseline_csv_path:
			# Create a CSV monitor to handle the file
			csv_monitor = CSVMonitor(baseline_csv_path)
			for i, row in self.rl_metrics.iterrows():
				csv_monitor.append_rl_metrics(int(row['epoch']), row.to_dict())

			# Also save a separate RL metrics CSV
			rl_csv_path = os.path.splitext(baseline_csv_path)[0] + '_rl_metrics.csv'
			self.rl_metrics.to_csv(rl_csv_path, index=False)
			logger.info(f"Saved RL metrics to {rl_csv_path}")

			# Create the extended CSV
			extended_csv_path = csv_monitor.create_extended_csv(baseline_csv_path, self.rl_metrics)

		# Record SLA violations for reporting
		baseline_sla_status = {}
		optimized_sla_status = {}

		# Extract SLA status from baseline
		if baseline_metrics is not None:
			# Update environment with last epoch metrics to check SLA
			last_metrics = baseline_metrics[baseline_metrics['epoch'] != 'SUMMARY'].iloc[-1].to_dict()
			self.cloud_env.update_metrics(last_metrics)
			baseline_sla_status = self.cloud_env.state.get_sla_status()

		# Extract SLA status from optimized run
		if optimized_metrics is not None:
			# Update environment with last epoch metrics to check SLA
			last_metrics = optimized_metrics[optimized_metrics['epoch'] != 'SUMMARY'].iloc[-1].to_dict()
			self.cloud_env.update_metrics(last_metrics)
			optimized_sla_status = self.cloud_env.state.get_sla_status()

		# Generate a final report
		total_time = time.time() - start_time

		report = {
			'status': 'success',
			'training_file': training_file_path,
			'optimization_time': total_time,
			'baseline': baseline_info,
			'optimized': optimized_info,
			'final_options': final_options,
			'rl_metrics_file': extended_csv_path,
			'baseline_sla_status': baseline_sla_status,
			'optimized_sla_status': optimized_sla_status,
			'user_preference': self.user_preference,
			'num_baseline_samples': num_baseline_samples,  # Add this and next line
			'baseline_sample_ratio': baseline_sample_ratio
		}

		# Save all tracked metrics
		metrics_files = self.metrics_tracker.save_all_metrics(use_timestamp=False)
		report['metrics_files'] = metrics_files

		# # if you to save with timestamps
		# metrics_files = self.metrics_tracker.save_all_metrics()
		# report['metrics_files'] = metrics_files

		# Save the agent
		self.agent.save()

		logger.info(f"Optimization completed successfully in {total_time:.2f} seconds")
		return report

	# >> this block updated to make it adaptive <<
	def _detect_significant_change(self, current_state, previous_state, threshold=0.1):
		"""
		Detect if there has been a significant change in the environment state
		that warrants taking a new action.
		"""
		# Calculate normalized change in state
		state_diff = np.abs(current_state - previous_state)
		
		# Focus on the most important metrics for change detection
		important_indices = [3, 4, 5, 7, 10, 11, 12]  # Utilization and SLA metrics
		
		# Check if any important metric changed significantly
		for idx in important_indices:
			if idx < len(state_diff) and state_diff[idx] > threshold:
				return True
				
		return False

	def save_optimization_metrics(self, result: Dict, output_file: str = 'optimization_metrics.csv'):
		"""Save optimization metrics in a format suitable for plotting"""
		# Extract key metrics for plotting
		metrics = {
			'experiment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
			'preference': result['user_preference'],
			'training_file': result['training_file'],
			
			# Add uncertainty metrics if available
			'average_uncertainty': self.rl_metrics['rl_uncertainty'].mean() if 'rl_uncertainty' in self.rl_metrics.columns else None,
			'max_uncertainty': self.rl_metrics['rl_uncertainty'].max() if 'rl_uncertainty' in self.rl_metrics.columns else None,
					
			# Baseline metrics
			'baseline_gpu_count': result['baseline'].get('gpu_count', 0),
			'baseline_cpu_count': result['baseline'].get('cpu_count', 0),
			'baseline_memory_gb': result['baseline'].get('memory_gb', 0),
			'baseline_time_minutes': result['baseline'].get('training_time', 0) / 60,
			'baseline_cost': result['baseline'].get('cost', 0),
			'baseline_throughput': result['baseline'].get('throughput', 0),
			
			# Optimized metrics
			'optimized_gpu_count': result.get('optimized', {}).get('gpu_count', 0),
			'optimized_cpu_count': result.get('optimized', {}).get('cpu_count', 0),
			'optimized_memory_gb': result.get('optimized', {}).get('memory_gb', 0),
			'optimized_time_minutes': result.get('optimized', {}).get('training_time', 0) / 60,
			'optimized_cost': result.get('optimized', {}).get('cost', 0),
			'optimized_throughput': result.get('optimized', {}).get('throughput', 0),
			
			# Improvement percentages
			'gpu_reduction_pct': ((result['baseline'].get('gpu_count', 0) - result.get('optimized', {}).get('gpu_count', 0)) / 
								  max(1, result['baseline'].get('gpu_count', 1))) * 100,
			'time_reduction_pct': ((result['baseline'].get('training_time', 0) - result.get('optimized', {}).get('training_time', 0)) / 
								  max(1, result['baseline'].get('training_time', 1))) * 100,
			'cost_reduction_pct': ((result['baseline'].get('cost', 0) - result.get('optimized', {}).get('cost', 0)) / 
								  max(1, result['baseline'].get('cost', 1))) * 100,
			
			# SLA compliance
			'baseline_sla_met_count': sum(1 for status in result.get('baseline_sla_status', {}).values() if status.get('met', False)),
			'optimized_sla_met_count': sum(1 for status in result.get('optimized_sla_status', {}).values() if status.get('met', False)),
			
			# Total optimization time
			'optimization_time_minutes': result['optimization_time'] / 60
		}
		
		# Save to CSV (append if exists)
		try:
			if os.path.exists(output_file):
				df = pd.read_csv(output_file)
				df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
			else:
				df = pd.DataFrame([metrics])
			
			df.to_csv(output_file, index=False)
			logger.info(f"Saved optimization metrics to {output_file}")
		except Exception as e:
			logger.error(f"Error saving optimization metrics: {str(e)}")
	
	def generate_report(self, result: Dict, output_file: str = 'optimization_report.txt') -> str:
		"""Generate a comprehensive report of the optimization results"""
		if result['status'] != 'success':
			return f"Optimization failed: {result.get('reason', 'Unknown error')}"
		
		report = "=== Cloud Resource Optimization Report ===\n\n"
		
		# Training details
		report += "Training Details:\n"
		report += f"Training file: {result['training_file']}\n"
		report += f"Baseline: {result.get('num_baseline_samples', 1)} samples at {result.get('baseline_sample_ratio', 1.0)*100:.0f}% of total epochs\n"
		report += f"Total optimization time: {result['optimization_time']/60:.2f} minutes\n"
		report += f"User preference: {result['user_preference']}\n\n"
		
		# Baseline details
		baseline = result.get('baseline', {})
		if baseline:
			report += "Baseline Training Run:\n"
			report += f"Resources: {baseline.get('gpu_count', 0)} GPUs, "
			report += f"{baseline.get('cpu_count', 0)} CPUs, "
			report += f"{baseline.get('memory_gb', 0):.1f} GB memory\n"
			report += f"Training time: {baseline.get('training_time', 0)/60:.2f} minutes\n"
			report += f"Cost: ${baseline.get('cost', 0):.2f}\n"
			report += f"Throughput: {baseline.get('throughput', 0):.2f} samples/sec\n\n"
		
		# SLA compliance for baseline
		baseline_sla = result.get('baseline_sla_status', {})
		if baseline_sla:
			report += "Baseline SLA Compliance:\n"
			for sla_name, sla_info in baseline_sla.items():
				status = "✓ Met" if sla_info.get('met', False) else f"✗ Violated (severity: {sla_info.get('severity', 0):.2f})"
				report += f"  {sla_name.replace('_', ' ').title()}: {status}\n"
			report += "\n"
		
		# Optimized details
		optimized = result.get('optimized', {})
		if optimized:
			report += "Optimized Training Run:\n"
			report += f"Optimization strategy: {optimized.get('option_used', 'unknown')}\n"
			report += f"Resources: {optimized.get('gpu_count', 0)} GPUs, "
			report += f"{optimized.get('cpu_count', 0)} CPUs, "
			report += f"{optimized.get('memory_gb', 0):.1f} GB memory\n"
			report += f"Training time: {optimized.get('training_time', 0)/60:.2f} minutes\n"
			report += f"Cost: ${optimized.get('cost', 0):.2f}\n"
			report += f"Throughput: {optimized.get('throughput', 0):.2f} samples/sec\n\n"
			
			# Calculate improvements
			if baseline:
				time_change = ((optimized.get('training_time', 0) - baseline.get('training_time', 0)) / 
							  baseline.get('training_time', 1)) * 100
				
				baseline_cost = baseline.get('cost', 0)
				optimized_cost = optimized.get('cost', 0)

				if baseline_cost != 0:
					cost_change = (optimized_cost - baseline_cost) / baseline_cost
				else:
					cost_change = float('inf') if optimized_cost > 0 else 0  # Avoid division by zero


				# cost_change = ((optimized.get('cost', 0) - baseline.get('cost', 0)) / 
				#             baseline.get('cost', 1)) * 100
				

				throughput_change = ((optimized.get('throughput', 0) - baseline.get('throughput', 0)) / 
									baseline.get('throughput', 1)) * 100
				
				report += "Improvements from Baseline:\n"
				report += f"  Time: {time_change:.1f}% ({'faster' if time_change < 0 else 'slower'})\n"
				report += f"  Cost: {cost_change:.1f}% ({'reduction' if cost_change < 0 else 'increase'})\n"
				report += f"  Throughput: {throughput_change:.1f}% ({'increase' if throughput_change > 0 else 'decrease'})\n\n"

				# Add resource changes section
				gpu_change = optimized.get('gpu_count', 0) - baseline.get('gpu_count', 0)
				cpu_change = optimized.get('cpu_count', 0) - baseline.get('cpu_count', 0)

				report += "Resource Changes:\n"
				if gpu_change != 0:
					report += f"  GPUs: {baseline.get('gpu_count', 0)} → {optimized.get('gpu_count', 0)} "
					report += f"({'increased' if gpu_change > 0 else 'decreased'} by {abs(gpu_change)})\n"
				else:
					report += f"  GPUs: {baseline.get('gpu_count', 0)} (unchanged)\n"
					
				if cpu_change != 0:
					report += f"  CPUs: {baseline.get('cpu_count', 0)} → {optimized.get('cpu_count', 0)} "
					report += f"({'increased' if cpu_change > 0 else 'decreased'} by {abs(cpu_change)})\n"
				else:
					report += f"  CPUs: {baseline.get('cpu_count', 0)} (unchanged)\n"

				report += "\n"

		
		# SLA compliance for optimized run
		optimized_sla = result.get('optimized_sla_status', {})
		if optimized_sla:
			report += "Optimized SLA Compliance:\n"
			for sla_name, sla_info in optimized_sla.items():
				status = "✓ Met" if sla_info.get('met', False) else f"✗ Violated (severity: {sla_info.get('severity', 0):.2f})"
				report += f"  {sla_name.replace('_', ' ').title()}: {status}\n"
			report += "\n"
			
			# Compare SLA improvements
			if baseline_sla:
				report += "SLA Compliance Improvements:\n"
				for sla_name in baseline_sla:
					if sla_name in optimized_sla:
						baseline_met = baseline_sla[sla_name].get('met', False)
						optimized_met = optimized_sla[sla_name].get('met', False)
						
						if not baseline_met and optimized_met:
							report += f"  {sla_name.replace('_', ' ').title()}: Fixed violation ✓\n"
						elif baseline_met and not optimized_met:
							report += f"  {sla_name.replace('_', ' ').title()}: New violation ✗\n"
						elif not baseline_met and not optimized_met:
							baseline_severity = baseline_sla[sla_name].get('severity', 0)
							optimized_severity = optimized_sla[sla_name].get('severity', 0)
							severity_change = ((optimized_severity - baseline_severity) / 
											  max(0.01, baseline_severity)) * 100
							
							if severity_change < 0:
								report += f"  {sla_name.replace('_', ' ').title()}: "
								report += f"Violation improved by {-severity_change:.1f}% "
								report += f"(severity: {baseline_severity:.2f} → {optimized_severity:.2f})\n"
							elif severity_change > 0:
								report += f"  {sla_name.replace('_', ' ').title()}: "
								report += f"Violation worsened by {severity_change:.1f}% "
								report += f"(severity: {baseline_severity:.2f} → {optimized_severity:.2f})\n"
				report += "\n"
		
		# RL decision process
		rl_metrics = result.get('rl_metrics_file', '')
		if rl_metrics:
			report += "Reinforcement Learning Process:\n"
			report += f"  Extended metrics file: {rl_metrics}\n"
			report += "  The RL agent learned from the baseline training run to make resource allocation\n"
			report += "  decisions based on SLA requirements and user preferences. It simulated actions\n"
			report += "  to find the optimal resource configuration for the given constraints.\n\n"
		
		# Optimization options
		if 'final_options' in result and result['final_options']:
			report += "Optimization Recommendations:\n"
			for option in result['final_options']:
				report += f"\n{option['name']} Option ({option['priority']}):\n"
				report += f"  GPUs: {option['gpu_count']}\n"
				report += f"  CPUs: {option['cpu_count']}\n"
				report += f"  Memory: {option['memory_gb']:.1f} GB\n"
				
				if 'estimated_time' in option:
					report += f"  Estimated training time: {option['estimated_time']/60:.2f} minutes"
					if 'time_change_pct' in option:
						report += f" ({option['time_change_pct']:.1f}% change)\n"
					else:
						report += "\n"
						
				if 'estimated_cost' in option:
					report += f"  Estimated cost: ${option['estimated_cost']:.2f}"
					if 'cost_change_pct' in option:
						report += f" ({option['cost_change_pct']:.1f}% change)\n"
					else:
						report += "\n"
						
				if 'estimated_throughput' in option:
					report += f"  Estimated throughput: {option['estimated_throughput']:.2f} samples/sec\n"
					
				if 'description' in option:
					report += f"  Description: {option['description']}\n"
		
		# Uncertainty Analysis section here
		report += "\nUncertainty Analysis:\n"
		if 'rl_uncertainty' in self.rl_metrics.columns:
			avg_uncertainty = self.rl_metrics['rl_uncertainty'].mean()
			max_uncertainty = self.rl_metrics['rl_uncertainty'].max()
			report += f"  Average prediction uncertainty: {avg_uncertainty:.4f}\n"
			report += f"  Maximum prediction uncertainty: {max_uncertainty:.4f}\n"
			report += "  Higher uncertainty indicates less confidence in resource decisions.\n"
			
			if avg_uncertainty > 0.3:
				report += "  ⚠ High uncertainty detected - consider longer training or more stable workloads.\n"
			else:
				report += "  ✓ Uncertainty levels acceptable for reliable resource decisions.\n"
		report += "\n"

		# Detailed MDP process explanation
		report += "\nReinforcement Learning Details:\n"
		report += "The optimization process used a Markov Decision Process (MDP) formulation where:\n"
		report += "  - States: Resource utilization, training progress, SLA compliance\n"
		report += "  - Actions: Increase/decrease GPUs, CPUs, memory\n"
		report += "  - Rewards: Based on cost efficiency, performance, and SLA compliance\n"
		report += "  - Policies: Different strategies for cost, time, and resource optimization\n\n"
		
		report += "The agent trained on the baseline run to learn optimal resource allocations\n"
		report += "for different objectives. It used multi-objective reinforcement learning to\n"
		report += "balance competing objectives like cost, time, and resource utilization.\n\n"
		
		# Summary of achievements
		report += "Summary of Achievements:\n"
		if optimized and baseline:
			time_change = ((optimized.get('training_time', 0) - baseline.get('training_time', 0)) / 
						  baseline.get('training_time', 1)) * 100
			
			baseline_cost = baseline.get('cost', 0)
			optimized_cost = optimized.get('cost', 0)

			if baseline_cost != 0:
				cost_change = (optimized_cost - baseline_cost) / baseline_cost
			else:
				cost_change = float('inf') if optimized_cost > 0 else 0
			
			if time_change < 0 or cost_change < 0:
				if time_change < 0 and cost_change < 0:
					report += f"✓ Successfully improved both training time ({-time_change:.1f}% reduction) and\n"
					report += f"  cost ({-cost_change:.1f}% reduction) simultaneously!\n"
				elif time_change < 0:
					report += f"✓ Successfully reduced training time by {-time_change:.1f}% based on user preference.\n"
				elif cost_change < 0:
					report += f"✓ Successfully reduced cost by {-cost_change:.1f}% based on user preference.\n"
			else:
				report += "⚠ The optimized run did not improve time or cost metrics.\n"
				report += "  This could be due to challenging SLA constraints or resource limitations.\n"
		
		# SLA achievement summary
		if baseline_sla and optimized_sla:
			baseline_met_count = sum(1 for status in baseline_sla.values() if status.get('met', False))
			optimized_met_count = sum(1 for status in optimized_sla.values() if status.get('met', False))
			
			if optimized_met_count > baseline_met_count:
				report += f"✓ Improved SLA compliance from {baseline_met_count}/{len(baseline_sla)} to "
				report += f"{optimized_met_count}/{len(optimized_sla)} SLAs met.\n"
			elif optimized_met_count == baseline_met_count:
				report += f"SLA compliance remained at {optimized_met_count}/{len(optimized_sla)} SLAs met.\n"
			else:
				report += f"⚠ SLA compliance decreased from {baseline_met_count}/{len(baseline_sla)} to "
				report += f"{optimized_met_count}/{len(optimized_sla)} SLAs met.\n"
				report += "  This is likely due to prioritizing the user's preference over certain SLAs.\n"
		
		report += "\n=== End of Report ===\n"
		
		# Save to file
		with open(output_file, 'w') as f:
			f.write(report)
		
		logger.info(f"Report saved to {output_file}")
		
		# Add before returning the report
		# self.save_optimization_metrics(report)
		self.save_optimization_metrics(result)

		return report

# ---------------------------------------------------------------------------
# The ResourceManager class definition.
# ---------------------------------------------------------------------------
class ResourceManager:
	@staticmethod
	def apply_configuration(gpu_count: int, cpu_count: int, batch_size: int, resource_config: dict):
		"""Apply resource configuration with improved GPU handling"""
		logger = logging.getLogger(__name__)
		
		# Record the original environment variables to verify changes later
		original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
		
		# --- Handle GPU allocation ---
		available_gpus = []
		try:
			available_gpus = list(range(torch.cuda.device_count()))
		except Exception as e:
			logger.warning(f"Error detecting GPUs: {str(e)}")
			available_gpus = []
			
		if gpu_count == 0 or not available_gpus:
			os.environ['CUDA_VISIBLE_DEVICES'] = ""
			logger.info("No GPU access: CUDA_VISIBLE_DEVICES set to empty string")
		else:
			visible_gpus = available_gpus[:gpu_count]
			os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, visible_gpus))
			logger.info(f"Setting CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")
		
		# Verify GPU configuration was applied
		logger.info(f"CUDA_VISIBLE_DEVICES changed from '{original_cuda}' to '{os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}'")
		
		# --- Handle CPU threads ---
		os.environ['OMP_NUM_THREADS'] = str(cpu_count)
		os.environ['MKL_NUM_THREADS'] = str(cpu_count)
		os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
		os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
		logger.info(f"Set CPU thread environment variables to {cpu_count}")
		
		# --- Set training parameters (batch size) ---
		os.environ['OPTIMIZER_BATCH_SIZE'] = str(batch_size)
		logger.info(f"Set training batch size to {batch_size}")
		
		# --- Set cost factors from resource_config ---
		os.environ['COST_PER_GPU_HOUR'] = str(resource_config.get('cost_per_gpu_hour', 10.0))
		os.environ['COST_PER_CPU_HOUR'] = str(resource_config.get('cost_per_cpu_hour', 3.0))
		os.environ['COST_PER_GB_HOUR'] = str(resource_config.get('cost_per_gb_hour', 2.0))
		logger.info("Applied cost factor environment variables")
		
		# Return the configuration for validation
		return {
			'gpu_count': gpu_count,
			'cpu_count': cpu_count,
			'batch_size': batch_size,
			'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')
		}

class ResourceCostCalculator:
	"""Helper class to calculate and validate resource costs"""
	
	def __init__(self, cost_per_gpu_hour=10.0, cost_per_cpu_hour=3.0, cost_per_gb_hour=2.0):
		self.cost_per_gpu_hour = cost_per_gpu_hour
		self.cost_per_cpu_hour = cost_per_cpu_hour
		self.cost_per_gb_hour = cost_per_gb_hour
	
	def calculate_hourly_cost(self, gpu_count, cpu_count, memory_gb):
		"""Calculate the hourly cost of the given resources"""
		return (
			gpu_count * self.cost_per_gpu_hour +
			cpu_count * self.cost_per_cpu_hour +
			memory_gb * self.cost_per_gb_hour
		)
	
	def compare_configurations(self, config1, config2, training_times=None):
		
		hourly_cost1 = self.calculate_hourly_cost(
			config1.get('gpu_count', 0),
			config1.get('cpu_count', 0),
			config1.get('memory_gb', 0)
		)
		
		hourly_cost2 = self.calculate_hourly_cost(
			config2.get('gpu_count', 0),
			config2.get('cpu_count', 0),
			config2.get('memory_gb', 0)
		)
		
		# Hourly cost change
		hourly_cost_change = hourly_cost2 - hourly_cost1
		hourly_cost_change_pct = (hourly_cost_change / max(0.01, hourly_cost1)) * 100
		
		results = {
			'baseline_hourly_cost': hourly_cost1,
			'optimized_hourly_cost': hourly_cost2,
			'hourly_cost_change': hourly_cost_change,
			'hourly_cost_change_pct': hourly_cost_change_pct,
			'gpu_change': config2.get('gpu_count', 0) - config1.get('gpu_count', 0),
			'cpu_change': config2.get('cpu_count', 0) - config1.get('cpu_count', 0),
			'memory_change': config2.get('memory_gb', 0) - config1.get('memory_gb', 0),
		}
		
		# If training times are provided, calculate total cost
		if training_times:
			baseline_time, optimized_time = training_times
			
			# Convert to hours
			baseline_hours = baseline_time / 3600.0
			optimized_hours = optimized_time / 3600.0
			
			baseline_total_cost = hourly_cost1 * baseline_hours
			optimized_total_cost = hourly_cost2 * optimized_hours
			
			total_cost_change = optimized_total_cost - baseline_total_cost
			total_cost_change_pct = (total_cost_change / max(0.01, baseline_total_cost)) * 100
			
			time_change = optimized_time - baseline_time
			time_change_pct = (time_change / max(0.01, baseline_time)) * 100
			
			results.update({
				'baseline_total_cost': baseline_total_cost,
				'optimized_total_cost': optimized_total_cost,
				'total_cost_change': total_cost_change,
				'total_cost_change_pct': total_cost_change_pct,
				'time_change': time_change,
				'time_change_pct': time_change_pct,
			})
			
			# Analyze if the tradeoff makes sense
			if hourly_cost_change > 0 and time_change > 0:
				results['tradeoff_validation'] = "INVALID: Both hourly cost and time increased"
			elif hourly_cost_change < 0 and time_change < 0:
				results['tradeoff_validation'] = "EXCELLENT: Both hourly cost and time decreased"
			elif hourly_cost_change > 0 and time_change < 0:
				# Higher cost but faster - calculate if worth it
				time_improvement_pct = -time_change_pct  # Convert to positive for improvement
				cost_increase_pct = hourly_cost_change_pct
				
				if time_improvement_pct > cost_increase_pct:
					results['tradeoff_validation'] = f"GOOD: {time_improvement_pct:.1f}% time improvement outweighs {cost_increase_pct:.1f}% cost increase"
				else:
					results['tradeoff_validation'] = f"QUESTIONABLE: {cost_increase_pct:.1f}% cost increase for only {time_improvement_pct:.1f}% time improvement"
			elif hourly_cost_change < 0 and time_change > 0:
				# Lower cost but slower - calculate if worth it
				cost_reduction_pct = -hourly_cost_change_pct  # Convert to positive for reduction
				time_degradation_pct = time_change_pct
				
				if cost_reduction_pct > time_degradation_pct:
					results['tradeoff_validation'] = f"GOOD: {cost_reduction_pct:.1f}% cost reduction outweighs {time_degradation_pct:.1f}% time increase"
				else:
					results['tradeoff_validation'] = f"QUESTIONABLE: {cost_reduction_pct:.1f}% cost reduction with {time_degradation_pct:.1f}% time degradation"
			
		return results
	
	def validate_gpu_batch_relationship(self, gpu_count1, batch_size1, gpu_count2, batch_size2):
		"""
		Validate if the batch size changes make sense with GPU count changes.
		
		In general, batch size should scale roughly with GPU count. If GPUs decrease,
		batch size typically decreases and vice versa.
		"""
		# Calculate per-GPU batch size
		if gpu_count1 > 0:
			per_gpu_batch1 = batch_size1 / gpu_count1
		else:
			per_gpu_batch1 = batch_size1
			
		if gpu_count2 > 0:
			per_gpu_batch2 = batch_size2 / gpu_count2
		else:
			per_gpu_batch2 = batch_size2
			
		# Check relationships
		results = {
			'baseline_gpu_count': gpu_count1,
			'baseline_batch_size': batch_size1,
			'baseline_per_gpu_batch': per_gpu_batch1,
			'optimized_gpu_count': gpu_count2,
			'optimized_batch_size': batch_size2,
			'optimized_per_gpu_batch': per_gpu_batch2,
		}
		
		gpu_change = gpu_count2 - gpu_count1
		batch_change = batch_size2 - batch_size1
		per_gpu_batch_change = per_gpu_batch2 - per_gpu_batch1
		
		# Check for sensible scaling
		if gpu_change > 0 and batch_change <= 0:
			results['batch_validation'] = "WARNING: GPUs increased but batch size didn't increase"
		elif gpu_change < 0 and batch_change > 0:
			results['batch_validation'] = "WARNING: GPUs decreased but batch size increased"
		elif gpu_change == 0 and abs(batch_change) > batch_size1 * 0.5:
			results['batch_validation'] = f"WARNING: Large batch size change ({batch_change}) with same GPU count"
		else:
			results['batch_validation'] = "OK: Batch size change consistent with GPU count change"
			
		# Check for per-GPU batch size consistency
		if abs(per_gpu_batch_change) > per_gpu_batch1 * 0.3:
			results['per_gpu_batch_validation'] = f"NOTE: Per-GPU batch size changed significantly from {per_gpu_batch1:.1f} to {per_gpu_batch2:.1f}"
		else:
			results['per_gpu_batch_validation'] = "OK: Per-GPU batch size remained relatively consistent"
			
		return results
		
	def diagnose_zero_gpu_utilization(self, metrics, gpu_count):
		"""
		Diagnose possible reasons for zero GPU utilization.
		
		Args:
			metrics: Dict with GPU utilization and other metrics
			gpu_count: Number of GPUs allocated
		
		Returns:
			List of possible issues
		"""
		issues = []
		
		# Check if GPU count is positive but utilization is zero
		if gpu_count > 0:
			# Check for utilization metrics
			gpu_utils = []
			for key, value in metrics.items():
				if key.startswith('gpu_utilization_'):
					gpu_utils.append(value)
			
			if not gpu_utils:
				issues.append("No GPU utilization metrics found in the data")
			elif all(util == 0 for util in gpu_utils):
				issues.append("All GPU utilization values are zero despite GPUs being allocated")
				
				# Check for possible causes
				if 'gpu_memory_used_gb' in metrics and metrics['gpu_memory_used_gb'] == 0:
					issues.append("GPU memory usage is zero - model may not be using GPU at all")
					
				if 'cuda_visible_devices' in metrics and not metrics['cuda_visible_devices']:
					issues.append("CUDA_VISIBLE_DEVICES environment variable is empty or not set")
				
				if 'cpu_percent' in metrics and metrics['cpu_percent'] > 90:
					issues.append("CPU usage is very high (>90%) - workload may be CPU-bound")
			
			# Check consistency between utilization metrics and count
			if len(gpu_utils) != gpu_count:
				issues.append(f"Mismatch between GPU count ({gpu_count}) and number of GPU utilization metrics ({len(gpu_utils)})")
		
		return issues

def main():
	#import argparse, os, #yaml
	parser = argparse.ArgumentParser(description='Cloud Resource Optimization for ML Training')
	parser.add_argument('--training_file', type=str, help='Path to training file(s)', required=True, nargs='+')
	parser.add_argument('--dataset_path', type=str, help='Path to dataset directory', default='./data')
	parser.add_argument('--offline_logs', type=str, help='Path(s) to offline log file(s)', nargs='*')
	parser.add_argument('--config', type=str, help='Path to configuration file')
	parser.add_argument('--baseline_samples', type=int, default=4,
						help='Number of baseline training samples to run')
	parser.add_argument('--baseline_ratio', type=float, default=0.2,
						help='Ratio of total epochs to use for each baseline sample')
	parser.add_argument('--time_max', type=float, help='Maximum training time in seconds')
	parser.add_argument('--cost_max', type=float, help='Maximum cost in dollars')
	parser.add_argument('--throughput_min', type=float, help='Minimum throughput in samples/second')
	parser.add_argument('--preference', type=str, choices=['balanced', 'cost', 'time', 'resource'],
						default='balanced', help='Optimization preference')
	args = parser.parse_args()

	# Load configuration from file or set defaults
	config = {}
	if args.config and os.path.exists(args.config):
		with open(args.config, 'r') as f:
			config = yaml.safe_load(f)
	else:
		config = {
			'resource': ResourceConfig().__dict__,
			'sla': SLAConfig().__dict__,
			'rl': RLConfig().__dict__,
			'multi_objective': MultiObjectiveConfig().__dict__
		}

	# Override SLA config with command line arguments if provided
	sla_config = config.get('sla', {})
	if args.time_max:
		sla_config['time_max'] = args.time_max
	if args.cost_max:
		sla_config['cost_max'] = args.cost_max
	if args.throughput_min:
		sla_config['throughput_min'] = args.throughput_min
	config['sla'] = sla_config

	# Create optimizer and set user preference.
	optimizer = CloudResourceOptimizer(config_dict=config)
	optimizer.set_user_preference(args.preference)

	# Run optimization for each training file.
	results = []
	for training_file in args.training_file:
		logger.info(f"Optimizing training file: {training_file}")
		result = optimizer.optimize_training(
			training_file_path=training_file,
			dataset_path=args.dataset_path,
			offline_logs=args.offline_logs,
			num_baseline_samples=args.baseline_samples,
			baseline_sample_ratio=args.baseline_ratio
		)

		output_file = f"optimization_report_{os.path.basename(training_file)}.txt"
		report = optimizer.generate_report(result, output_file)
		print(report)
		results.append(result)

	return results

if __name__ == "__main__":
	main()
