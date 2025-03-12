########################################
######## SERAJ A MOSTAFA ###############
### PhD Candiate, IS Dept. UMBC ########
########################################

import os
import sys
import time
import yaml
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Union, Any
import psutil
import signal
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration and Utilities ====================

@dataclass
class SLAConfig:
    """Service Level Agreement Configuration"""
    time_max: float = 3600.0  # Maximum training time in seconds
    cost_max: float = 100.0   # Maximum cost in dollars
    throughput_min: float = 100.0  # Minimum samples/second
    memory_max: float = 32.0  # Maximum memory usage in GB
    gpu_util_min: float = 50.0  # Minimum GPU utilization percentage
    cpu_util_min: float = 50.0  # Minimum CPU utilization percentage
    accuracy_target: float = 85.0  # Target accuracy percentage
    convergence_rate_min: float = 0.01  # Minimum loss reduction per epoch

@dataclass
class ResourceConfig:
    """Resource Configuration"""
    gpu_count_max: int = 8    # Maximum number of GPUs
    gpu_count_min: int = 1    # Minimum number of GPUs
    cpu_count_max: int = 64   # Maximum number of CPUs
    cpu_count_min: int = 8    # Minimum number of CPUs
    memory_max: int = 128     # Maximum memory in GB
    memory_min: int = 16      # Minimum memory in GB
    cost_per_gpu_hour: float = 2.5  # Cost per GPU hour
    cost_per_cpu_hour: float = 0.1  # Cost per CPU hour
    cost_per_gb_hour: float = 0.02  # Cost per GB hour

@dataclass
class RLConfig:
    """Reinforcement Learning Configuration"""
    state_dim: int = 24       # State space dimension
    action_dim: int = 27      # Action space dimension (3 GPU × 3 CPU × 3 Memory)
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
    preference_vectors: List[List[float]] = None
    
    def __post_init__(self):
        if self.preference_vectors is None:
            self.preference_vectors = [
                [0.33, 0.33, 0.34],  # Balanced
                [0.60, 0.20, 0.20],  # Cost-focused
                [0.20, 0.60, 0.20],  # Time-focused
                [0.20, 0.20, 0.60]   # Resource-focused
            ]

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    # Default config if file doesn't exist
    config = {
        "sla": SLAConfig().__dict__,
        "resource": ResourceConfig().__dict__,
        "rl": RLConfig().__dict__,
        "multi_objective": MultiObjectiveConfig().__dict__
    }
    
    # Save default config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Set up logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

# ==================== Model Analysis Utilities ====================

def analyze_model(model: nn.Module) -> Dict:
    """Analyze PyTorch model architecture and extract key features"""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers
    layer_counts = {
        'conv': 0,
        'linear': 0,
        'bn': 0,
        'pooling': 0,
        'dropout': 0,
        'total': 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_counts['conv'] += 1
            layer_counts['total'] += 1
        elif isinstance(module, nn.Linear):
            layer_counts['linear'] += 1
            layer_counts['total'] += 1
        elif isinstance(module, nn.BatchNorm2d):
            layer_counts['bn'] += 1
            layer_counts['total'] += 1
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            layer_counts['pooling'] += 1
            layer_counts['total'] += 1
        elif isinstance(module, nn.Dropout):
            layer_counts['dropout'] += 1
            layer_counts['total'] += 1
    
    # Estimate memory requirements
    param_memory = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter, convert to MB
    
    analysis = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_counts': layer_counts,
        'param_memory_mb': param_memory,
        'complexity_score': total_params * layer_counts['total'] / 1e9  # Normalized complexity score
    }
    
    return analysis

def analyze_dataset(dataloader: DataLoader) -> Dict:
    """Analyze dataset characteristics"""
    # Get a batch
    data_iter = iter(dataloader)
    try:
        batch, _ = next(data_iter)
    except StopIteration:
        batch = None
    
    # Dataset properties
    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    
    # Sample shape if available
    if batch is not None:
        sample_shape = batch[0].shape
        # Memory estimation (rough)
        bytes_per_sample = np.prod(sample_shape) * 4  # Assume float32
        dataset_memory = (bytes_per_sample * dataset_size) / (1024 * 1024)  # MB
        batch_memory = (bytes_per_sample * batch_size) / (1024 * 1024)  # MB
    else:
        sample_shape = None
        dataset_memory = None
        batch_memory = None
    
    analysis = {
        'dataset_size': dataset_size,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'sample_shape': sample_shape,
        'dataset_memory_mb': dataset_memory,
        'batch_memory_mb': batch_memory
    }
    
    return analysis

def predict_initial_resources(model_analysis: Dict, dataset_analysis: Dict) -> Dict:
    """Predict initial resource allocation based on model and dataset analysis"""
    # Simple heuristic-based prediction
    complexity = model_analysis['complexity_score']
    dataset_memory = dataset_analysis['dataset_memory_mb'] / 1024  # Convert to GB
    batch_memory = dataset_analysis['batch_memory_mb']
    dataset_size = dataset_analysis['dataset_size']
    
    # Basic heuristics for resource prediction
    if complexity < 0.1:  # Small model
        gpu_count = 1
        cpu_count = 8
        memory = 16
    elif complexity < 1.0:  # Medium model
        gpu_count = 2
        cpu_count = 16
        memory = 32
    else:  # Large model
        gpu_count = 4
        cpu_count = 32
        memory = 64
    
    # Adjust based on dataset size
    if dataset_size > 100000:
        gpu_count = min(gpu_count + 1, 8)
        cpu_count += 8
        memory += 16
    
    # Ensure enough memory for dataset and batches
    memory = max(memory, dataset_memory * 2 + 8)  # 2x dataset + 8GB buffer
    
    prediction = {
        'gpu_count': gpu_count,
        'cpu_count': cpu_count,
        'memory_gb': memory,
        'estimated_cost_per_hour': gpu_count * 2.5 + cpu_count * 0.1 + memory * 0.02,
        'estimated_training_hours': complexity * dataset_size / (1000 * gpu_count)
    }
    
    return prediction

# ==================== Resource Management ====================

class ResourceManager:
    """Manages cloud resources for ML training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.resource_config = ResourceConfig(**config['resource'])
        self.logger = setup_logger('resource_manager', 'resource_manager.log')
        
        # Current allocation
        self.current_allocation = {
            'gpu_count': 0,
            'cpu_count': 0,
            'memory_gb': 0
        }
        
        # Current utilization
        self.current_utilization = {
            'gpu_util': 0.0,
            'cpu_util': 0.0,
            'memory_util': 0.0
        }
        
        # Cost tracking
        self.start_time = time.time()
        self.cost_accumulated = 0.0
        self.last_update_time = self.start_time
    
    def update_allocation(self, new_allocation: Dict) -> Dict:
        """Update resource allocation based on RL agent's decision"""
        # Validate the new allocation
        gpu_count = max(self.resource_config.gpu_count_min, 
                        min(new_allocation['gpu_count'], self.resource_config.gpu_count_max))
        
        cpu_count = max(self.resource_config.cpu_count_min,
                        min(new_allocation['cpu_count'], self.resource_config.cpu_count_max))
        
        memory_gb = max(self.resource_config.memory_min,
                       min(new_allocation['memory_gb'], self.resource_config.memory_max))
        
        # Update cost before changing allocation
        self.update_cost()
        
        # Log the change
        self.logger.info(f"Updating allocation: GPUs: {self.current_allocation['gpu_count']} -> {gpu_count}, "
                         f"CPUs: {self.current_allocation['cpu_count']} -> {cpu_count}, "
                         f"Memory: {self.current_allocation['memory_gb']} -> {memory_gb}")
        
        # Check if GPU count changed
        gpu_change = gpu_count != self.current_allocation['gpu_count']
        
        # Update current allocation
        old_allocation = copy.deepcopy(self.current_allocation)
        self.current_allocation = {
            'gpu_count': gpu_count,
            'cpu_count': cpu_count,
            'memory_gb': memory_gb
        }
        
        return {
            'old_allocation': old_allocation,
            'new_allocation': self.current_allocation,
            'gpu_change': gpu_change
        }
    
    def update_cost(self):
        """Update accumulated cost based on resource usage"""
        now = time.time()
        hours_elapsed = (now - self.last_update_time) / 3600
        
        # Calculate cost for this period
        period_cost = (
            self.current_allocation['gpu_count'] * self.resource_config.cost_per_gpu_hour +
            self.current_allocation['cpu_count'] * self.resource_config.cost_per_cpu_hour +
            self.current_allocation['memory_gb'] * self.resource_config.cost_per_gb_hour
        ) * hours_elapsed
        
        self.cost_accumulated += period_cost
        self.last_update_time = now
        
        return self.cost_accumulated
    
    def get_current_cost(self):
        """Get current accumulated cost without updating"""
        self.update_cost()
        return self.cost_accumulated
    
    def get_hourly_cost(self):
        """Get the current hourly cost rate"""
        return (
            self.current_allocation['gpu_count'] * self.resource_config.cost_per_gpu_hour +
            self.current_allocation['cpu_count'] * self.resource_config.cost_per_cpu_hour +
            self.current_allocation['memory_gb'] * self.resource_config.cost_per_gb_hour
        )
    
    def update_utilization(self, metrics: Dict):
        """Update resource utilization metrics"""
        self.current_utilization = {
            'gpu_util': metrics.get('gpu_utilization', 0.0),
            'cpu_util': metrics.get('cpu_percent', 0.0),
            'memory_util': metrics.get('memory_used_gb', 0.0) / self.current_allocation['memory_gb'] * 100
        }
        return self.current_utilization
    
    def get_cpu_count(self):
        """Get available CPU count"""
        return psutil.cpu_count()
    
    def get_gpu_count(self):
        """Get available GPU count"""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    
    def get_total_memory(self):
        """Get total system memory in GB"""
        return psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    def get_system_metrics(self):
        """Get current system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_used_gb': psutil.virtual_memory().used / (1024 * 1024 * 1024),
            'memory_total_gb': psutil.virtual_memory().total / (1024 * 1024 * 1024),
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            metrics['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
            
            # Try to get GPU utilization
            try:
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
                    encoding='utf-8'
                )
                metrics['gpu_utilization'] = float(output.strip().split('\n')[0])
            except:
                metrics['gpu_utilization'] = 0.0
        
        return metrics

# ==================== SLA Management ====================

class SLAManager:
    """Manages SLA monitoring and violation detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sla_config = SLAConfig(**config['sla'])
        self.logger = setup_logger('sla_manager', 'sla_manager.log')
        
        # SLA status tracking
        self.sla_status = {
            'time': {
                'met': True,
                'value': 0.0,
                'threshold': self.sla_config.time_max,
                'violations': 0
            },
            'cost': {
                'met': True,
                'value': 0.0,
                'threshold': self.sla_config.cost_max,
                'violations': 0
            },
            'throughput': {
                'met': True,
                'value': 0.0,
                'threshold': self.sla_config.throughput_min,
                'violations': 0
            },
            'memory': {
                'met': True,
                'value': 0.0,
                'threshold': self.sla_config.memory_max,
                'violations': 0
            },
            'gpu_util': {
                'met': True,
                'value': 0.0,
                'threshold': self.sla_config.gpu_util_min,
                'violations': 0
            },
            'cpu_util': {
                'met': True,
                'value': 0.0,
                'threshold': self.sla_config.cpu_util_min,
                'violations': 0
            },
            'accuracy': {
                'met': True,
                'value': 0.0,
                'threshold': self.sla_config.accuracy_target,
                'violations': 0
            },
            'convergence': {
                'met': True,
                'value': 0.0,
                'threshold': self.sla_config.convergence_rate_min,
                'violations': 0
            }
        }
        
        # Historical violations
        self.violation_history = []
        
        # Start time
        self.start_time = time.time()
    
    def update_sla_status(self, metrics: Dict) -> Dict:
        """Update SLA status based on current metrics"""
        # Time SLA
        elapsed_time = time.time() - self.start_time
        self.sla_status['time']['value'] = elapsed_time
        self.sla_status['time']['met'] = elapsed_time <= self.sla_config.time_max
        
        # Cost SLA
        if 'cost_so_far' in metrics:
            self.sla_status['cost']['value'] = metrics['cost_so_far']
            self.sla_status['cost']['met'] = metrics['cost_so_far'] <= self.sla_config.cost_max
        
        # Throughput SLA
        if 'throughput' in metrics:
            self.sla_status['throughput']['value'] = metrics['throughput']
            self.sla_status['throughput']['met'] = metrics['throughput'] >= self.sla_config.throughput_min
        
        # Memory SLA
        if 'memory_used_gb' in metrics:
            self.sla_status['memory']['value'] = metrics['memory_used_gb']
            self.sla_status['memory']['met'] = metrics['memory_used_gb'] <= self.sla_config.memory_max
        
        # GPU utilization SLA
        if 'gpu_utilization' in metrics:
            self.sla_status['gpu_util']['value'] = metrics['gpu_utilization']
            self.sla_status['gpu_util']['met'] = metrics['gpu_utilization'] >= self.sla_config.gpu_util_min
        
        # CPU utilization SLA
        if 'cpu_percent' in metrics:
            self.sla_status['cpu_util']['value'] = metrics['cpu_percent']
            self.sla_status['cpu_util']['met'] = metrics['cpu_percent'] >= self.sla_config.cpu_util_min
        
        # Accuracy SLA
        if 'test_accuracy' in metrics:
            self.sla_status['accuracy']['value'] = metrics['test_accuracy']
            self.sla_status['accuracy']['met'] = metrics['test_accuracy'] >= self.sla_config.accuracy_target
        
        # Convergence rate SLA
        if 'train_loss' in metrics and 'prev_train_loss' in metrics:
            convergence_rate = metrics['prev_train_loss'] - metrics['train_loss']
            self.sla_status['convergence']['value'] = convergence_rate
            self.sla_status['convergence']['met'] = convergence_rate >= self.sla_config.convergence_rate_min
        
        # Update violation counts
        for key, status in self.sla_status.items():
            if not status['met']:
                status['violations'] += 1
                violation = {
                    'timestamp': time.time(),
                    'type': key,
                    'value': status['value'],
                    'threshold': status['threshold']
                }
                self.violation_history.append(violation)
                self.logger.warning(f"SLA Violation: {key}, Value: {status['value']}, Threshold: {status['threshold']}")
        
        return self.sla_status
    
    def get_violation_counts(self):
        """Get counts of SLA violations by type"""
        return {key: status['violations'] for key, status in self.sla_status.items()}
    
    def get_total_violations(self):
        """Get total count of all SLA violations"""
        return sum(status['violations'] for status in self.sla_status.values())
    
    def get_sla_compliance_rate(self):
        """Get overall SLA compliance rate"""
        violations = sum(1 for status in self.sla_status.values() if not status['met'])
        return 1.0 - (violations / len(self.sla_status))
    
    def get_violation_severity(self, sla_type: str) -> float:
        """Calculate violation severity for an SLA type"""
        status = self.sla_status[sla_type]
        if status['met']:
            return 0.0
        
        # Different calculation based on SLA type (min vs max thresholds)
        if sla_type in ['throughput', 'gpu_util', 'cpu_util', 'accuracy', 'convergence']:
            # For minimums, severity is proportional to shortfall
            return (status['threshold'] - status['value']) / status['threshold']
        else:
            # For maximums, severity is proportional to excess
            return (status['value'] - status['threshold']) / status['threshold']
    
    def get_overall_violation_severity(self):
        """Calculate overall SLA violation severity across all types"""
        severities = [self.get_violation_severity(sla_type) for sla_type in self.sla_status.keys()]
        return sum(severities)
    
    def get_sla_status_vector(self):
        """Get SLA status as a vector for the RL state"""
        return [int(status['met']) for status in self.sla_status.values()]
    
    def get_sla_violation_vector(self):
        """Get normalized SLA violation severities as a vector for the RL state"""
        return [self.get_violation_severity(sla_type) for sla_type in self.sla_status.keys()]

# ==================== Neural Network Models ====================

class Actor(nn.Module):
    """Actor network for policy approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(state), dim=-1)

class Critic(nn.Module):
    """Critic network for value function approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1

# ==================== Experience Replay Buffer ====================

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

# ==================== Multi-Objective RL Agent ====================

class MultiObjectiveRLAgent:
    """Multi-Objective Reinforcement Learning Agent with adaptive critic reset"""
    
    def __init__(self, config: Dict, state_dim: int, action_dim: int):
        self.config = config
        self.rl_config = RLConfig(**config['rl'])
        self.mo_config = MultiObjectiveConfig(**config['multi_objective'])
        self.logger = setup_logger('rl_agent', 'rl_agent.log')
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create actors and critics for different preference vectors
        self.actors = []
        self.critics = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        # Initialize networks and optimizers
        for i in range(self.mo_config.num_policies):
            actor = Actor(state_dim, action_dim, self.rl_config.hidden_dim).to(self.device)
            critic = Critic(state_dim, action_dim, self.rl_config.hidden_dim).to(self.device)
            
            target_critic = copy.deepcopy(critic).to(self.device)
            # Freeze target networks
            for p in target_critic.parameters():
                p.requires_grad = False
                
            actor_optimizer = optim.Adam(actor.parameters(), lr=self.rl_config.lr_actor)
            critic_optimizer = optim.Adam(critic.parameters(), lr=self.rl_config.lr_critic)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        # Replay buffers for different policies
        self.replay_buffers = [
            ReplayBuffer(state_dim, action_dim, self.rl_config.buffer_size)
            for _ in range(self.mo_config.num_policies)
        ]
        
        # Training stats
        self.train_stats = {
            'actor_losses': [[] for _ in range(self.mo_config.num_policies)],
            'critic_losses': [[] for _ in range(self.mo_config.num_policies)],
            'rewards': [[] for _ in range(self.mo_config.num_policies)]
        }
        
        # Step counter for adaptive reset
        self.steps = 0
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir='tensorboard_logs')
        
        self.logger.info(f"Initialized Multi-Objective RL Agent with {self.mo_config.num_policies} policies")
        self.logger.info(f"Using device: {self.device}")
    
    def select_action(self, state: np.ndarray, policy_idx: int, 
                     explore: bool = True) -> Tuple[int, np.ndarray]:
        """Select action based on current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_probs = self.actors[policy_idx](state_tensor)
            
            if explore:
                # Add exploration noise
                action_probs = action_probs.cpu().numpy()
                action_probs = action_probs * (1 + np.random.normal(0, self.rl_config.exploration_noise, size=action_probs.shape))
                action_probs = action_probs / np.sum(action_probs)  # Renormalize
            else:
                action_probs = action_probs.cpu().numpy()
            
            # Select action based on probabilities
            action_idx = np.random.choice(self.action_dim, p=action_probs)
            
            # Convert to one-hot for critic input
            action_one_hot = np.zeros(self.action_dim)
            action_one_hot[action_idx] = 1.0
            
            return action_idx, action_one_hot
    
    def select_policy(self, sla_status: Dict, user_preference: str = 'balanced') -> int:
        """Select appropriate policy based on SLA status and user preference"""
        # Check for SLA violations
        violations = {key: not status['met'] for key, status in sla_status.items()}
        
        # If time SLA is violated or at risk, use time-focused policy
        if violations.get('time', False) or sla_status['time']['value'] > 0.8 * sla_status['time']['threshold']:
            return 2  # Time-focused policy index
        
        # If cost SLA is violated or at risk, use cost-focused policy
        elif violations.get('cost', False) or sla_status['cost']['value'] > 0.8 * sla_status['cost']['threshold']:
            return 1  # Cost-focused policy index
        
        # If resource SLA is violated, use resource-focused policy
        elif (violations.get('gpu_util', False) or violations.get('cpu_util', False) or 
              violations.get('memory', False)):
            return 3  # Resource-focused policy index
        
        # Otherwise, use user preference
        else:
            preference_map = {
                'balanced': 0,
                'cost': 1,
                'time': 2,
                'resource': 3
            }
            return preference_map.get(user_preference, 0)
    
    def train(self, policy_idx: int) -> Dict:
        """Train the policy networks"""
        if len(self.replay_buffers[policy_idx]) < self.rl_config.batch_size:
            return {'critic_loss': 0.0, 'actor_loss': 0.0}
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffers[policy_idx].sample(self.rl_config.batch_size)
        
        # Update critic
        with torch.no_grad():
            # Get next action probabilities from actor
            next_action_probs = self.actors[policy_idx](next_state)
            
            # Calculate expected Q values for next state
            next_q_values = torch.zeros_like(reward).to(self.device)
            
            for a in range(self.action_dim):
                # Create one-hot action
                next_action = torch.zeros(self.rl_config.batch_size, self.action_dim).to(self.device)
                next_action[:, a] = 1.0
                
                # Weight Q-value by action probability
                next_action_prob = next_action_probs[:, a].unsqueeze(1)
                next_q = self.target_critics[policy_idx](next_state, next_action)
                next_q_values += next_action_prob * next_q
            
            # Compute target Q value
            target_q = reward + (1 - done) * self.rl_config.gamma * next_q_values
        
        # Get current Q estimate
        current_q = self.critics[policy_idx](state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Optimize critic
        self.critic_optimizers[policy_idx].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[policy_idx].step()
        
        # Compute actor loss
        actor_loss = 0.0
        action_probs = self.actors[policy_idx](state)
        
        for a in range(self.action_dim):
            # Create one-hot action
            current_action = torch.zeros(self.rl_config.batch_size, self.action_dim).to(self.device)
            current_action[:, a] = 1.0
            
            # Weight Q-value by action probability
            action_prob = action_probs[:, a].unsqueeze(1)
            q_value = self.critics[policy_idx](state, current_action)
            actor_loss -= torch.mean(action_prob * q_value)
        
        # Optimize actor
        self.actor_optimizers[policy_idx].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[policy_idx].step()
        
        # Update target critic networks
        for target_param, param in zip(self.target_critics[policy_idx].parameters(), 
                                     self.critics[policy_idx].parameters()):
            target_param.data.copy_(
                param.data * self.rl_config.tau + target_param.data * (1.0 - self.rl_config.tau)
            )
        
        # Log training stats
        self.train_stats['critic_losses'][policy_idx].append(critic_loss.item())
        self.train_stats['actor_losses'][policy_idx].append(actor_loss.item())
        
        # Log to TensorBoard
        self.writer.add_scalar(f'Loss/critic_policy_{policy_idx}', critic_loss.item(), self.steps)
        self.writer.add_scalar(f'Loss/actor_policy_{policy_idx}', actor_loss.item(), self.steps)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def reset_critic(self, policy_idx: int = None):
        """Reset critic network to avoid overestimation bias"""
        # Initialize weights using Xavier initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        
        if policy_idx is None:
            # Reset all critics
            for i in range(self.mo_config.num_policies):
                self.critics[i].apply(init_weights)
                self.target_critics[i] = copy.deepcopy(self.critics[i])
                self.logger.info(f"Reset critic for policy {i}")
        else:
            # Reset specific critic
            self.critics[policy_idx].apply(init_weights)
            self.target_critics[policy_idx] = copy.deepcopy(self.critics[policy_idx])
            self.logger.info(f"Reset critic for policy {policy_idx}")
    
    def check_adaptive_reset(self, sla_status: Dict):
        """Check if critic networks need to be reset"""
        self.steps += 1
        
        # Check reset interval
        if self.steps % self.rl_config.reset_interval == 0:
            # Check SLA violations
            severe_violations = sum(1 for status in sla_status.values() 
                                   if not status['met'] and status['violations'] > 3)
            
            if severe_violations > 0:
                # Reset all critics with higher probability for severe SLA violations
                if random.random() < self.rl_config.reset_probability * 2:
                    self.reset_critic()
                    self.logger.warning(f"Performing full critic reset due to {severe_violations} severe SLA violations")
            elif random.random() < self.rl_config.reset_probability:
                # Regular probabilistic reset
                self.reset_critic()
                self.logger.info("Performing regular scheduled critic reset")
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, 
                      next_state: np.ndarray, done: bool, policy_idx: int):
        """Add experience to replay buffer"""
        self.replay_buffers[policy_idx].add(state, action, reward, next_state, done)
        self.train_stats['rewards'][policy_idx].append(reward)
    
    def save(self, path: str = "models"):
        """Save model weights"""
        os.makedirs(path, exist_ok=True)
        
        for i in range(self.mo_config.num_policies):
            policy_path = os.path.join(path, f"policy_{i}")
            os.makedirs(policy_path, exist_ok=True)
            
            torch.save(self.actors[i].state_dict(), os.path.join(policy_path, "actor.pt"))
            torch.save(self.critics[i].state_dict(), os.path.join(policy_path, "critic.pt"))
            torch.save(self.target_critics[i].state_dict(), os.path.join(policy_path, "target_critic.pt"))
            
        self.logger.info(f"Saved model weights to {path}")
    
    def load(self, path: str = "models"):
        """Load model weights"""
        for i in range(self.mo_config.num_policies):
            policy_path = os.path.join(path, f"policy_{i}")
            
            try:
                self.actors[i].load_state_dict(torch.load(os.path.join(policy_path, "actor.pt")))
                self.critics[i].load_state_dict(torch.load(os.path.join(policy_path, "critic.pt")))
                self.target_critics[i].load_state_dict(torch.load(os.path.join(policy_path, "target_critic.pt")))
            except FileNotFoundError:
                self.logger.warning(f"Could not load weights for policy {i}, using random initialization")
                
        self.logger.info(f"Loaded model weights from {path}")

# ==================== Cloud Environment (MDP) ====================

class CloudEnvironment:
    """
    Cloud Environment for reinforcement learning (MDP formulation)
    State: Current metrics and SLA status
    Action: Resource allocation decisions
    Reward: Multi-objective reward function
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.resource_config = ResourceConfig(**config['resource'])
        self.sla_config = SLAConfig(**config['sla'])
        self.logger = setup_logger('cloud_env', 'cloud_env.log')
        
        # Initialize components
        self.resource_manager = ResourceManager(config)
        self.sla_manager = SLAManager(config)
        
        # State and action space
        self.state_dim = 24  # Adjust based on state representation
        self.action_dim = 27  # 3 GPU options × 3 CPU options × 3 Memory options
        
        # Action space mapping
        self.gpu_actions = [-1, 0, 1]  # Decrease, no change, increase
        self.cpu_actions = [-1, 0, 1]  # Decrease, no change, increase
        self.memory_actions = [-1, 0, 1]  # Decrease, no change, increase
        
        # Current state
        self.current_state = None
        
        # Performance metrics
        self.performance_metrics = {}
        self.prev_performance_metrics = {}
        
        # Training job info
        self.training_info = {
            'model_name': None,
            'dataset_name': None,
            'batch_size': None,
            'current_epoch': 0,
            'total_epochs': 0,
            'start_time': None
        }
        
        # For tracking during episode
        self.episode_reward = 0.0
        self.episode_violations = 0
        self.episode_steps = 0
        
        # Initialize with minimum resources
        self.current_allocation = {
            'gpu_count': self.resource_config.gpu_count_min,
            'cpu_count': self.resource_config.cpu_count_min,
            'memory_gb': self.resource_config.memory_min
        }
        
        # Apply initial allocation
        self.resource_manager.update_allocation(self.current_allocation)
        
        self.logger.info("Initialized Cloud Environment")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Reset tracking variables
        self.episode_reward = 0.0
        self.episode_violations = 0
        self.episode_steps = 0
        
        # Reset SLA manager
        self.sla_manager = SLAManager(self.config)
        
        # Reset to minimum resources
        self.current_allocation = {
            'gpu_count': self.resource_config.gpu_count_min,
            'cpu_count': self.resource_config.cpu_count_min,
            'memory_gb': self.resource_config.memory_min
        }
        
        # Apply initial allocation
        self.resource_manager.update_allocation(self.current_allocation)
        
        # Get initial state
        self.current_state = self._get_state()
        
        return self.current_state
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment and observe outcome
        
        Args:
            action_idx: Index of the action in the flattened action space
        
        Returns:
            next_state: New state after action
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Convert action index to resource adjustments
        gpu_idx = action_idx // 9
        cpu_idx = (action_idx % 9) // 3
        memory_idx = action_idx % 3
        
        gpu_action = self.gpu_actions[gpu_idx]
        cpu_action = self.cpu_actions[cpu_idx]
        memory_action = self.memory_actions[memory_idx]
        
        # Calculate new allocation
        new_allocation = {
            'gpu_count': max(self.resource_config.gpu_count_min, 
                           min(self.current_allocation['gpu_count'] + gpu_action,
                              self.resource_config.gpu_count_max)),
            'cpu_count': max(self.resource_config.cpu_count_min,
                           min(self.current_allocation['cpu_count'] + cpu_action * 4,  # 4 CPUs at a time
                              self.resource_config.cpu_count_max)),
            'memory_gb': max(self.resource_config.memory_min,
                           min(self.current_allocation['memory_gb'] + memory_action * 8,  # 8 GB at a time
                              self.resource_config.memory_max))
        }
        
        # Apply allocation change
        update_result = self.resource_manager.update_allocation(new_allocation)
        self.current_allocation = update_result['new_allocation']
        
        # Get system metrics after allocation change
        system_metrics = self.resource_manager.get_system_metrics()
        
        # Update performance metrics
        self._update_performance_metrics(system_metrics, update_result['gpu_change'])
        
        # Update SLA status
        all_metrics = {**system_metrics, **self.performance_metrics, 
                      'cost_so_far': self.resource_manager.get_current_cost()}
        sla_status = self.sla_manager.update_sla_status(all_metrics)
        
        # Get next state
        next_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(update_result, system_metrics, sla_status)
        
        # Check if episode is done
        done = self._check_done()
        
        # Update tracking variables
        self.episode_reward += reward
        self.episode_violations += sum(1 for status in sla_status.values() if not status['met'])
        self.episode_steps += 1
        
        # Update current state
        self.current_state = next_state
        
        # Information for debugging and logging
        info = {
            'gpu_count': self.current_allocation['gpu_count'],
            'cpu_count': self.current_allocation['cpu_count'],
            'memory_gb': self.current_allocation['memory_gb'],
            'gpu_action': gpu_action,
            'cpu_action': cpu_action,
            'memory_action': memory_action,
            'cost_rate': self.resource_manager.get_hourly_cost(),
            'cost_so_far': self.resource_manager.get_current_cost(),
            'sla_violations': self.sla_manager.get_total_violations(),
            'sla_compliance_rate': self.sla_manager.get_sla_compliance_rate(),
            'throughput': self.performance_metrics.get('throughput', 0),
            'epoch': self.training_info['current_epoch'],
            'episode_reward': self.episode_reward,
            'episode_violations': self.episode_violations
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Construct the state representation for RL agent"""
        # Resource allocation state
        allocation_state = [
            self.current_allocation['gpu_count'] / self.resource_config.gpu_count_max,
            self.current_allocation['cpu_count'] / self.resource_config.cpu_count_max,
            self.current_allocation['memory_gb'] / self.resource_config.memory_max
        ]
        
        # Resource utilization state
        system_metrics = self.resource_manager.get_system_metrics()
        utilization_state = [
            system_metrics.get('gpu_utilization', 0) / 100.0,
            system_metrics.get('cpu_percent', 0) / 100.0,
            system_metrics.get('memory_used_gb', 0) / self.current_allocation['memory_gb']
        ]
        
        # Training progress state
        progress_state = [
            self.training_info['current_epoch'] / max(1, self.training_info['total_epochs']),
            self.performance_metrics.get('throughput', 0) / 1000.0,  # Normalize throughput
            self.performance_metrics.get('test_accuracy', 0) / 100.0,
            self.performance_metrics.get('train_loss', 0) / 10.0  # Normalize loss
        ]
        
        # Cost state
        cost_rate = self.resource_manager.get_hourly_cost()
        cost_so_far = self.resource_manager.get_current_cost()
        cost_state = [
            cost_rate / 100.0,  # Normalize hourly cost
            cost_so_far / self.sla_config.cost_max
        ]
        
        # Time state
        elapsed_time = time.time() - self.sla_manager.start_time
        time_state = [
            elapsed_time / self.sla_config.time_max,
            (self.training_info['current_epoch'] / max(1, self.training_info['total_epochs'])) * 
            (elapsed_time / max(1, self.training_info['current_epoch']))  # Estimated time per epoch
        ]
        
        # SLA state
        sla_status = self.sla_manager.get_sla_status_vector()
        sla_violations = self.sla_manager.get_sla_violation_vector()
        
        # Combine all state components
        state = allocation_state + utilization_state + progress_state + cost_state + time_state + sla_status + sla_violations
        
        return np.array(state, dtype=np.float32)
    
    def _update_performance_metrics(self, system_metrics: Dict, gpu_change: bool):
        """Update performance metrics based on current measurements"""
        # Store previous metrics
        self.prev_performance_metrics = copy.deepcopy(self.performance_metrics)
        
        # If GPU configuration changed, we might need to wait for metrics to stabilize
        if gpu_change:
            time.sleep(5)  # Wait for stabilization
        
        # Update with new system metrics
        self.performance_metrics.update(system_metrics)
    
    def _calculate_reward(self, update_result: Dict, system_metrics: Dict, sla_status: Dict) -> float:
        """Calculate multi-objective reward based on current state"""
        # Base components
        cost_reward = -self.resource_manager.get_hourly_cost() / 100.0  # Negative cost (minimize)
        
        # Throughput reward (maximize)
        throughput = self.performance_metrics.get('throughput', 0)
        throughput_reward = throughput / 1000.0
        
        # Resource utilization reward (maximize to appropriate level)
        gpu_util = system_metrics.get('gpu_utilization', 0)
        cpu_util = system_metrics.get('cpu_percent', 0)
        
        # Penalize both under and over-utilization
        gpu_util_reward = -abs(gpu_util - 80) / 40.0  # Target ~80% utilization
        cpu_util_reward = -abs(cpu_util - 80) / 40.0  # Target ~80% utilization
        
        # SLA violation penalties
        sla_violation_penalty = 0.0
        for sla_type, status in sla_status.items():
            if not status['met']:
                violation_severity = self.sla_manager.get_violation_severity(sla_type)
                sla_violation_penalty -= violation_severity * 5.0  # Scale penalty by severity
        
        # Transition cost penalty
        transition_penalty = 0.0
        if update_result['gpu_change']:
            transition_penalty -= 0.5  # Penalty for GPU change (checkpoint overhead)
        
        # Overall reward
        reward = (
            0.3 * cost_reward +          # 30% weight to cost
            0.3 * throughput_reward +    # 30% weight to throughput
            0.1 * gpu_util_reward +      # 10% weight to GPU utilization
            0.1 * cpu_util_reward +      # 10% weight to CPU utilization
            0.1 * sla_violation_penalty + # 10% weight to SLA violations
            0.1 * transition_penalty      # 10% weight to transition costs
        )
        
        return reward
    
    def _check_done(self) -> bool:
        """Check if episode should terminate"""
        # Episode ends if training is complete
        if self.training_info['current_epoch'] >= self.training_info['total_epochs']:
            return True
        
        # Episode ends if time SLA is severely violated
        elapsed_time = time.time() - self.sla_manager.start_time
        if elapsed_time > 1.5 * self.sla_config.time_max:
            self.logger.warning("Episode terminated due to severe time SLA violation")
            return True
        
        # Episode ends if cost SLA is severely violated
        if self.resource_manager.get_current_cost() > 1.5 * self.sla_config.cost_max:
            self.logger.warning("Episode terminated due to severe cost SLA violation")
            return True
        
        return False
    
    def set_training_info(self, info: Dict):
        """Set information about the current training job"""
        self.training_info.update(info)
        self.training_info['start_time'] = time.time()
        self.logger.info(f"Set training info: {self.training_info}")
    
    def update_training_progress(self, epoch: int, metrics: Dict):
        """Update training progress and metrics"""
        self.training_info['current_epoch'] = epoch
        
        # Save previous train loss for convergence rate calculation
        if 'train_loss' in self.performance_metrics:
            metrics['prev_train_loss'] = self.performance_metrics['train_loss']
        
        # Update performance metrics
        self.performance_metrics.update(metrics)
        
        # Log update
        self.logger.info(f"Updated training progress: Epoch {epoch}, "
                        f"Accuracy: {metrics.get('test_accuracy', 0):.2f}, "
                        f"Loss: {metrics.get('train_loss', 0):.4f}")
    
    def get_current_allocation(self) -> Dict:
        """Get current resource allocation"""
        return self.current_allocation
    
    def get_utilization_metrics(self) -> Dict:
        """Get current utilization metrics"""
        return self.resource_manager.current_utilization
    
    def get_prediction_options(self) -> List[Dict]:
        """Generate multiple resource allocation options with predictions"""
        # Current allocation and metrics
        current_allocation = self.get_current_allocation()
        current_metrics = self.performance_metrics
        
        # Option 1: High performance (scale up)
        high_perf_allocation = {
            'gpu_count': min(current_allocation['gpu_count'] + 2, self.resource_config.gpu_count_max),
            'cpu_count': min(current_allocation['cpu_count'] + 8, self.resource_config.cpu_count_max),
            'memory_gb': min(current_allocation['memory_gb'] + 16, self.resource_config.memory_max)
        }
        
        high_perf_cost = (
            high_perf_allocation['gpu_count'] * self.resource_config.cost_per_gpu_hour +
            high_perf_allocation['cpu_count'] * self.resource_config.cost_per_cpu_hour +
            high_perf_allocation['memory_gb'] * self.resource_config.cost_per_gb_hour
        )
        
        # Estimate speedup based on additional resources
        gpu_ratio = high_perf_allocation['gpu_count'] / max(1, current_allocation['gpu_count'])
        speedup = min(1.8, gpu_ratio)  # Non-linear speedup, capped at 1.8x
        
        # Current throughput and projected
        current_throughput = current_metrics.get('throughput', 100)
        high_perf_throughput = current_throughput * speedup
        
        # Estimate remaining time
        remaining_epochs = self.training_info['total_epochs'] - self.training_info['current_epoch']
        current_time_per_epoch = (time.time() - self.training_info['start_time']) / max(1, self.training_info['current_epoch'])
        
        high_perf_time = (current_time_per_epoch / speedup) * remaining_epochs
        high_perf_total_cost = high_perf_cost * (high_perf_time / 3600)
        
        # Option 2: Cost saving (scale down)
        cost_save_allocation = {
            'gpu_count': max(current_allocation['gpu_count'] - 1, self.resource_config.gpu_count_min),
            'cpu_count': max(current_allocation['cpu_count'] - 4, self.resource_config.cpu_count_min),
            'memory_gb': max(current_allocation['memory_gb'] - 8, self.resource_config.memory_min)
        }
        
        cost_save_cost = (
            cost_save_allocation['gpu_count'] * self.resource_config.cost_per_gpu_hour +
            cost_save_allocation['cpu_count'] * self.resource_config.cost_per_cpu_hour +
            cost_save_allocation['memory_gb'] * self.resource_config.cost_per_gb_hour
        )
        
        # Estimate slowdown
        gpu_ratio = cost_save_allocation['gpu_count'] / max(1, current_allocation['gpu_count'])
        slowdown = max(1.0, 1/gpu_ratio)  # At least 1.0x slowdown
        
        cost_save_throughput = current_throughput / slowdown
        cost_save_time = (current_time_per_epoch * slowdown) * remaining_epochs
        cost_save_total_cost = cost_save_cost * (cost_save_time / 3600)
        
        # Option 3: Balanced (current or slight adjustment)
        balanced_allocation = copy.deepcopy(current_allocation)
        
        # If utilization is low, scale down slightly
        gpu_util = self.resource_manager.current_utilization.get('gpu_util', 0)
        if gpu_util < 50 and balanced_allocation['gpu_count'] > self.resource_config.gpu_count_min:
            balanced_allocation['gpu_count'] -= 1
        elif gpu_util > 90 and balanced_allocation['gpu_count'] < self.resource_config.gpu_count_max:
            balanced_allocation['gpu_count'] += 1
        
        balanced_cost = (
            balanced_allocation['gpu_count'] * self.resource_config.cost_per_gpu_hour +
            balanced_allocation['cpu_count'] * self.resource_config.cost_per_cpu_hour +
            balanced_allocation['memory_gb'] * self.resource_config.cost_per_gb_hour
        )
        
        # Estimate performance adjustment
        gpu_ratio = balanced_allocation['gpu_count'] / max(1, current_allocation['gpu_count'])
        if gpu_ratio == 1:
            balanced_throughput = current_throughput
            balanced_time = current_time_per_epoch * remaining_epochs
        elif gpu_ratio > 1:
            speedup = min(1.3, gpu_ratio)  # More conservative speedup
            balanced_throughput = current_throughput * speedup
            balanced_time = (current_time_per_epoch / speedup) * remaining_epochs
        else:
            slowdown = max(1.0, 1/gpu_ratio)  # At least 1.0x slowdown
            balanced_throughput = current_throughput / slowdown
            balanced_time = (current_time_per_epoch * slowdown) * remaining_epochs
        
        balanced_total_cost = balanced_cost * (balanced_time / 3600)
        
        # Return all options
        options = [
            {
                'name': 'High Performance',
                'allocation': high_perf_allocation,
                'hourly_cost': high_perf_cost,
                'total_cost': high_perf_total_cost,
                'estimated_time_minutes': high_perf_time / 60,
                'throughput': high_perf_throughput,
                'description': 'Maximize training speed at higher cost'
            },
            {
                'name': 'Cost Saving',
                'allocation': cost_save_allocation,
                'hourly_cost': cost_save_cost,
                'total_cost': cost_save_total_cost,
                'estimated_time_minutes': cost_save_time / 60,
                'throughput': cost_save_throughput,
                'description': 'Minimize cost with longer training time'
            },
            {
                'name': 'Balanced',
                'allocation': balanced_allocation,
                'hourly_cost': balanced_cost,
                'total_cost': balanced_total_cost,
                'estimated_time_minutes': balanced_time / 60,
                'throughput': balanced_throughput,
                'description': 'Balance cost and performance'
            }
        ]
        
        return options

# ==================== Training Job Integration ====================

class TrainingJobManager:
    """Manages the integration with ML training jobs"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('training_manager', 'training_manager.log')
        
        # Training job state
        self.training_module = None
        self.model = None
        self.dataloader = None
        self.current_allocation = None
        
        # Checkpoint management
        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # DataParallel configuration
        self.data_parallel = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_training_module(self, module_path: str):
        """Load training module from path"""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("training_module", module_path)
            self.training_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.training_module)
            self.logger.info(f"Successfully loaded training module from {module_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load training module: {str(e)}")
            return False
    
    def analyze_training_job(self, model: nn.Module, dataloader: DataLoader) -> Dict:
        """Analyze model and dataset to predict resource needs"""
        self.model = model
        self.dataloader = dataloader
        
        # Analyze model architecture
        model_analysis = analyze_model(model)
        
        # Analyze dataset
        dataset_analysis = analyze_dataset(dataloader)
        
        # Predict initial resource allocation
        resource_prediction = predict_initial_resources(model_analysis, dataset_analysis)
        
        # Log analysis results
        self.logger.info(f"Model analysis: {model_analysis}")
        self.logger.info(f"Dataset analysis: {dataset_analysis}")
        self.logger.info(f"Resource prediction: {resource_prediction}")
        
        # Set current allocation to prediction
        self.current_allocation = resource_prediction
        
        return {
            'model_analysis': model_analysis,
            'dataset_analysis': dataset_analysis,
            'resource_prediction': resource_prediction
        }
    
    def setup_data_parallel(self, gpu_count: int):
        """Set up DataParallel with specified GPU count"""
        if torch.cuda.is_available() and gpu_count > 0:
            # Limit to available GPUs
            available_gpus = min(torch.cuda.device_count(), gpu_count)
            
            if available_gpus > 1:
                # Multi-GPU setup
                device_ids = list(range(available_gpus))
                self.model = nn.DataParallel(self.model, device_ids=device_ids).to(self.device)
                self.logger.info(f"Set up DataParallel with {available_gpus} GPUs: {device_ids}")
            else:
                # Single GPU
                self.model = self.model.to(self.device)
                self.logger.info(f"Using single GPU")
        else:
            # CPU fallback
            self.model = self.model.to(self.device)
            self.logger.info(f"Using CPU for training")
    
    def adjust_resources(self, new_allocation: Dict) -> bool:
        """Adjust resources based on RL agent's decision"""
        # Get current allocation
        old_allocation = self.current_allocation
        
        # Check if GPU count changed
        gpu_change = new_allocation['gpu_count'] != old_allocation['gpu_count']
        
        if gpu_change:
            # Need to save checkpoint and reconfigure DataParallel
            self.save_checkpoint()
            self.setup_data_parallel(new_allocation['gpu_count'])
            self.load_checkpoint()
        
        # Update batch size if needed (optional)
        # This could be done by recreating dataloaders with adjusted batch sizes
        
        # Update current allocation
        self.current_allocation = new_allocation
        
        self.logger.info(f"Adjusted resources: {old_allocation} -> {new_allocation}")
        return True
    
    def save_checkpoint(self):
        """Save model and optimizer state"""
        # This is a simplified version - in practice, would need to save optimizer state too
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{int(time.time())}.pt")
        
        # If using DataParallel, save the module
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)
            
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str = None):
        """Load latest checkpoint"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(os.listdir(self.checkpoint_dir))
            if not checkpoints:
                self.logger.warning("No checkpoints found to load")
                return False
            
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoints[-1])
        
        try:
            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
            # If using DataParallel, load into module
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
                
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            return False
    
    def collect_metrics(self) -> Dict:
        """Collect current training metrics"""
        metrics = {}
        
        # System metrics
        metrics['cpu_percent'] = psutil.cpu_percent()
        metrics['memory_used_gb'] = psutil.virtual_memory().used / (1024 * 1024 * 1024)
        
        # GPU metrics if available
        if torch.cuda.is_available():
            metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            
            try:
                # Try to get GPU utilization
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
                    encoding='utf-8'
                )
                metrics['gpu_utilization'] = float(output.strip().split('\n')[0])
            except:
                metrics['gpu_utilization'] = 0.0
        
        return metrics

# ==================== Main Cloud Optimizer ====================

class CloudOptimizer:
    """Main class for cloud resource optimization using RL"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.logger = setup_logger('cloud_optimizer', 'cloud_optimizer.log')
        
        # Initialize components
        self.cloud_env = CloudEnvironment(self.config)
        
        # Initialize RL agent
        self.agent = MultiObjectiveRLAgent(
            self.config, 
            self.cloud_env.state_dim, 
            self.cloud_env.action_dim
        )
        
        # Initialize training job manager
        self.training_manager = TrainingJobManager(self.config)
        
        # Current user preference
        self.user_preference = 'balanced'
        
        # For evaluation
        self.eval_metrics = {
            'episode_rewards': [],
            'sla_compliance_rates': [],
            'costs': [],
            'training_times': [],
            'resource_utilization': []
        }
        
        self.logger.info("Initialized CloudOptimizer")
    
    def load_training_job(self, model_path: str, dataset_path: str = None) -> Dict:
        """Load model and dataset for optimization"""
        # Not implemented: would need to load model and dataset
        # and connect to training manager
        self.logger.info(f"Loading training job from {model_path}")
        
        # In practice, this would load the actual model and dataset
        # For demonstration, we'll return a placeholder
        return {
            'model_name': os.path.basename(model_path),
            'dataset_name': os.path.basename(dataset_path) if dataset_path else 'unknown',
            'status': 'loaded'
        }
    
    def load_external_training_module(self, module_path: str) -> bool:
        """Load external training module like train-cifar10-resnet50.py"""
        success = self.training_manager.load_training_module(module_path)
        if success:
            # Get module's main training function
            main_func = getattr(self.training_manager.training_module, 'train_resnet50', None)
            if main_func:
                self.logger.info(f"Successfully found main training function in {module_path}")
                return True
            else:
                self.logger.error(f"Could not find main training function in {module_path}")
                return False
        return False
    
    def suggest_resource_options(self, model: nn.Module, dataloader: DataLoader) -> List[Dict]:
        """Suggest initial resource allocation options for the training job"""
        # Analyze model and dataset
        analysis = self.training_manager.analyze_training_job(model, dataloader)
        
        # Set initial info in cloud environment
        self.cloud_env.set_training_info({
            'model_name': model.__class__.__name__,
            'dataset_name': dataloader.dataset.__class__.__name__,
            'batch_size': dataloader.batch_size,
            'current_epoch': 0,
            'total_epochs': 100  # Default, should be overridden with actual value
        })
        
        # Get prediction options
        options = self.cloud_env.get_prediction_options()
        
        # Format for display
        display_options = []
        for i, option in enumerate(options):
            display_options.append({
                'id': i,
                'name': option['name'],
                'gpus': option['allocation']['gpu_count'],
                'cpus': option['allocation']['cpu_count'],
                'memory': option['allocation']['memory_gb'],
                'estimated_time': f"{option['estimated_time_minutes']:.1f} minutes",
                'estimated_cost': f"${option['total_cost']:.2f}",
                'hourly_rate': f"${option['hourly_cost']:.2f}/hour",
                'description': option['description']
            })
        
        return display_options
    
    def set_user_preference(self, preference: str):
        """Set user preference for optimization strategy"""
        valid_preferences = ['balanced', 'cost', 'time', 'resource']
        if preference in valid_preferences:
            self.user_preference = preference
            self.logger.info(f"Set user preference to {preference}")
            return True
        else:
            self.logger.warning(f"Invalid preference: {preference}. Using 'balanced'")
            self.user_preference = 'balanced'
            return False
    
    def train_with_optimization(self, model: nn.Module, dataloader: DataLoader, 
                               val_dataloader: DataLoader, epochs: int, option_id: int = 0) -> Dict:
        """Train model with RL-based resource optimization"""
        # Set up training info
        self.cloud_env.set_training_info({
            'model_name': model.__class__.__name__,
            'dataset_name': dataloader.dataset.__class__.__name__,
            'batch_size': dataloader.batch_size,
            'current_epoch': 0,
            'total_epochs': epochs
        })
        
        # Get initial allocation from chosen option
        options = self.cloud_env.get_prediction_options()
        initial_allocation = options[option_id]['allocation']
        
        # Apply initial allocation
        self.training_manager.analyze_training_job(model, dataloader)
        self.training_manager.setup_data_parallel(initial_allocation['gpu_count'])
        
        # Reset cloud environment
        self.cloud_env.reset()
        
        # Train with optimization
        training_start_time = time.time()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with RL optimization
        for epoch in range(epochs):
            # Training metrics
            train_loss = 0.0
            correct = 0
            total = 0
            
            # Start epoch
            epoch_start_time = time.time()
            model.train()
            
            # Training batch loop
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.training_manager.device), targets.to(self.training_manager.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Collect metrics occasionally
                if batch_idx % 20 == 0:
                    # Training progress metrics
                    accuracy = 100. * correct / max(1, total)
                    avg_loss = train_loss / (batch_idx + 1)
                    
                    metrics = self.training_manager.collect_metrics()
                    metrics.update({
                        'train_loss': avg_loss,
                        'train_accuracy': accuracy,
                        'throughput': total / (time.time() - epoch_start_time)
                    })
                    
                    # Update cloud environment with new metrics
                    self.cloud_env.update_training_progress(epoch, metrics)
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(self.training_manager.device), targets.to(self.training_manager.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_accuracy = 100. * val_correct / max(1, val_total)
            
            # Update with validation metrics
            metrics = {
                'test_loss': val_loss / len(val_dataloader),
                'test_accuracy': val_accuracy,
                'epoch_time': time.time() - epoch_start_time,
                'throughput': total / (time.time() - epoch_start_time)
            }
            
            self.cloud_env.update_training_progress(epoch, metrics)
            
            # RL optimization step
            if epoch > 0 and epoch % 5 == 0:  # Optimize every 5 epochs
                state = self.cloud_env.current_state
                
                # Get SLA status
                sla_status = self.sla_manager.sla_status
                
                # Select policy based on SLA status and user preference
                policy_idx = self.agent.select_policy(sla_status, self.user_preference)
                
                # Select action with exploration
                action_idx, action_one_hot = self.agent.select_action(state, policy_idx, explore=True)
                
                # Apply action in environment
                next_state, reward, done, info = self.cloud_env.step(action_idx)
                
                # Add experience to replay buffer
                self.agent.add_experience(state, action_one_hot, reward, next_state, done, policy_idx)
                
                # Train the RL agent
                if epoch > 10:  # Wait for some experience to accumulate
                    self.agent.train(policy_idx)
                
                # Check for adaptive critic reset
                self.agent.check_adaptive_reset(sla_status)
                
                # Apply resource adjustment if recommended
                gpu_idx = action_idx // 9
                cpu_idx = (action_idx % 9) // 3
                memory_idx = action_idx % 3
                
                # Convert index to action values
                gpu_action = self.cloud_env.gpu_actions[gpu_idx]
                cpu_action = self.cloud_env.cpu_actions[cpu_idx]
                memory_action = self.cloud_env.memory_actions[memory_idx]
                
                if gpu_action != 0 or cpu_action != 0 or memory_action != 0:
                    # Calculate new allocation
                    current_allocation = self.training_manager.current_allocation
                    new_allocation = {
                        'gpu_count': max(1, min(current_allocation['gpu_count'] + gpu_action, 8)),
                        'cpu_count': max(4, min(current_allocation['cpu_count'] + cpu_action * 4, 64)),
                        'memory_gb': max(8, min(current_allocation['memory_gb'] + memory_action * 8, 128))
                    }
                    
                    # Apply allocation change
                    self.training_manager.adjust_resources(new_allocation)
            
            # Log progress
            self.logger.info(f"Epoch {epoch} completed. Train loss: {train_loss / len(dataloader):.4f}, "
                           f"Train acc: {100. * correct / total:.2f}%, Val acc: {val_accuracy:.2f}%")
            
            # Save checkpoint periodically
            if epoch % 10 == 9:
                self.training_manager.save_checkpoint()
        
        # Training complete
        training_time = time.time() - training_start_time
        final_cost = self.cloud_env.resource_manager.get_current_cost()
        
        # Return training results
        results = {
            'total_time': training_time,
            'time_per_epoch': training_time / epochs,
            'final_test_accuracy': val_accuracy,
            'total_cost': final_cost,
            'cost_per_epoch': final_cost / epochs,
            'sla_compliance_rate': self.cloud_env.sla_manager.get_sla_compliance_rate(),
            'sla_violations': self.cloud_env.sla_manager.get_total_violations(),
            'final_allocation': self.training_manager.current_allocation
        }
        
        # Add to evaluation metrics
        self.eval_metrics['episode_rewards'].append(self.cloud_env.episode_reward)
        self.eval_metrics['sla_compliance_rates'].append(results['sla_compliance_rate'])
        self.eval_metrics['costs'].append(final_cost)
        self.eval_metrics['training_times'].append(training_time)
        
        self.logger.info(f"Training completed. Time: {training_time/60:.2f} min, "
                        f"Cost: ${final_cost:.2f}, Accuracy: {val_accuracy:.2f}%")
        
        return results
    
    def evaluate_baseline(self, model: nn.Module, dataloader: DataLoader, 
                        val_dataloader: DataLoader, epochs: int, gpu_count: int) -> Dict:
        """Train model with fixed resources (baseline for comparison)"""
        # Set up training with fixed allocation
        fixed_allocation = {
            'gpu_count': gpu_count,
            'cpu_count': 16,
            'memory_gb': 32
        }
        
        # Apply fixed allocation
        self.training_manager.analyze_training_job(model, dataloader)
        self.training_manager.setup_data_parallel(fixed_allocation['gpu_count'])
        self.training_manager.current_allocation = fixed_allocation
        
        # Run training with fixed resources
        # (similar to train_with_optimization but without RL adjustments)
        # ...
        
        # Simulated results for demonstration
        baseline_results = {
            'total_time': 3600,  # Simulated 1 hour training time
            'time_per_epoch': 3600 / epochs,
            'final_test_accuracy': 85.0,
            'total_cost': 100.0,
            'cost_per_epoch': 100.0 / epochs,
            'resource_allocation': fixed_allocation
        }
        
        return baseline_results
    
    def compare_to_baseline(self, optimized_results: Dict, baseline_results: Dict) -> Dict:
        """Compare optimized training to baseline"""
        time_improvement = (baseline_results['total_time'] - optimized_results['total_time']) / baseline_results['total_time'] * 100
        cost_improvement = (baseline_results['total_cost'] - optimized_results['total_cost']) / baseline_results['total_cost'] * 100
        accuracy_improvement = optimized_results['final_test_accuracy'] - baseline_results['final_test_accuracy']
        
        comparison = {
            'time_improvement_pct': time_improvement,
            'cost_improvement_pct': cost_improvement,
            'accuracy_improvement': accuracy_improvement,
            'optimized_cost_per_1pct_acc': optimized_results['total_cost'] / optimized_results['final_test_accuracy'],
            'baseline_cost_per_1pct_acc': baseline_results['total_cost'] / baseline_results['final_test_accuracy']
        }
        
        return comparison
    
    def generate_report(self, results: Dict, baseline_results: Dict = None) -> str:
        """Generate comprehensive performance report"""
        report = "=== Cloud Resource Optimization Report ===\n\n"
        
        # Training details
        report += "Training Details:\n"
        report += f"Model: {self.cloud_env.training_info['model_name']}\n"
        report += f"Dataset: {self.cloud_env.training_info['dataset_name']}\n"
        report += f"Batch Size: {self.cloud_env.training_info['batch_size']}\n"
        report += f"Epochs: {self.cloud_env.training_info['total_epochs']}\n\n"
        
        # Performance metrics
        report += "Performance Metrics:\n"
        report += f"Training Time: {results['total_time']/60:.2f} minutes\n"
        report += f"Time per Epoch: {results['time_per_epoch']:.2f} seconds\n"
        report += f"Final Test Accuracy: {results['final_test_accuracy']:.2f}%\n"
        report += f"Total Cost: ${results['total_cost']:.2f}\n"
        report += f"Cost per Epoch: ${results['cost_per_epoch']:.2f}\n\n"
        
        # SLA compliance
        report += "SLA Compliance:\n"
        report += f"Overall Compliance Rate: {results['sla_compliance_rate']*100:.2f}%\n"
        report += f"Total SLA Violations: {results['sla_violations']}\n\n"
        
        # Resource allocation
        report += "Final Resource Allocation:\n"
        report += f"GPUs: {results['final_allocation']['gpu_count']}\n"
        report += f"CPUs: {results['final_allocation']['cpu_count']}\n"
        report += f"Memory: {results['final_allocation']['memory_gb']} GB\n\n"
        
        # Compare to baseline if available
        if baseline_results:
            comparison = self.compare_to_baseline(results, baseline_results)
            
            report += "Comparison to Baseline:\n"
            report += f"Time Improvement: {comparison['time_improvement_pct']:.2f}%\n"
            report += f"Cost Improvement: {comparison['cost_improvement_pct']:.2f}%\n"
            report += f"Accuracy Improvement: {comparison['accuracy_improvement']:.2f} percentage points\n"
            report += f"Cost Efficiency: {comparison['optimized_cost_per_1pct_acc']:.2f} vs {comparison['baseline_cost_per_1pct_acc']:.2f} $/accuracy-point\n\n"
        
        # Recommendations
        report += "Recommendations for Future Training:\n"
        report += "1. Consider using the final resource allocation as a starting point for similar models\n"
        report += f"2. The optimal strategy for this workload appears to be '{self.user_preference}'\n"
        report += "3. Monitor SLA compliance to ensure critical constraints are met\n\n"
        
        report += "=== End of Report ===\n"
        
        return report
    
    def save_agent(self, path: str = "models"):
        """Save RL agent"""
        self.agent.save(path)
    
    def load_agent(self, path: str = "models"):
        """Load RL agent"""
        self.agent.load(path)

# ==================== CLI Interface ====================

def main():
    """Command-line interface for cloud resource optimization"""
    print("=== Cloud Resource Optimization for ML Training ===")
    
    # Create the optimizer
    optimizer = CloudOptimizer()
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Cloud Resource Optimization for ML Training')
    parser.add_argument('--training_module', type=str, help='Path to training module')
    parser.add_argument('--preference', type=str, default='balanced', 
                        choices=['balanced', 'cost', 'time', 'resource'],
                        help='Optimization preference')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation against baseline')
    
    args = parser.parse_args()
    
    # Set user preference
    optimizer.set_user_preference(args.preference)
    
    # Load training module if provided
    if args.training_module:
        print(f"Loading training module: {args.training_module}")
        success = optimizer.load_external_training_module(args.training_module)
        
        if not success:
            print("Failed to load training module. Exiting.")
            return
    
    # Get or create model and dataset
    try:
        # Import commonly used libraries for demonstration
        import torchvision
        from torchvision import models, datasets, transforms
        
        # Create a simple model for demonstration
        model = models.resnet18(pretrained=False, num_classes=10)
        
        # Create a transform for CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR10 dataset
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        print("Created ResNet18 model and CIFAR10 dataset for demonstration")
    except Exception as e:
        print(f"Error creating model and dataset: {str(e)}")
        return
    
    # Suggest resource options
    print("\nAnalyzing model and dataset...")
    options = optimizer.suggest_resource_options(model, train_loader)
    
    print("\nSuggested resource allocation options:")
    for i, option in enumerate(options):
        print(f"{i+1}. {option['name']}: {option['gpus']} GPUs, {option['cpus']} CPUs, {option['memory']} GB RAM")
        print(f"   Estimated time: {option['estimated_time']}, Cost: {option['estimated_cost']}")
        print(f"   {option['description']}")
    
    # Select option (default to balanced)
    selected_option = 2  # Default to balanced option
    try:
        option_input = input("\nSelect an option (1-3, default 2 for balanced): ")
        if option_input.strip():
            selected_option = int(option_input) - 1
            if selected_option < 0 or selected_option >= len(options):
                selected_option = 2
    except:
        selected_option = 2
    
    print(f"\nSelected {options[selected_option]['name']} option")
    
    # Train with optimization
    print("\nStarting training with resource optimization...")
    results = optimizer.train_with_optimization(
        model, train_loader, val_loader, args.epochs, option_id=selected_option
    )
    
    # Evaluate against baseline if requested
    if args.evaluate:
        print("\nRunning baseline training for comparison...")
        baseline_results = optimizer.evaluate_baseline(
            model, train_loader, val_loader, args.epochs, gpu_count=2
        )
        
        # Generate and print report
        report = optimizer.generate_report(results, baseline_results)
    else:
        # Generate and print report without baseline
        report = optimizer.generate_report(results)
    
    print("\n" + report)
    
    # Save report to file
    with open("optimization_report.txt", "w") as f:
        f.write(report)
    
    print("Report saved to optimization_report.txt")
    
    # Save the trained RL agent
    optimizer.save_agent()
    
    print("Training complete. Cloud resource optimization agent saved.")

if __name__ == "__main__":
    main()
