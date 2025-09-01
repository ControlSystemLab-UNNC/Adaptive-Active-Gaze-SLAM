# Adaptive Active Gaze SLAM - Python Modules Documentation

This repository contains a collection of Python modules for implementing an Adaptive Active Gaze SLAM system with reinforcement learning-based gaze control. The system uses Fisher information analysis to guide robot gaze direction for optimal feature extraction and SLAM performance.

## 📁 Module Overview

| Module | Purpose | Status |
|--------|---------|---------|
| `aag_slam_simulator.py` | Core robot simulation environment | ✅ Active |
| `aag_slam_fisher_analyzer.py` | Fisher information direction analysis | ✅ Active |
| `aag_slam_ppo_agent.py` | PPO reinforcement learning agent | ✅ Active |
| `aag_slam_greedy_agent.py` | Greedy baseline agent | ✅ Active |

---

## 🤖 1. Robot Simulator (`aag_slam_simulator.py`)

### Description
Core simulation environment providing robot physics, field-of-view geometry, Fisher information mapping, and collision detection. Designed with a clean separation between computation (`RobotCore`) and visualization (`RobotRenderer`).

### Key Classes

#### `RobotCore`
Headless computational core handling robot physics and environment simulation.

**Key Features:**
- 2D robot simulation with velocity control
- Field-of-view based feature detection
- Fisher information mapping (robot-centered local view)
- Collision detection and obstacle avoidance
- Feature decay simulation

**Main Methods:**
- `reset(regenerate_map=True)` - Reset simulation environment
- `set_velocity(linear_vel, angular_vel)` - Control robot movement
- `set_gaze(gaze_angle_deg)` - Control gaze direction
- `step()` - Advance simulation by one time step
- `update_maps()` - Update Fisher information maps
- `fisher_map_stats()` - Get global Fisher statistics
- `fov_fisher_stats()` - Get FOV-specific Fisher statistics

#### `RobotRenderer`
Visualization component for real-time rendering of simulation state.

**Features:**
- Real-time OpenCV-based visualization
- Robot, obstacles, and FOV overlay rendering
- Fisher information map visualization
- Interactive controls (press 'q' or ESC to quit)

### Usage Examples

#### Basic Simulation
```bash
# Run basic simulation with visualization
python aag_slam_simulator.py --steps 1000 --world-size 40.0

# Run headless simulation
python aag_slam_simulator.py --steps 1000 --headless

# Real-time simulation
python aag_slam_simulator.py --steps 1000 --realtime
```

#### Programmatic Usage
```python
from aag_slam_simulator import RobotCore, RobotRenderer

# Create core simulation
core = RobotCore(world_width=40.0, world_height=40.0, fov_angle=90)

# Optional: Add renderer for visualization
renderer = RobotRenderer(core)

# Simulation loop
for step in range(1000):
    core.set_velocity(1.0, 0.1)  # Move forward with slight turn
    core.set_gaze(45.0)          # Look 45 degrees
    core.step()                  # Advance simulation
    core.update_maps()           # Update Fisher maps
    
    if renderer:
        renderer.render()        # Visualize current state
```

### Command Line Arguments
- `--steps`: Number of simulation steps (default: 1000)
- `--world-size`: World size in meters (default: 40.0)
- `--fov-angle`: Field of view angle in degrees (default: 90)
- `--headless`: Run without visualization
- `--realtime`: Run in real-time speed

---

## 🎯 2. Fisher Information Analyzer (`aag_slam_fisher_analyzer.py`)

### Description
Analyzes Fisher information maps to extract primary and secondary directional information for guiding robot gaze. Uses directional sector integration to identify optimal viewing directions.

### Key Classes

#### `FisherDirectionInfo`
Data structure containing direction analysis results.

**Attributes:**
- `angle`: Direction angle [0, 360)
- `strength`: Fisher information strength
- `confidence`: Direction confidence [0, 1]
- `center`: Center point in local map coordinates

#### `FisherMapAnalyzer`
Main analyzer class for extracting Fisher directions from 2D maps.

**Key Features:**
- Directional sector scanning with configurable parameters
- Primary and secondary direction extraction
- Confidence-based filtering
- Robot-centered local map analysis

**Main Methods:**
- `analyze(fmap)` - Extract primary and secondary directions
- `_scan_directions(fmap, center)` - Scan all directions for Fisher information
- `_extract_points(fmap)` - Extract significant Fisher points

### Usage Examples

#### Standalone Analysis
```bash
# Run Fisher analyzer with visualization
python aag_slam_fisher_analyzer.py --steps 1000 --world-size 40.0

# Custom analysis parameters
python aag_slam_fisher_analyzer.py --steps 1000 --angle-step 5 --sector-width 30.0

# Headless analysis
python aag_slam_fisher_analyzer.py --steps 1000 --headless
```

#### Programmatic Usage
```python
from aag_slam_fisher_analyzer import FisherMapAnalyzer
from aag_slam_simulator import RobotCore

# Create analyzer
analyzer = FisherMapAnalyzer(
    threshold_ratio=0.2,
    min_points=15,
    fov_angle=90.0
)

# Create simulator
core = RobotCore()
core.reset()
core.update_maps()

# Analyze Fisher directions
primary, secondary = analyzer.analyze(core.feature_map)

if primary:
    print(f"Primary direction: {primary.angle:.1f}° (strength: {primary.strength:.3f})")
if secondary:
    print(f"Secondary direction: {secondary.angle:.1f}° (strength: {secondary.strength:.3f})")
```

### Command Line Arguments
- `--steps`: Number of simulation steps (default: 1000)
- `--world-size`: World size in meters (default: 40.0)
- `--fov-angle`: FOV angle in degrees (default: 90)
- `--headless`: Run without visualization
- `--realtime`: Run in real-time speed
- `--angle-step`: Directional scan angle step in degrees (default: 5)
- `--sector-width`: Sector width in degrees around direction (default: 30.0)

---

## 🧠 3. PPO Reinforcement Learning Agent (`aag_slam_ppo_agent.py`)

### Description
Proximal Policy Optimization (PPO) agent for learning optimal gaze control policies. Uses discrete action space with 72 actions (5° increments) and simplified 3-layer neural networks optimized for the 3D observation space.

### Key Classes

#### `Actor`
Policy network for action selection.

**Architecture:**
- Input: 3D observation [cos(θ), sin(θ), strength_norm]
- Hidden: 64 units with ReLU activation
- Output: 72 discrete action logits

#### `Critic`
Value function network for advantage estimation.

**Architecture:**
- Input: 3D observation [cos(θ), sin(θ), strength_norm]
- Hidden: 64 units with ReLU activation
- Output: Single value estimate

#### `PPO`
Main PPO algorithm implementation.

**Key Features:**
- Discrete action space (72 actions for 360° gaze control)
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Entropy regularization
- Gradient clipping for stable training

**Main Methods:**
- `act(obs)` - Select action based on current policy
- `value(obs)` - Estimate state value
- `store(s, a, logp, r, v, done)` - Store experience
- `update()` - Perform PPO policy update

### Usage Examples

#### Training
```bash
# Basic training
python aag_slam_ppo_agent.py --train --episodes 300

# Headless training with custom parameters
python aag_slam_ppo_agent.py --train --episodes 500 --max-steps 1000 --headless

# Advanced training configuration
python aag_slam_ppo_agent.py --train \
    --episodes 1000 \
    --max-steps 1000 \
    --update-freq 4096 \
    --lr 1e-4 \
    --hidden 128 \
    --headless
```

#### Testing
```bash
# Test trained model
python aag_slam_ppo_agent.py --test

# Test with custom model
python aag_slam_ppo_agent.py --test --model checkpoints/best.pth

# Test in headless mode
python aag_slam_ppo_agent.py --test --headless
```

#### Programmatic Usage
```python
from aag_slam_ppo_agent import PPO
from aag_slam_simulator import RobotCore
from aag_slam_fisher_analyzer import FisherMapAnalyzer

# Create components
core = RobotCore()
analyzer = FisherMapAnalyzer()
agent = PPO(obs_dim=3, action_dim=72, hidden=64)

# Training loop example
for episode in range(100):
    core.reset()
    episode_reward = 0
    
    for step in range(500):
        # Get observation
        primary, _ = analyzer.analyze(core.feature_map)
        obs = PPO.obs_from_primary(primary, core.feature_map)
        
        # Select action
        gaze_angle, log_prob, action_idx = agent.act(obs)
        value = agent.value(obs)
        
        # Execute action
        core.set_gaze(gaze_angle)
        core.step()
        core.update_maps()
        
        # Compute reward and store experience
        # ... (reward computation logic)
        
        agent.store(obs, action_idx, log_prob, reward, value, False)
    
    # Update policy
    agent.update()
```

### Command Line Arguments

**Mode Selection:**
- `--train`: Training mode
- `--test`: Testing mode

**Environment Parameters:**
- `--world-size`: World size in meters (default: 40.0)
- `--fov-angle`: FOV angle in degrees (default: 90.0)
- `--headless`: Run without visualization
- `--realtime`: Run in real-time speed

**Training Parameters:**
- `--episodes`: Number of training episodes (default: 300)
- `--max-steps`: Maximum steps per episode (default: 500)
- `--update-freq`: PPO update frequency (default: 2048)
- `--vel-interval`: Velocity change interval (default: 50)
- `--seed`: Random seed (default: 42)

**Agent Parameters:**
- `--action-dim`: Number of discrete actions (default: 72)
- `--hidden`: Hidden layer dimension (default: 64)
- `--lr`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.95)
- `--lam`: GAE lambda (default: 0.90)
- `--clip`: PPO clip ratio (default: 0.20)
- `--vcoef`: Value loss coefficient (default: 0.5)
- `--ecoef`: Entropy loss coefficient (default: 0.01)
- `--ppo-epochs`: PPO update epochs (default: 4)

**Model Parameters:**
- `--model`: Model save/load path (default: "ppo_gaze.pth")

---

## 🎲 4. Greedy Baseline Agent (`aag_slam_greedy_agent.py`)

### Description
Enhanced greedy baseline agent that uses heuristic strategies to select optimal gaze directions based on Fisher information analysis. Serves as a baseline for comparing reinforcement learning performance.

### Key Classes

#### `EnhancedGreedyFisherAgent`
Sophisticated greedy agent with multiple heuristic strategies.

**Key Features:**
- Fisher direction alignment scoring
- Anti-decay mechanisms to prevent information loss
- Gaze smoothness constraints
- Adaptive parameter adjustment
- Comprehensive decision tracking and statistics

**Main Methods:**
- `get_action(core)` - Compute optimal gaze angle
- `_calculate_optimal_gaze_angle()` - Core optimization logic
- `_calculate_alignment_score()` - Score direction alignment
- `_calculate_anti_decay_score()` - Prevent information decay
- `_apply_smoothness_constraint()` - Ensure smooth gaze transitions
- `get_statistics()` - Return performance statistics

### Usage Examples

#### Running Greedy Agent Test
```bash
# Basic greedy agent test
python aag_slam_greedy_agent.py --steps 1000 --world-size 40.0

# Custom parameters
python aag_slam_greedy_agent.py \
    --steps 2000 \
    --world-size 50.0 \
    --alignment-weight 0.7 \
    --anti-decay-weight 0.2 \
    --smoothness-weight 0.1

# Headless mode for performance testing
python aag_slam_greedy_agent.py --steps 1000 --headless
```

#### Programmatic Usage
```python
from aag_slam_greedy_agent import EnhancedGreedyFisherAgent
from aag_slam_simulator import RobotCore
from aag_slam_fisher_analyzer import FisherMapAnalyzer

# Create components
core = RobotCore()
analyzer = FisherMapAnalyzer()
agent = EnhancedGreedyFisherAgent(
    alignment_weight=0.6,
    anti_decay_weight=0.3,
    smoothness_weight=0.1
)

# Test loop
core.reset()
for step in range(1000):
    # Get optimal gaze angle from greedy agent
    gaze_angle = agent.get_action(core)
    
    # Execute action
    core.set_gaze(gaze_angle)
    core.step()
    core.update_maps()

# Get performance statistics
stats = agent.get_statistics()
print(f"Average decision score: {stats['avg_decision_score']:.3f}")
print(f"Gaze stability: {stats['gaze_stability']:.3f}")
```

### Command Line Arguments

**Environment Parameters:**
- `--steps`: Number of simulation steps (default: 1000)
- `--world-size`: World size in meters (default: 40.0)
- `--fov-angle`: FOV angle in degrees (default: 90)
- `--headless`: Run without visualization
- `--realtime`: Run in real-time speed

**Agent Strategy Parameters:**
- `--alignment-weight`: Direction alignment weight (default: 0.6)
- `--anti-decay-weight`: Anti-decay weight (default: 0.3)
- `--smoothness-weight`: Smoothness weight (default: 0.1)
- `--max-angle-change`: Maximum angle change per step (default: 45.0)
- `--adaptive-mode`: Enable adaptive parameter adjustment (default: True)

**Fisher Analyzer Parameters:**
- `--fisher-threshold-ratio`: Fisher threshold ratio (default: 0.2)
- `--fisher-min-points`: Minimum points for direction (default: 15)

---

## 📊 5. Performance Comparison and Metrics

### Common Evaluation Metrics

All agents can be evaluated using consistent metrics:

1. **Fisher Information Metrics:**
   - Mean Fisher value over time
   - Total features detected
   - Feature density in maps

2. **Gaze Control Metrics:**
   - Gaze direction alignment with Fisher directions
   - Gaze smoothness (angular velocity)
   - Decision consistency

3. **Exploration Metrics:**
   - World coverage percentage
   - Feature map completeness
   - Information gain rate

### Comparative Analysis Example

```python
# Compare PPO vs Greedy agent performance
from aag_slam_ppo_agent import PPO
from aag_slam_greedy_agent import EnhancedGreedyFisherAgent

# Test both agents with same environment setup
def compare_agents(num_episodes=10):
    results = {'ppo': [], 'greedy': []}
    
    for episode in range(num_episodes):
        # Test PPO agent
        ppo_score = test_ppo_agent()
        results['ppo'].append(ppo_score)
        
        # Test Greedy agent
        greedy_score = test_greedy_agent()
        results['greedy'].append(greedy_score)
    
    print(f"PPO Average: {np.mean(results['ppo']):.3f}")
    print(f"Greedy Average: {np.mean(results['greedy']):.3f}")
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Activate virtual environment
source aag-slam-env/bin/activate

# Verify dependencies
python -c "import torch, cv2, numpy as np; print('Dependencies OK')"
```

### 2. Basic Testing
```bash
# Test simulator
python aag_slam_simulator.py --steps 100

# Test Fisher analyzer
python aag_slam_fisher_analyzer.py --steps 100

# Test greedy agent
python aag_slam_greedy_agent.py --steps 100
```

### 3. Train PPO Agent
```bash
# Quick training run
python aag_slam_ppo_agent.py --train --episodes 50 --headless

# Test trained model
python aag_slam_ppo_agent.py --test --episodes 5
```

---

## 🔧 Development Guidelines

### Code Structure
- **Modular Design**: Each component is self-contained with clear interfaces
- **Separation of Concerns**: Computation (Core) and visualization (Renderer) are separated
- **Consistent APIs**: All agents use similar observation and action interfaces
- **Comprehensive Logging**: Detailed statistics and debugging information

### Adding New Agents
To add a new gaze control agent:

1. Implement the agent class with a `get_action(core)` method
2. Use the same observation format as existing agents
3. Follow the established command-line argument patterns
4. Include comprehensive statistics tracking

### Testing and Validation
- Use consistent environment parameters across agents
- Implement statistical significance testing for comparisons
- Validate against known baseline behaviors
- Monitor for training stability and convergence

---

## 📚 Dependencies

### Core Requirements
- **Python 3.10+**
- **NumPy**: Numerical computations
- **OpenCV (cv2)**: Visualization and image processing
- **PyTorch**: Deep learning framework for PPO agent

### Optional Dependencies
- **Matplotlib**: Additional plotting and visualization
- **Scipy**: Advanced mathematical functions
- **Jupyter**: Interactive development and analysis

### Installation
```bash
pip install torch numpy opencv-python matplotlib scipy jupyter
```

---

## 📄 License and Citation

This project is part of the Adaptive Active Gaze SLAM research at the University of Nottingham Ningbo China, Control Systems Lab.

**Development Team**: Control Systems Lab, UNNC  
**Last Updated**: September 1, 2025

For academic use, please cite the corresponding research papers and acknowledge the Control Systems Lab at UNNC.
