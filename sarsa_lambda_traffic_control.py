"""
SARSA(λ) Algorithm for Intelligent Traffic Signal Control
==========================================================

This implementation uses SARSA(λ) reinforcement learning to optimize
traffic signal timing at a four-way intersection.

Key Components:
- State: Queue lengths on each lane
- Actions: Different signal phase configurations
- Reward: Negative of total waiting time and queue length
- Eligibility Traces: λ parameter for credit assignment

Author: Implementation based on research papers on RL-based traffic control
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class TrafficEnvironment:
    """
    Simulates a 4-way intersection with traffic signals.
    
    State Space: Queue lengths on 4 approaches (North, South, East, West)
    Action Space: 4 signal phases (NS-Green, EW-Green, NS-Left, EW-Left)
    """
    
    def __init__(self, max_queue=20):
        self.max_queue = max_queue
        self.num_lanes = 4  # N, S, E, W
        self.num_actions = 4  # Different signal phases
        
        # Action definitions (which lanes get green light)
        self.actions = {
            0: [0, 1],      # North-South through traffic
            1: [2, 3],      # East-West through traffic
            2: [0],         # North turn
            3: [2]          # East turn
        }
        
        # Traffic arrival rates (vehicles per timestep)
        self.arrival_rates = {
            'peak': [0.7, 0.7, 0.8, 0.8],      # Peak hours
            'normal': [0.4, 0.4, 0.5, 0.5],    # Normal hours
            'light': [0.2, 0.2, 0.3, 0.3]      # Light traffic
        }
        
        self.current_traffic_mode = 'normal'
        self.state = np.zeros(self.num_lanes)
        self.time_step = 0
        self.total_waiting_time = 0
        self.total_vehicles_served = 0
        
    def reset(self):
        """Reset the environment to initial state."""
        self.state = np.random.randint(0, 5, self.num_lanes)
        self.time_step = 0
        self.total_waiting_time = 0
        self.total_vehicles_served = 0
        return self._discretize_state()
    
    def _discretize_state(self):
        """Convert continuous queue lengths to discrete state."""
        # Discretize into bins: [0-3, 4-7, 8-11, 12-15, 16+]
        discrete_state = tuple(min(int(q // 4), 4) for q in self.state)
        return discrete_state
    
    def step(self, action):
        """
        Execute one timestep in the environment.
        
        Args:
            action: Signal phase to activate (0-3)
            
        Returns:
            next_state: New state after action
            reward: Reward signal
            done: Whether episode is complete
            info: Additional information
        """
        # Vehicles depart from lanes with green light
        green_lanes = self.actions[action]
        departures = np.zeros(self.num_lanes)
        
        for lane in green_lanes:
            # Service rate: 2-4 vehicles per timestep on green
            service = min(self.state[lane], np.random.randint(2, 5))
            departures[lane] = service
            self.state[lane] -= service
            self.total_vehicles_served += service
        
        # New vehicle arrivals (Poisson-like distribution)
        rates = self.arrival_rates[self.current_traffic_mode]
        arrivals = np.random.poisson(np.array(rates) * 2)
        self.state = np.minimum(self.state + arrivals, self.max_queue)
        
        # Calculate reward (negative cost)
        waiting_cost = np.sum(self.state)  # Total queue length
        delay_cost = np.sum(self.state ** 1.5)  # Penalize long queues more
        
        reward = -(waiting_cost + 0.5 * delay_cost)
        
        self.total_waiting_time += waiting_cost
        self.time_step += 1
        
        # Change traffic mode periodically
        if self.time_step % 100 == 0:
            modes = ['peak', 'normal', 'light']
            self.current_traffic_mode = np.random.choice(modes, p=[0.3, 0.5, 0.2])
        
        done = self.time_step >= 500  # Episode length
        
        info = {
            'queue_lengths': self.state.copy(),
            'waiting_time': waiting_cost,
            'vehicles_served': self.total_vehicles_served,
            'traffic_mode': self.current_traffic_mode
        }
        
        return self._discretize_state(), reward, done, info


class SARSALambdaAgent:
    """
    SARSA(λ) agent for traffic signal control.
    
    Uses eligibility traces to propagate credit to earlier state-action pairs.
    """
    
    def __init__(self, num_actions=4, alpha=0.1, gamma=0.95, lambda_param=0.6, epsilon=0.2):
        """
        Initialize SARSA(λ) agent.
        
        Args:
            num_actions: Number of possible actions
            alpha: Learning rate
            gamma: Discount factor
            lambda_param: Eligibility trace decay
            epsilon: Exploration rate
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        
        # Q-table: Q(s, a)
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        
        # Eligibility traces: E(s, a)
        self.E = defaultdict(lambda: np.zeros(num_actions))
        
        # For tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            action: Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action):
        """
        SARSA(λ) update rule.
        
        Updates Q-values and eligibility traces based on TD error.
        """
        # TD error: δ = R + γQ(s', a') - Q(s, a)
        td_error = (reward + 
                   self.gamma * self.Q[next_state][next_action] - 
                   self.Q[state][action])
        
        # Update eligibility trace for current state-action
        self.E[state][action] += 1
        
        # Update all Q-values and decay eligibility traces
        for s in list(self.Q.keys()):
            for a in range(self.num_actions):
                # Q(s, a) ← Q(s, a) + α * δ * E(s, a)
                self.Q[s][a] += self.alpha * td_error * self.E[s][a]
                
                # E(s, a) ← γ * λ * E(s, a)
                self.E[s][a] *= self.gamma * self.lambda_param
                
                # Remove traces that are too small
                if abs(self.E[s][a]) < 1e-5:
                    self.E[s][a] = 0
    
    def reset_traces(self):
        """Reset eligibility traces at the start of each episode."""
        self.E = defaultdict(lambda: np.zeros(self.num_actions))
    
    def decay_epsilon(self, episode, total_episodes):
        """Decay exploration rate over time."""
        self.epsilon = max(0.01, 0.2 * (1 - episode / total_episodes))


def train_agent(num_episodes=1000, render_interval=100):
    """
    Train SARSA(λ) agent on traffic control task.
    
    Args:
        num_episodes: Number of training episodes
        render_interval: How often to print progress
        
    Returns:
        agent: Trained agent
        env: Environment
        metrics: Training metrics
    """
    env = TrafficEnvironment()
    agent = SARSALambdaAgent()
    
    # Metrics tracking
    episode_rewards = []
    episode_waiting_times = []
    episode_vehicles_served = []
    avg_queue_lengths = []
    
    print("Training SARSA(λ) Agent for Traffic Signal Control")
    print("=" * 60)
    
    for episode in range(num_episodes):
        state = env.reset()
        action = agent.choose_action(state, training=True)
        agent.reset_traces()
        
        episode_reward = 0
        episode_queue_lengths = []
        
        done = False
        while not done:
            # Take action
            next_state, reward, done, info = env.step(action)
            next_action = agent.choose_action(next_state, training=True)
            
            # SARSA(λ) update
            agent.update(state, action, reward, next_state, next_action)
            
            # Track metrics
            episode_reward += reward
            episode_queue_lengths.append(np.mean(info['queue_lengths']))
            
            # Move to next state
            state = next_state
            action = next_action
        
        # Decay exploration
        agent.decay_epsilon(episode, num_episodes)
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(env.total_waiting_time)
        episode_vehicles_served.append(env.total_vehicles_served)
        avg_queue_lengths.append(np.mean(episode_queue_lengths))
        
        # Print progress
        if (episode + 1) % render_interval == 0:
            avg_reward = np.mean(episode_rewards[-render_interval:])
            avg_waiting = np.mean(episode_waiting_times[-render_interval:])
            avg_served = np.mean(episode_vehicles_served[-render_interval:])
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Waiting Time: {avg_waiting:.2f}")
            print(f"  Avg Vehicles Served: {avg_served:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print("-" * 60)
    
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_waiting_times': episode_waiting_times,
        'episode_vehicles_served': episode_vehicles_served,
        'avg_queue_lengths': avg_queue_lengths
    }
    
    return agent, env, metrics


def evaluate_agent(agent, env, num_episodes=50):
    """
    Evaluate trained agent performance.
    
    Args:
        agent: Trained SARSA(λ) agent
        env: Traffic environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        eval_metrics: Evaluation results
    """
    eval_rewards = []
    eval_waiting_times = []
    eval_vehicles_served = []
    action_distribution = np.zeros(4)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, training=False)
            action_distribution[action] += 1
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
        eval_waiting_times.append(env.total_waiting_time)
        eval_vehicles_served.append(env.total_vehicles_served)
    
    eval_metrics = {
        'rewards': eval_rewards,
        'waiting_times': eval_waiting_times,
        'vehicles_served': eval_vehicles_served,
        'action_distribution': action_distribution / action_distribution.sum()
    }
    
    return eval_metrics


def compare_with_fixed_timing(env, num_episodes=50):
    """
    Compare SARSA agent with fixed-timing baseline.
    
    Args:
        env: Traffic environment
        num_episodes: Number of episodes for comparison
        
    Returns:
        baseline_metrics: Fixed-timing performance
    """
    baseline_rewards = []
    baseline_waiting_times = []
    baseline_vehicles_served = []
    
    # Fixed timing: cycle through phases
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Simple fixed cycle: 10 steps per phase
            action = (step // 10) % 4
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
        
        baseline_rewards.append(episode_reward)
        baseline_waiting_times.append(env.total_waiting_time)
        baseline_vehicles_served.append(env.total_vehicles_served)
    
    baseline_metrics = {
        'rewards': baseline_rewards,
        'waiting_times': baseline_waiting_times,
        'vehicles_served': baseline_vehicles_served
    }
    
    return baseline_metrics


def visualize_results(metrics, eval_metrics, baseline_metrics):
    """
    Create comprehensive visualization of training and evaluation results.
    
    Args:
        metrics: Training metrics
        eval_metrics: Evaluation metrics
        baseline_metrics: Baseline comparison metrics
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Training Progress - Cumulative Reward
    ax1 = plt.subplot(3, 3, 1)
    window = 50
    smoothed_rewards = pd.Series(metrics['episode_rewards']).rolling(window).mean()
    ax1.plot(smoothed_rewards, linewidth=2, color='#2E86AB', label='SARSA(λ)')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Cumulative Reward', fontsize=11)
    ax1.set_title('Training Progress: Reward Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average Waiting Time During Training
    ax2 = plt.subplot(3, 3, 2)
    smoothed_waiting = pd.Series(metrics['episode_waiting_times']).rolling(window).mean()
    ax2.plot(smoothed_waiting, linewidth=2, color='#A23B72', label='Waiting Time')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Total Waiting Time', fontsize=11)
    ax2.set_title('Training Progress: Waiting Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Vehicles Served During Training
    ax3 = plt.subplot(3, 3, 3)
    smoothed_served = pd.Series(metrics['episode_vehicles_served']).rolling(window).mean()
    ax3.plot(smoothed_served, linewidth=2, color='#F18F01', label='Vehicles Served')
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Vehicles Served', fontsize=11)
    ax3.set_title('Training Progress: Throughput', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. SARSA vs Fixed-Timing: Reward Comparison
    ax4 = plt.subplot(3, 3, 4)
    comparison_data = [eval_metrics['rewards'], baseline_metrics['rewards']]
    bp = ax4.boxplot(comparison_data, labels=['SARSA(λ)', 'Fixed-Timing'],
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#C73E1D')
    ax4.set_ylabel('Cumulative Reward', fontsize=11)
    ax4.set_title('Performance Comparison: Reward', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. SARSA vs Fixed-Timing: Waiting Time
    ax5 = plt.subplot(3, 3, 5)
    comparison_data = [eval_metrics['waiting_times'], baseline_metrics['waiting_times']]
    bp = ax5.boxplot(comparison_data, labels=['SARSA(λ)', 'Fixed-Timing'],
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#C73E1D')
    ax5.set_ylabel('Total Waiting Time', fontsize=11)
    ax5.set_title('Performance Comparison: Waiting Time', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. SARSA vs Fixed-Timing: Throughput
    ax6 = plt.subplot(3, 3, 6)
    comparison_data = [eval_metrics['vehicles_served'], baseline_metrics['vehicles_served']]
    bp = ax6.boxplot(comparison_data, labels=['SARSA(λ)', 'Fixed-Timing'],
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#C73E1D')
    ax6.set_ylabel('Vehicles Served', fontsize=11)
    ax6.set_title('Performance Comparison: Throughput', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Action Distribution
    ax7 = plt.subplot(3, 3, 7)
    actions = ['NS-Through', 'EW-Through', 'N-Turn', 'E-Turn']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax7.bar(actions, eval_metrics['action_distribution'], color=colors, alpha=0.7)
    ax7.set_ylabel('Frequency', fontsize=11)
    ax7.set_title('Learned Policy: Action Distribution', fontsize=12, fontweight='bold')
    ax7.set_xticklabels(actions, rotation=45, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Performance Improvement Metrics
    ax8 = plt.subplot(3, 3, 8)
    
    # Calculate improvements
    reward_improvement = ((np.mean(eval_metrics['rewards']) - 
                          np.mean(baseline_metrics['rewards'])) / 
                         abs(np.mean(baseline_metrics['rewards'])) * 100)
    
    waiting_improvement = ((np.mean(baseline_metrics['waiting_times']) - 
                           np.mean(eval_metrics['waiting_times'])) / 
                          np.mean(baseline_metrics['waiting_times']) * 100)
    
    throughput_improvement = ((np.mean(eval_metrics['vehicles_served']) - 
                              np.mean(baseline_metrics['vehicles_served'])) / 
                             np.mean(baseline_metrics['vehicles_served']) * 100)
    
    improvements = [reward_improvement, waiting_improvement, throughput_improvement]
    metrics_names = ['Reward\nImprovement', 'Waiting Time\nReduction', 'Throughput\nIncrease']
    colors_imp = ['#2E86AB' if x > 0 else '#C73E1D' for x in improvements]
    
    bars = ax8.barh(metrics_names, improvements, color=colors_imp, alpha=0.7)
    ax8.set_xlabel('Improvement (%)', fontsize=11)
    ax8.set_title('SARSA(λ) vs Fixed-Timing: % Improvement', fontsize=12, fontweight='bold')
    ax8.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax8.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # 9. Average Queue Length During Training
    ax9 = plt.subplot(3, 3, 9)
    smoothed_queue = pd.Series(metrics['avg_queue_lengths']).rolling(window).mean()
    ax9.plot(smoothed_queue, linewidth=2, color='#6A994E', label='Avg Queue Length')
    ax9.set_xlabel('Episode', fontsize=11)
    ax9.set_ylabel('Average Queue Length', fontsize=11)
    ax9.set_title('Training Progress: Queue Management', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/sarsa_lambda_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: sarsa_lambda_results.png")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nSARSA(λ) Performance:")
    print(f"  Average Reward: {np.mean(eval_metrics['rewards']):.2f} ± {np.std(eval_metrics['rewards']):.2f}")
    print(f"  Average Waiting Time: {np.mean(eval_metrics['waiting_times']):.2f} ± {np.std(eval_metrics['waiting_times']):.2f}")
    print(f"  Average Vehicles Served: {np.mean(eval_metrics['vehicles_served']):.2f} ± {np.std(eval_metrics['vehicles_served']):.2f}")
    
    print(f"\nFixed-Timing Baseline:")
    print(f"  Average Reward: {np.mean(baseline_metrics['rewards']):.2f} ± {np.std(baseline_metrics['rewards']):.2f}")
    print(f"  Average Waiting Time: {np.mean(baseline_metrics['waiting_times']):.2f} ± {np.std(baseline_metrics['waiting_times']):.2f}")
    print(f"  Average Vehicles Served: {np.mean(baseline_metrics['vehicles_served']):.2f} ± {np.std(baseline_metrics['vehicles_served']):.2f}")
    
    print(f"\nPerformance Improvements:")
    print(f"  Reward Improvement: {reward_improvement:.2f}%")
    print(f"  Waiting Time Reduction: {waiting_improvement:.2f}%")
    print(f"  Throughput Increase: {throughput_improvement:.2f}%")
    print("="*70)


def visualize_learned_policy(agent, env):
    """
    Visualize the learned Q-values and policy.
    
    Args:
        agent: Trained SARSA(λ) agent
        env: Traffic environment
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample some states to visualize
    sample_states = []
    for n in range(5):
        for s in range(5):
            for e in range(5):
                for w in range(5):
                    if (n + s + e + w) <= 12:  # Only reasonable states
                        sample_states.append((n, s, e, w))
    
    # Randomly sample 20 states
    np.random.shuffle(sample_states)
    sample_states = sample_states[:20]
    
    # Extract Q-values
    q_values_matrix = []
    state_labels = []
    
    for state in sample_states:
        if state in agent.Q:
            q_values_matrix.append(agent.Q[state])
            state_labels.append(f"({state[0]},{state[1]},{state[2]},{state[3]})")
    
    q_values_matrix = np.array(q_values_matrix)
    
    # Plot 1: Q-values heatmap
    sns.heatmap(q_values_matrix.T, 
                xticklabels=state_labels,
                yticklabels=['NS-Through', 'EW-Through', 'N-Turn', 'E-Turn'],
                cmap='RdYlGn', center=0, annot=False, fmt='.1f',
                cbar_kws={'label': 'Q-value'},
                ax=axes[0])
    axes[0].set_xlabel('State (N,S,E,W queue lengths)', fontsize=11)
    axes[0].set_ylabel('Action', fontsize=11)
    axes[0].set_title('Learned Q-Values for Sample States', fontsize=12, fontweight='bold')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=90, ha='right', fontsize=8)
    
    # Plot 2: Optimal actions
    optimal_actions = np.argmax(q_values_matrix, axis=1)
    action_names = ['NS-Through', 'EW-Through', 'N-Turn', 'E-Turn']
    action_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    colors = [action_colors[a] for a in optimal_actions]
    axes[1].barh(range(len(state_labels)), [1]*len(state_labels), color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(state_labels)))
    axes[1].set_yticklabels(state_labels, fontsize=8)
    axes[1].set_xlabel('Optimal Action', fontsize=11)
    axes[1].set_ylabel('State (N,S,E,W queue lengths)', fontsize=11)
    axes[1].set_title('Learned Policy (Optimal Actions)', fontsize=12, fontweight='bold')
    axes[1].set_xticks([])
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=action_colors[i], label=action_names[i], alpha=0.7) 
                      for i in range(4)]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/sarsa_lambda_policy.png', dpi=300, bbox_inches='tight')
    print("\n✓ Policy visualization saved to: sarsa_lambda_policy.png")
    plt.show()


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" SARSA(λ) FOR INTELLIGENT TRAFFIC SIGNAL CONTROL")
    print("="*70)
    print("\nThis implementation demonstrates reinforcement learning")
    print("for adaptive traffic signal optimization at intersections.\n")
    
    # Train the agent
    print("Phase 1: Training SARSA(λ) Agent")
    print("-" * 70)
    agent, env, metrics = train_agent(num_episodes=1000, render_interval=200)
    
    print("\n✓ Training completed successfully!\n")
    
    # Evaluate the agent
    print("Phase 2: Evaluating Trained Agent")
    print("-" * 70)
    eval_metrics = evaluate_agent(agent, env, num_episodes=50)
    print("✓ Evaluation completed!\n")
    
    # Compare with baseline
    print("Phase 3: Comparing with Fixed-Timing Baseline")
    print("-" * 70)
    baseline_metrics = compare_with_fixed_timing(env, num_episodes=50)
    print("✓ Baseline comparison completed!\n")
    
    # Visualize results
    print("Phase 4: Generating Visualizations")
    print("-" * 70)
    visualize_results(metrics, eval_metrics, baseline_metrics)
    visualize_learned_policy(agent, env)
    
    print("\n" + "="*70)
    print(" EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nAll results have been saved and are ready for presentation.")
    
    return agent, env, metrics, eval_metrics, baseline_metrics


# Run the complete experiment
if __name__ == "__main__":
    agent, env, metrics, eval_metrics, baseline_metrics = main()
