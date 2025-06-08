# Policy Annealing

This repository implements a new class of a thermodynamics-inspired algorithms for reinforcement learning, together with experiments demonstrating its usability. Just as particles in nature generally prefer to occupy states/locations with lower total energy (there are more air particles down here compared to 100km away from earth), the algorithm proposed and implemented here enforces that action sequences that get high rewards have high total probabilities. Total probability is here defined as the product of probabilities of all actions taken: $p\_{total}=p_1 p_2 ... p_n$.

As a quick demo, here's a neural network trained using the Policy Annealing algorithm to take a moonlander to a safe stop. For training, see [Usage/Lunar Lander](#lunar-lander).

![Lunar lander is landing!](https://github.com/user-attachments/assets/9cecaade-91cd-4dbe-b2c1-19a632622d43)

## Installation

This project requires Python 3.10-3.12 to be installed. The current codebase does not utilize a GPU; all computations are run on the CPU.

Installing dependencies requires `swig`, a system-level dependency.

**On Linux:**

```bash
# Ubuntu/Debian/Mint
sudo apt install swig

# Arch Linux
sudo pacman -S swig
```

Once `swig` is installed, you can install the Python dependencies:

```bash
# Install Pytorch (CPU version)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
pip install -r requirements.txt
```

## Usage

Experiments can be run with the following command:

```bash
python scripts/main.py
```

Please see `--help` for setting the training environment and type/parameters, as well as whether to render the training progress. All metadata and renders will be stored in a `runs/[timestamp]/` directory.

Performance curves of different runs can be plotted and compared with:

```bash
python scripts/plot.py runs/[timestamp1]/experiment.json runs/[timestamp2]/experiment.json
```

Below is a list of environments together with hyperparameters and visualizations of the trained agents. **Validation mode**, used in some examples, means the agent will run deterministically, always choosing the action with the highest probability instead of sampling from the probability distribution.

### Cart Pole

The Cart Pole environment requires the agent to learn to balance a vertical pole by accelerating the cart left or right. A reward is given for each time step as long as the pole is vertical within 15 degrees and does not veer out of the image. There's a maximum time limit of 500 simulation steps.

Here's a very aggressive setup for training the CartPole agent. The training can be extremely fast if we're lucky, but is quite unstable and often does not converge.

```bash
python scripts/main.py --value-function direct --batch-size 4 --num-episode-batches 10 --learning-rate 0.005 --num-optim-steps 300 --env-name CartPole-v1 --render full
```

A relatively lucky from-scratch training:

https://github.com/user-attachments/assets/7df2edd7-20dc-49d8-97dc-27ac8d1c4313

Here's a much more stable setup that almost always converges:

```bash
python scripts/main.py --value-function grouped --env-name CartPole-v1 --num-episode-batches 10 --temp-start 0.5 --temp-end 0.5
```

The final solution seems more convincing:

https://github.com/user-attachments/assets/50dae70d-556a-4041-9a1d-7fb8ff77ad46

With validation mode on (always choosing the highest probability action), the solution gets so good that it's even boring:

https://github.com/user-attachments/assets/d861589b-2d54-40f7-b0c2-839c1039b4fb

### Lunar Lander

In this environment, the moonlander agent must learn to fire the three (one main, and two side) thrusters to successfully land on a platform. Reward is increased for being near the platform and landing, while it's reduced for thruster usage (fuel consumption), fast movement, and non-vertical orientation.

```bash
python scripts/main.py --env-name LunarLander-v3 --num-episode-batches 300 --clip-eps 2 --temp-start 0.1 --temp-end 0.1 --learning-rate 0.0005
```

To visualize a trained agent in validation mode, load its policy file:

```bash
# Replace [timestamp], [experiment_number] and [episode_batch_number] with directory names
python scripts/main.py --env-name LunarLander-v3 --validate --render full --load-model runs/[timestamp]/[experiment_number]/models/[episode_batch_number]/policy.pth
```

https://github.com/user-attachments/assets/4d831a23-83fe-4633-a66d-4175672a93fb

## Technical description

At a fixed temperature $T$, a particle's probability to be in a certain location varies according to the Boltzmann distribution:

```math
p \propto e^{-E/kT}
```

where

- $p$ is the probability of the particle being at a certain location
- $E$ is the energy of the particle at that location
- $k$ is the Boltzmann constant
- $T$ is the absolute temperature of the system

Essentially, particles are more likely to be in lower energy states, with the exact probability distribution depending on the temperature.

In a deep reinforcement learning setting, we want the neural network to have a high probability of emitting actions that lead to high rewards. If we substitute negative energy with reward, $R = -E$ (so that high reward equals low energy), and set the physical constant $k=1$, we get:

```math
p \propto e^{R/T} \Rightarrow p = c\ e^{R/T}
```

where $c$ is a normalization constant for a particular distribution.

Now, let's consider a full trajectory. The total probability of a trajectory is the product of the probabilities of all actions taken, and the total reward is the sum of rewards from all steps:

```math
p_{total} = p_0 ... p_n \quad \text{and} \quad R_{total} = R_0 + ... + R_n
```

This leads to the relationship:

```math
p_0 p_1 ... p_n = c\ e^{(R_0 + R_1 + ... + R_n) / T}
```

To avoid dealing with products of small probabilities and make the equation easier to work with, we apply the logarithm to both sides:

```math
\log(p_0) + ... + \log(p_n) = \log(c) + \frac{R_0 + ... + R_n}{T}
```

Using summation notation, this is:

```math
\sum_{i=0}^{n} \log(p_i) = \log(c) + \frac{1}{T}\sum_{i=0}^{n} R_i
```

Multiplying by $T$ and rearranging to isolate the constant term gives us:

```math
-T \cdot \log(c) = \sum_{i=0}^{n} R_i - T \sum_{i=0}^{n} \log(p_i)
```

We can combine the terms under a single summation. Let's define the "trajectory value" $V' = -T \cdot \log(c)$, which is a constant for a given deterministic environment and starting position. This gives us our core equation:

```math
\begin{equation}
\boxed{V' = \sum_{i=0}^{n} (R_i - T \cdot \log(p_i))}
\end{equation}
```

This equation is the basis for all variants of the policy annealing algorithm. If we can train a policy network to produce probabilities $p_i$ that satisfy this equation, we have learned an agent that has a high probability of achieving high rewards.

### Grouped Policy Annealing

The easiest way to enforce the above condition is to group environment runs by identical seed (starting conditions), so they should all have an identical trajectory value $V'$. We can then use gradient descent to optimize the neural network to produce probabilities that satisfy the equation.

Specifically, for the loss function, we note that the condition is satisfied when the computed values of all trajectories in a group are equal to their mean. We thus use a squared error loss:

```math
L = (V_{traj_A}' - \langle V' \rangle)^2 + (V_{traj_B}' - \langle V' \rangle)^2 + ...
```

where $\langle V' \rangle$ is the average of values across trajectories in the group, and each $V_{traj_X}'$ is computed using the rewards and action probabilities from that trajectory.

### Value Function Policy Annealing

To improve learning efficiency, we can enforce the boxed condition $(1)$ on every timestep. We can also introduce a discount factor $\gamma$ to prioritize more immediate rewards:

```math
V_j' = \sum_{i=0}^{k} \gamma^i (R_{j+i} - T \cdot \log(p_{j+i}))
```

with $0 < \gamma < 1$.

We can then define two neural networks: one for computing the value $V_i'$ and another for the policy that provides the action probabilities $p_i$. Both are optimized simultaneously by minimizing the squared error of the above equation.

Experiments show that this technique can be quite sample-efficient, however it may need a relatively large number of training steps on each batch of samples. This is because the value and policy networks learn together from one equation, but the policy network can learn wrong behavior while the value network has not yet converged.

### As a regularization method (Better Entropy)

**Note:** this is currently work in progress and can't be run from `scripts/main.py`, however there are some quick experiments [showing much better stability properties](#policy-annealing-seems-to-be-more-stable) compared to other regularization methods.

One problem with the methods described above is that they define their own loss functions, which aren't obviously similar to the Policy Gradients method. This means that Policy Annealing, in its previous forms, is hard to apply to state-of-the-art algorithms like PPO.

To remedy this, we can define an "entropy-regularized reward" (ERR) as:

```math
R_i' = R_i - T \cdot \log(p_i)
```

We can then see that our core equation is simply the sum of these new rewards:

```math
V_{i}' = R_{i}' + R_{i+1}' + ... + R_{i+k}'
```

Thus, we can treat these entropy-regularized rewards just as we would treat standard rewards in any RL method (for both value estimation and policy updates) and still, theoretically, arrive at the same Boltzmann distribution.

## Comparison to Entropy Bonus

Entropy bonus regularization is a common technique in modern deep reinforcement learning. The goal is to encourage exploration by adding the policy's entropy, scaled by a temperature parameter $\alpha$ (equivalent to T in our model), to the reward signal. The objective is to find a policy $\pi$ that maximizes the expectation of $R + \alpha \cdot H(\pi)$, where $H$ is the policy's entropy.

### Policy Annealing seems to be more stable

Both theoretical arguments and toy simulations show that Policy Annealing based algorithms have much better stability characteristics than Entropy Bonus.

For example, let's take an RL environment that has no input but three possible actions: $A$ (low reward), $B$ (low reward), and $C$ (high reward). Each episode only lasts one action.

The simulation below shows how the action probabilities change for three methods: Policy Gradients without regularization (blue), Policy Gradients with Entropy Bonus (red), and Policy Gradients with Policy Annealing Regularization (green).

https://github.com/user-attachments/assets/3c387fd8-7f02-49be-bbcb-fdeed156a2de

Note that Policy Annealing regularization allows the policy to become stationary at an optimal distribution, while the Entropy Bonus policy never truly settles. For Policy Annealing, at the optimal distribution, the rewards are fully compensated by the $-T \cdot \log(p)$ terms, resulting in zero advantage and a stationary policy. For Entropy Bonus, the loss function constantly fluctuates as it tries to balance different rewards with a uniform exploration drive, leading to an unstable final policy.

## Codebase

The current project is divided into the following parts:

```
- demos/                          # Code for some README visualizations
- lib/
    - anneal_*.py                 # Annealing logic implementations
    - loss_*.py                   # Loss function implementations
    - batch_environment.py        # Environments with grouped seeding logic
    - model.py                    # PyTorch model generator
    - sample.py                   # Sampling logic
    - tracking.py                 # Visualization and experiment tracking logic. You can safely ignore this file if you just want to understand the algorithm.
- scripts/
    - main.py                     # Run experiments
    - plot.py                     # Compare and visualize performance plots of experiments
- tests/
    - test_.py                    # Pytest tests
```

In addition, the following directory is generated while running the experiments:

```
- runs/
    - [timestamp]/
        - experiment.json                    # Experiment metadata + rewards. Feed this into scripts/plot.py
        - [experiment number]/               # If num_runs > 0
            - models/
                - [episode batch number]/
                    - policy.pth             # Policy network
                    - value.pth              # Value network (if applicable)
            training.mp4                     # Video of all training episodes
            first_observation_value.html     # Visualization of value network changes
```

### Coding conventions

The following coding conventions are used:

- Tensor channels are ordered as: `(num_groups, group_size, steps, num_actions/num_observations)`. Note that `batch_size = num_groups * group_size`.
- Variables whose type can't be inferred by mypy are explicitly typed (e.g. function signatures, empty list initializations)
- Ensure that linting and tests succeed: `pre-commit run -a && pytest`
