# Policy Annealing

This repository implements an new class of a thermodynamics inspired algorithms for reinforcement learning, together with experiments demonstrating it's usability. Just as particles in nature generally prefer to occupy states/locations with lower total energy (there's more air particles down here compared to 100km away from earth), the algorithm enforces that action sequences that get high rewards have high total probabilities. Total probability is here defined as the product of probabilities of all actions taken: $p\_{total}=p_1 p_2 ... p_n$.

<figure>
    <img src="https://github.com/user-attachments/assets/9cecaade-91cd-4dbe-b2c1-19a632622d43" alt="">
    <figcaption>
        A neural network trained the Policy Annealing algorithm taking a lunar lander to a safe stop. For training, see "Usage".
    </figcaption>
</figure>

## Installation

This project requires Python 3.10+ to be installed.

On Linux based systems, installing other dependencies is relatively easy:

```bash
# Install Swig on the system level: Ubuntu/Mint
sudo apt install swig

# Install Swig on the system level: Arch Linux
sudo pacman -S swig

# Install Pytorch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
pip install -r requirements.txt
```

## Usage

Experiments can be run with the following command:

```bash
python scripts/main.py
```

Please see `--help` for setting the training environment and type/parameters, as well as whether to render the training progress. All metadata and renders will be stored in `runs/[timestamp]/` directory.

Performance curves of different runs can be plotted and compared with

```bash
python scripts/plot.py runs/[timestamp1]/experiment.json runs/[timestamp2]/experiment.json
```

Below is a list of environments together with hyperparameters and visualizations of the trained agents.

### Cart Pole

Cart Pole environment requires the agent to learn to balance a vertical pole by accelerating the cart left or right. Reward is given for each time step as long as the pole is vertical within 15 degrees accuracy and does not veer out of the image.

Here's a very aggressive setup for training the CartPole agent. The training can be extremely fast if we're lucky, but is quite unstable and often does not converge.

```bash
python scripts/main.py --value-function direct --batch-size 4 --num-episode-batches 10 --learning-rate 0.005 --num-optim-steps 300 --env-name CartPole-v1 --render full
```

A relatively lucky run visualized:

https://github.com/user-attachments/assets/7df2edd7-20dc-49d8-97dc-27ac8d1c4313

Here's a much more stable setup that almost always converges:

```bash
python scripts/main.py --value-function grouped --env-name CartPole-v1 --num-episode-batches 10 --temp-start 0.5 --temp-end 0.5
```

The final solution seems more convincing:

https://github.com/user-attachments/assets/50dae70d-556a-4041-9a1d-7fb8ff77ad46

With validation mode on, it's even getting boring:

https://github.com/user-attachments/assets/d861589b-2d54-40f7-b0c2-839c1039b4fb

### Lunar Lander

In this environment, the moonlander agent must learn to fire the three (one main, and two side) thrusters to successfully land on a platform. Reward is increased for being near the platform and landing (either leg contact or full landing), while it's reduced for thruster usage (fuel consumption), fast movement and non-vertical orientation.

```bash
python scripts/main.py --env-name LunarLander-v3 --num-episode-batches 300 --clip-eps 2 --temp-start 0.1 --temp-end 0.1 --learning-rate 0.0005
```

To visualize the final trained agent in validation mode (always choose the highest probability action):

```bash
python scripts/plot.py --env-name LunarLander-v3 --validate --render full --batch-size 4
```

https://github.com/user-attachments/assets/4d831a23-83fe-4633-a66d-4175672a93fb

## Technical description

At a fixed temperature $T$, a particle's probability to be in a certain location varies according to the Boltzmann distribution:

```math
p \propto e^{-E/kT}
```

where

- $p$ is the probability of the particle being at a certain location
- $e \approx 2.718$
- $E$ is the energy of the particle due to being at that location
- $k$ is the Boltzmann constant
- $T$ is the absolute temperature of the system
- $\propto$ signifies that $p$ varies proportionally to the right hand side

Essentially, particles are always more concentrated to lower energy states/locations, with the probability distribution depending on the temperature.

In a deep reinforcement learning setting, we want the neural network to have a high probability of emitting actions that lead to high rewards and a low probability of actions that lead to low rewards. If we substitute negative energy with reward $R = -E$ (high reward = high likelihood = low energy), and set Boltzmann constant $k=1$ ($k$ is just some physical constant which we don't need in the algorithm), we get:

```math
p \propto e^{R/T} \Rightarrow p = c\ e^{R/T}
```

where $c$ is a constant for a particular distribution.

Now let's keep in mind that the total probability of a trajectory is the product of the probabilities of all actions taken on all steps:

```math
p = p_1 p_2 ... p_n
```

And the reward is the sum of all rewards at all steps:

```math
R = R_1 + R_2 + ... + R_n
```

We find that

```math
p_1 p_2 ... p_n = c\ e^{(R_1 + R_2 + ... + R_n) / T}
```

Now, the probabilities on the left side and the exponents on the right side can become unmanageably small. Let's apply the logarithm to both sides:

```math
log(p_1) + ... + log(p_n) = log(c) + \frac{R_1 + ... + R_n}{T}
```

which we can rearrange as

```math
\begin{equation}
\boxed{V' = -log(c) = (R_1 - T \cdot log(p_1)) + ... + (R_n - T \cdot log(p_n))}
\end{equation}
```

Where $V' = -log(c)$ (let's call it "value") is a constant specific given we have a deterministic environment and starting position. This equation is the bases for all variants of the policy algorithm. If we can enforce the probabilities to satisfy this equation closely, we've learned an agent that has a high probability of achieving high rewards.

### Grouped Policy Annealing

The easiest way to enforce the above condition is to group environment runs by identical seed (starting conditions) so they should have an identical value. We can then use gradient descent to optimize the neural network to produce probabilities that satisfy the equations.

Specifically, for the loss function, we note that the condition is satisfied, when all computed values of the trajectories are equal to the mean of those values. We thus use our beloved squaring to compute the loss:

```math
L = (V_{traj_A} - \langle V \rangle)^2 + (V_{traj_B} - \langle V \rangle)^2 + ...
```

where $\langle V \rangle$ is the average of values

```math
\langle V \rangle = \frac{V_{traj_A} + V_{traj_B} + ...}{N_{trajectories}}
```

and individual $V\_{traj_X}$ are computed using rewards and probabilities of taken actions of the trajectory $X$.

### Value Function Policy Annealing

To improve the efficiency of the learning, we'd ideally enforce the boxed condition $(1)$ on every timestep. As a further improvement, this gives us to give a higher priority to more immediate rewards by using a discount factor:

```math
V_i' = (R_{i} - T \cdot log(p_{i})) + \gamma \cdot (R_{i+1} - T \cdot log(p_{i+1})) + ... + \gamma^k \cdot (R_{i+k} - T \cdot log(p_{i+k}))
```

with

```math
0 < \gamma < 1
```

We can then define two neural networks, one for computing $V'$ and another for the policy that assigns the probabilities of actions. Both are optimized at once by minimizing the squared error of the above equation.

Experiments show that this technique can be quite sample efficient, however it needs a relatively large number of training steps on each batch of samples. This is because the value and policy networks learn together on one equation, but the policy network often learns wrong behaviour while the value network has not converged yet.

## Comparison to related algorithms

### Simulated Annealing

While Simulated Annealing (SA) has some superficial similarities with Policy Annealing (PA), the task solved is completely different. SA attempts to find an optimal solution by choosing solutions based on Boltzmann probability distribution. PA on the other hand, trains a function (neural network) that generates optimal solutions.

### Entropy Bonus regularization

Entropy bonus regularization is a common technique in modern deep reinforcement learning. The goal is to encourage exploration by adding the policy's entropy, scaled by a temperature parameter $\alpha$ (equivalent to T in our model), to the reward signal. The objective is to find a policy $\pi$ that maximizes the expectation of $R + \alpha * H(\pi)$, where $H$ is the policy's entropy.

#### Shared theoretical optimum

The core mathematical form $R - T * log(p)$ is central to both Policy Annealing and entropy-regularized RL. In fact, the policy that maximizes the entropy-augmented objective $E[R] + T*H(π)$ is the Boltzmann distribution, $p \propto e^{R/T}$. This is the same distribution that Policy Annealing aims to produce by enforcing its "constant value" condition. Therefore, both algorithms are striving towards the same optimal policy for a given temperature.

#### Policy Annealing is more stable (at least in theory)

Policy annealing finds and stays at the optimal distribution, while Entropy bonus may cause the policy to never settle. As an example, let's assume our RL environment has no input, but three possible actions: $A$ (fixed high reward), $B$ (fixed low reward) and $C$ (fixed low reward), and each episode only lasts one action (essentially 3-armed bandit with deterministic rewards). For policy annealing, given a temperature T, we can solve for the probability distribution, and find that no matter what action was taken, there is zero gradient (the solution is stationary). On the other hand, for entropy bonus, if we take action $B$, the loss function would attempt to equalize the probability between $A$ and $C$, thus never really resulting in a stationary policy.

## Codebase

The current project is divided into the following parts:

```
- lib/
    - anneal_*.py                 # Annealing logic implementations (grouped and value function implementations)
    - loss_*.py                   # Loss function implementations (grouped and value function implementations)
    - batch_environment.py        # Environments with grouped seeding logic support
    - model.py                    # Pytorch model generator for policy and value networks
    - sample.py                   # Sampling logic for batch_environment.py
    - tracking.py                 # Visualization and experiment tracking logic. Most of it is LLM generated and it's pretty large, but you can safely ignore it if you just want to understand the algorithm.
- scripts/
    - main.py                     # Run experiments
    - plot.py                     # Compare and visualize performance plots of experiments using files generated by main.py
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
            first_observation_value.html     # Visualization of the changes in value network over episode batches (if applicable)
            value_over_steps/
                [episode batch number].html  # Visualization of the changes in value network over optimization steps of a single
```

### Coding conventions

The following coding conventions are used:

- Tensor's channels are ordered like this: `(num_groups, group_size, steps, num_actions/num_observations)`. Note that `batch_size = num_groups * group_size`.
- Variables whose type can't be inferred by mypy are explicitly typed: function signatures, empty list initializations, etc.
- Ensure that linting and tests succeed: `pre-commit run -a && pytest`
