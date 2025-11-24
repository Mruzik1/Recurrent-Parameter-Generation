# Experimental RPG Training: The "Neural Policy Factory"

This document outlines three advanced Reinforcement Learning (RL) experiments designed to demonstrate the RPG model's ability to generate specialized control policies on demand.

**Target Hardware:**
- **Training:** NVIDIA A100 40G (Massive parallel RL training).
- **Inference:** NVIDIA RTX 2050 (Fast generation of tiny policy networks).

**Performance Note:**
These experiments target "Tiny Policy" architectures (~5k - 20k parameters).
-   **Token Count:** With `dim_per_token=1024`, these networks represent sequences of only **5 to 20 tokens**.
-   **Generation Speed:** Even on an RTX 2050, generating such short sequences with Mamba/Diffusion is **sub-second**, allowing for near-interactive demos.

---

## Experiment 1: The "Morphology Adapter" (Zero-Shot Sim-to-Sim)

Can the model generate a brain that perfectly fits a specific body and environment?

### Concept
Train an RPG model to generate walking policies for a **Bipedal Walker** where the robot's physical structure and the world's gravity vary. The model must learn the relationship between *body shape* and *gait*.

### Target Architecture
-   **Environment:** `BipedalWalker-v3` (Box2D).
-   **Type:** Continuous Control Policy (MLP).
-   **Structure:** `[24 (Sensors) -> 64 -> 64 -> 4 (Joint Torques)]`.
-   **Parameter Count:** ~6k.

### Data Collection Strategy
1.  **Domain Randomization:** Create a training loop that randomizes:
    -   **Leg Length:** $L \in [0.5, 2.0]$
    -   **Leg Width:** $W \in [0.1, 0.5]$
    -   **Gravity:** $g \in [-5.0, -20.0]$
    -   **Ground Friction:** $\mu \in [0.1, 5.0]$
2.  **Massive Training:** Use a vectorized RL trainer (like `cleanrl` or `rllib`) to train 10,000 agents. Each agent is trained to convergence on *one specific configuration*.
3.  **Storage:** `walker_L{L}_W{W}_g{g}_mu{mu}.pth`.

### RPG Configuration
-   **Condition Vector:** `[Leg_Length, Leg_Width, Gravity, Friction]`.
-   **Inference Prompt:** "Long legs, Low gravity, Slippery floor".

### The Demo
A simulation of the walker.
-   **Sliders:** "Leg Length", "Gravity".
-   **Effect:** As you increase leg length, the RPG generates a new policy. You see the walker's gait shift from a high-frequency shuffle (short legs) to a slow, loping stride (long legs) to maintain balance.

---

## Experiment 2: The "Skill Composer" (Multi-Objective Control)

Can the model generate a policy that balances conflicting objectives based on user preference?

### Concept
Train an RPG model to generate a piloting policy for a **Lunar Lander** or **Drone** that satisfies a complex, weighted set of behavioral constraints.

### Target Architecture
-   **Environment:** `LunarLanderContinuous-v2` or a custom Drone env.
-   **Type:** Continuous Control Policy.
-   **Structure:** `[8 (State) -> 128 -> 128 -> 2 (Main/Side Engines)]`.
-   **Parameter Count:** ~18k.

### Data Collection Strategy
1.  **Reward Shaping:** Define a parameterized reward function $R = w_1 \cdot R_{landing} + w_2 \cdot R_{fuel} + w_3 \cdot R_{time} + w_4 \cdot R_{smoothness}$.
2.  **Sampling:** Sample 10,000 weight vectors $\mathbf{w} = [w_1, w_2, w_3, w_4]$ from a Dirichlet distribution (so they sum to 1).
3.  **Training:** Train an agent for each weight vector.
    -   Agent A might learn to crash-land quickly (High $w_{time}$).
    -   Agent B might hover gently and land softly (High $w_{smoothness}$).
4.  **Storage:** `lander_weights_{w1}_{w2}_{w3}_{w4}.pth`.

### RPG Configuration
-   **Condition Vector:** `[w_landing, w_fuel, w_time, w_smoothness]`.
-   **Inference Prompt:** "Maximize smoothness, ignore time" vs "Emergency landing, save fuel".

### The Demo
A landing simulation.
-   **Controls:** A "Behavior Mixer" (like an audio equalizer).
-   **Effect:** Boosting "Smoothness" generates a pilot that makes gentle corrections. Boosting "Speed" generates a pilot that uses full thrust for a suicide burn. This shows the model interpolating between *strategies*.

---

## Experiment 3: The "Adversarial Counter-Play" (Competitive Gaming)

Can the model generate a strategy specifically designed to defeat a known opponent?

### Concept
Train an RPG model to generate a policy for **Slime Volleyball** (or a simple Sumo game) conditioned on the *opponent's physical stats and playstyle*.

### Target Architecture
-   **Environment:** `SlimeVolley-v0`.
-   **Type:** Competitive Policy.
-   **Structure:** `[12 (State) -> 128 -> 128 -> 3 (Actions)]`.
-   **Parameter Count:** ~18k.

### Data Collection Strategy
1.  **Opponent Pool:** Create a pool of scripted or pre-trained "Opponent" agents with varying attributes:
    -   **Aggression:** Probability of spiking vs. defending.
    -   **Reaction Time:** Delay in frames before moving.
    -   **Jump Height:** Physical attribute modifier.
2.  **Training:** Train a "Hero" agent to beat *each* specific Opponent configuration.
3.  **Storage:** `hero_vs_agg{agg}_react{react}_jump{jump}.pth`.

### RPG Configuration
-   **Condition Vector:** `[Opponent_Aggression, Opponent_Reaction, Opponent_Jump]`.
-   **Inference Prompt:** "Opponent is highly aggressive and jumps high".

### The Demo
You play against the AI.
-   **Setup:** You adjust sliders that represent *your* playstyle (or the AI estimates them).
-   **Effect:** If you set "Opponent Jump" to High, the RPG generates a policy that keeps the ball low (under your block). If you set "Opponent Reaction" to Slow, it generates a policy that plays fast spikes. It generates the *perfect counter-strategy* for the given parameters.


---

## Experiment 2: The "Split-Personality Agent" (Reinforcement Learning Policies)

This experiment demonstrates the unique capability of RPG to generate **behavior**, not just static data. Standard diffusion models generate images; this generates *agents that act*.

### Concept
Train an RPG model to generate the policy network (brain) for a simple control task (e.g., **LunarLander** or a **2D Walker**). The condition controls the agent's "personality" or trade-offs.

### Target Architecture
-   **Type:** Policy MLP (State $\to$ Action)
-   **Structure:** `[8 (State inputs) -> 64 -> 64 -> 4 (Action logits)]`
-   **Parameter Count:** ~4.6k.

### Data Collection Strategy
1.  **Environment:** OpenAI Gym `LunarLander-v2`.
2.  **Sampling:** Define a "Reward Landscape" based on two parameters:
    -   **Frugality:** Penalty for using engines (Fuel efficiency).
    -   **Urgency:** Penalty for time taken (Speed).
3.  **Training (A100):**
    -   Use a fast RL algorithm like PPO or ES (Evolution Strategies).
    -   Train 5,000 agents, each with a different mix of `(Frugality, Urgency)` in their reward function.
    -   Save the converged policy weights.
4.  **Storage:** `agent_f{frugality}_u{urgency}.pth`.

### RPG Configuration
-   **Condition Vector:** `[frugality, urgency]`.
-   **Inference:**
    -   Prompt: "High Frugality, Low Urgency" $\to$ Generates a cautious agent that barely uses fuel.
    -   Prompt: "Low Frugality, High Urgency" $\to$ Generates a "suicide burn" agent that lands fast and hard.

### Visualization (The Demo)
A live window showing the Lunar Lander game.
-   **Sliders:** "Fuel Saving" and "Haste".
-   **Action:** As the lander falls, moving the sliders instantly swaps its "brain". You can watch it switch from a cautious descent to an aggressive maneuver in mid-air.
-   **Why it fits:** It proves the model understands the manifold of *optimal control strategies*.

---

## Experiment 3: The "Infinite Resolution Glyph" (Implicit Neural Representations)

This experiment highlights the benefit of generating **weights** (functions) rather than **pixels**.

### Concept
Train an RPG model to generate a **Signed Distance Function (SDF)** network. This network takes an $(x, y)$ coordinate and outputs the distance to the nearest edge of a shape. This allows for rendering shapes at *infinite resolution*.

### Target Architecture
-   **Type:** SIREN (Sinusoidal Representation Network) or ReLU MLP with Positional Encoding.
-   **Structure:** `[2 (x, y) -> 128 -> 128 -> 128 -> 1 (distance)]`
-   **Parameter Count:** ~33k.

### Data Collection Strategy
1.  **Task:** Represent 2D procedural shapes (e.g., "Superformulas" or Gears).
2.  **Sampling:** Generate shapes based on parameters:
    -   **N-Symmetry:** Number of points/teeth (3=Triangle, 5=Star, 50=Circle).
    -   **Spikiness:** How sharp the corners are.
    -   **Thickness:** Hollow vs. Filled.
3.  **Training (A100):**
    -   For each shape parameter set, generate a ground truth SDF image.
    -   Train an MLP to overfit this single shape (map $x,y \to d$).
    -   Train 5,000 such MLPs.
4.  **Storage:** `glyph_n{n}_s{spike}_t{thick}.pth`.

### RPG Configuration
-   **Condition Vector:** `[symmetry, spikiness, thickness]`.
-   **Inference:**
    -   Prompt: "5 points, High Spikiness" $\to$ Generates a Star.
    -   Prompt: "50 points, Low Spikiness" $\to$ Generates a Circle.

### Visualization (The Demo)
A rendering window showing a 2D shape.
-   **Interaction:** You can zoom in infinitely. The edges never get pixelated because the RPG generates the *mathematical function* for the shape, not an image.
-   **Morphing:** Sliders change the number of points. The shape morphs smoothly from a Triangle to a Square to a Star to a Gear.
-   **Why it fits:** It demonstrates that the RPG has learned the *topology* of geometry. It's a "Neural CAD" system.

1.  **Dataset Generation (A100):**
    -   Use `joblib` or `multiprocessing` to train these tiny MLPs in parallel. You can likely train hundreds per minute on an A100.
    -   Aim for ~10,000 checkpoints for robust manifold learning.

2.  **RPG Model Sizing (for RTX 2050 Inference):**
    -   Since the target MLPs are small, the sequence length (number of tokens) will be short.
    -   **Config:**
        -   `dim_per_token`: 1024 (Smaller than the default 8192).
        -   `d_model`: 512 or 768.
        -   `num_layers`: 4-6.
    -   This reduced size will easily fit in the 4GB VRAM of an RTX 2050 and allow for very fast inference.

3.  **Code Adjustments:**
    -   You will need to create a new Dataset class in `dataset/register.py` that inherits from `ConditionalDataset`.
    -   Override `_extract_condition(self, index)` to parse floats from the filename and return a `torch.FloatTensor`.
