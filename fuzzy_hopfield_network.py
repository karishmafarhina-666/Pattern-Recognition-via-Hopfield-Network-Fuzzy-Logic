import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. FUZZY LOGIC NOISE GENERATOR
# ==========================================
def fuzzy_noise_level(interference_score):
    """
    Fuzzy system mapping interference (0-10) to a noise percentage.
    Rules:
    - IF Interference is Low, THEN Noise is Low (10%)
    - IF Interference is Medium, THEN Noise is Medium (25%)
    - IF Interference is High, THEN Noise is High (45%)
    """
    # Fuzzification (Membership Functions)
    low = max(0, min(1, (4 - interference_score) / 4))
    med = max(0, min((interference_score - 2) / 3, (8 - interference_score) / 3))
    high = max(0, min(1, (interference_score - 6) / 4))
    
    # Defuzzification (Weighted Average / Centroid)
    noise_low, noise_med, noise_high = 0.10, 0.25, 0.45
    total_weight = low + med + high
    
    if total_weight == 0:
        return 0.0
    
    crisp_noise = (low * noise_low + med * noise_med + high * noise_high) / total_weight
    return crisp_noise

def apply_fuzzy_noise(pattern, interference_score):
    noise_ratio = fuzzy_noise_level(interference_score)
    noisy_pattern = np.copy(pattern)
    num_flips = int(noise_ratio * len(pattern))
    
    flip_indices = random.sample(range(len(pattern)), num_flips)
    for idx in flip_indices:
        noisy_pattern[idx] *= -1  # Flip 1 to -1 or -1 to 1
        
    return noisy_pattern, noise_ratio

# ==========================================
# 2. HOPFIELD NETWORK (ASSOCIATIVE MEMORY)
# ==========================================
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        
    def train(self, patterns):
        """Hebbian Learning Rule to store patterns"""
        for p in patterns:
            self.weights += np.outer(p, p)
        # No self-connections allowed in Hopfield networks
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size

    def recall(self, pattern, steps=5):
        """Asynchronous update to retrieve memorized pattern"""
        state = np.copy(pattern)
        for _ in range(steps):
            for i in range(self.size):
                # Calculate weighted sum of inputs
                raw_input = np.dot(self.weights[i], state)
                # Apply step activation function (Sign)
                state[i] = 1 if raw_input >= 0 else -1
        return state

# ==========================================
# 3. DATASET & EXECUTION
# ==========================================
# 5x5 Binary Patterns (-1 for background, 1 for foreground)
pattern_C = np.array([
    1, 1, 1, 1, 1,
    1,-1,-1,-1,-1,
    1,-1,-1,-1,-1,
    1,-1,-1,-1,-1,
    1, 1, 1, 1, 1
])

pattern_S = np.array([
    1, 1, 1, 1, 1,
    1,-1,-1,-1,-1,
    1, 1, 1, 1, 1,
   -1,-1,-1,-1, 1,
    1, 1, 1, 1, 1
])

# Initialize and train network
patterns = [pattern_C, pattern_S]
hopfield = HopfieldNetwork(size=25)
hopfield.train(patterns)

# Test with Fuzzy Noise (Interference Score: 7.5 out of 10)
interference = 7.5
noisy_S, applied_noise_ratio = apply_fuzzy_noise(pattern_S, interference)

# Recall pattern from noisy input
recalled_S = hopfield.recall(noisy_S)

# ==========================================
# 4. VISUALIZATION
# ==========================================
fig, ax = plt.subplots(1, 3, figsize=(10, 4))

ax[0].imshow(pattern_S.reshape(5,5), cmap='Blues')
ax[0].set_title("Original Pattern ('S')")

ax[1].imshow(noisy_S.reshape(5,5), cmap='Reds')
ax[1].set_title(f"Fuzzy Corrupted\n(Interference: {interference}/10 | Noise: {applied_noise_ratio:.0%})")

ax[2].imshow(recalled_S.reshape(5,5), cmap='Greens')
ax[2].set_title("Hopfield Recalled Pattern")

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

plt.tight_layout()
plt.savefig("hopfield_fuzzy_result.png")
plt.show()
