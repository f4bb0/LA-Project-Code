import numpy as np
from itertools import product
from scipy.sparse import lil_matrix, csr_matrix

# ====================
# 1. Parameter Settings (Modifiable)
# ====================
NUM_QUEUES = 5  # Number of queues
MAX_QUEUE_LENGTH = 5  # Maximum number of customers per queue
ARRIVAL_PROB = 0.8  # Probability of customer arrival per time unit λ
CHOOSE_SMART_PROB = 0.9  # Probability of choosing shortest queue p
SERVICE_PROB = 0.07  # Probability of serving front customer per time unit μ

# ================================
# 2. Enumerate All States & Build Index Mapping
# ================================
# Each state is a tuple of length NUM_QUEUES (q1,q2,...,q5), where qi ∈ {0,1,...,MAX_QUEUE_LENGTH}
states = list(product(range(MAX_QUEUE_LENGTH + 1), repeat=NUM_QUEUES))
N = len(states)  # Total number of states = (MAX_QUEUE_LENGTH+1)^NUM_QUEUES = 6^5 = 7776

state_to_index = {state: idx for idx, state in enumerate(states)}


def get_shortest_queues(state):
    """Returns a list of indices of all shortest queues in the current state."""
    min_len = min(state)
    return [i for i, q in enumerate(state) if q == min_len]


def add_customer(state, queue_idx):
    """Add a customer to queue_idx in state, return None if queue is full."""
    if state[queue_idx] < MAX_QUEUE_LENGTH:
        lst = list(state)
        lst[queue_idx] += 1
        return tuple(lst)
    return None  # Queue is full


def serve(state):
    """
    Returns a list of possible service results and their probabilities for a given state.
    Each queue independently serves a customer with probability SERVICE_PROB (if queue is not empty).
    mask[i] == 1 means queue i completes one service (if qi>0, decrease by 1).
    mask[i] == 0 means queue i does not serve.
    Returns [(next_state_tuple, prob), ...]
    """
    options = []
    # Iterate through all 2^NUM_QUEUES combinations of "whether each queue serves successfully"
    for mask in product([0, 1], repeat=NUM_QUEUES):
        prob = 1.0
        next_state = list(state)
        for i in range(NUM_QUEUES):
            if mask[i] == 1:
                prob *= SERVICE_PROB
                if next_state[i] > 0:
                    next_state[i] -= 1
            else:
                prob *= (1 - SERVICE_PROB)
        options.append((tuple(next_state), prob))
    return options


# =====================================
# 3. Construct Sparse Transition Matrix P (N×N)
# =====================================
# P[j, i] represents the probability of transitioning from state i to state j
P = lil_matrix((N, N))

for i_state, state in enumerate(states):
    service_results = serve(state)
    # For each service result s1 and its probability p1
    for s1, p1 in service_results:
        # ── Case 1: No customer arrives in this step (probability = 1 - ARRIVAL_PROB)
        # Maintain service state s1 → s1
        idx_s1 = state_to_index[s1]
        P[idx_s1, i_state] += p1 * (1 - ARRIVAL_PROB)

        # ── Case 2: A customer arrives in this step (probability = ARRIVAL_PROB)
        #   Customer chooses one of the shortest queues with probability p;
        #   chooses a non-shortest queue uniformly with probability (1-p).
        shortest_idxs = get_shortest_queues(s1)
        num_short = len(shortest_idxs)
        num_other = NUM_QUEUES - num_short

        # If shortest_queues = NUM_QUEUES (i.e., all queues have same length),
        # treated as "randomly picking one from shortest_idxs"
        for q_idx in range(NUM_QUEUES):
            if s1[q_idx] == MAX_QUEUE_LENGTH:
                # Skip if this queue is full, cannot add customer
                continue

            if q_idx in shortest_idxs:
                prob_choose = ARRIVAL_PROB * (CHOOSE_SMART_PROB / num_short)
            else:
                if num_other > 0:
                    prob_choose = ARRIVAL_PROB * (
                        (1 - CHOOSE_SMART_PROB) / num_other)
                else:
                    prob_choose = 0.0

            next_state = add_customer(s1, q_idx)
            if next_state is not None:
                j = state_to_index[next_state]
                P[j, i_state] += p1 * prob_choose
            # If next_state is None, the queue is full, and the customer is rejected → probability remains unchanged

# Convert to CSR format (for efficient multiplication)
P = csr_matrix(P)

# =============================================
# 4. Power Iteration to Calculate Steady State Distribution π: π = π P, π length = N
# =============================================
# Initialize with uniform distribution
pi = np.ones(N, dtype=float) / N

max_iters = 5000
tol = 1e-12

for it in range(max_iters):
    pi_next = P.dot(pi)
    pi_next_sum = pi_next.sum()
    if pi_next_sum <= 0:
        raise RuntimeError("Probability vector sum is 0 during iteration, parameters may be inappropriate.")
    pi_next /= pi_next_sum  # Normalization
    diff = np.linalg.norm(pi_next - pi, ord=1)
    pi = pi_next
    if diff < tol:
        print(f"Power iteration converged after {it+1} steps, L1 difference = {diff:.2e}")
        break
else:
    print(f"Warning: Reached maximum iterations {max_iters} without full convergence, final L1 difference = {diff:.2e}")

# Final π is the approximate steady-state distribution
# ------------------------------------------

# =============================
# 5. Calculate and Output Performance Metrics
# =============================
# 5.1 Average total number of customers in system E[∑ qi]
queue_lengths = np.array([sum(state)
                          for state in states])  # For each state, calculate sum of all 5 queues
average_total_queue = float(np.dot(pi, queue_lengths))
print(f"\nAverage total number of customers in system (sum of 5 queues) ≈ {average_total_queue:.4f}")

# 5.2 Average number of customers per queue
#    Calculate: E[qi] = ∑_{state} pi[state] * state[i], then output per queue
avg_each_column = []
for col_idx in range(NUM_QUEUES):
    col_length_vector = np.array([state[col_idx] for state in states])
    avg_each_column.append(float(np.dot(pi, col_length_vector)))

for i, val in enumerate(avg_each_column, start=1):
    print(f"Queue {i} average number of customers ≈ {val:.4f}")

# 5.3 Output states with highest probabilities to see common queue configurations
top_k = 10
top_indices = np.argsort(pi)[-top_k:][::-1]
print(f"\nTop {top_k} states with highest probabilities (format: (q1,q2,q3,q4,q5) : π)")
for idx in top_indices:
    print(f"  {states[idx]}  :  {pi[idx]:.6e}")

import matplotlib.pyplot as plt

# =============================
# 6. Draw Complete 7776×7776 Transition Matrix Heatmap (Memory Optimized Version)
# =============================
print("Starting to draw complete transition matrix heatmap...")

# Memory optimization strategy: block processing + resolution reduction
chunk_size = 500  # Process 500x500 blocks at a time
downsample_factor = 4  # Downsample factor, compress 7776 to about 1944
target_size = N // downsample_factor

# Create downsampled matrix
P_downsampled = np.zeros((target_size, target_size), dtype=np.float32)

print(f"Original matrix size: {N}×{N}, After downsampling: {target_size}×{target_size}")

# Process blocks and downsample
for i in range(0, target_size, chunk_size // downsample_factor):
    for j in range(0, target_size, chunk_size // downsample_factor):
        # Calculate corresponding range in original matrix
        i_start = i * downsample_factor
        i_end = min((i + chunk_size // downsample_factor) * downsample_factor, N)
        j_start = j * downsample_factor
        j_end = min((j + chunk_size // downsample_factor) * downsample_factor, N)
        
        # Extract sparse matrix block and convert to dense matrix
        P_chunk = P[i_start:i_end, j_start:j_end].toarray()
        
        # Downsample: take average of each downsample_factor×downsample_factor block
        for di in range(0, P_chunk.shape[0], downsample_factor):
            for dj in range(0, P_chunk.shape[1], downsample_factor):
                block = P_chunk[di:di+downsample_factor, dj:dj+downsample_factor]
                if block.size > 0:
                    P_downsampled[i + di//downsample_factor, j + dj//downsample_factor] = np.mean(block)
        
        # Clean up memory
        del P_chunk
        
        print(f"Processing progress: {((i//chunk_size*downsample_factor + j//chunk_size*downsample_factor + 1) / (target_size//chunk_size*downsample_factor)**2 * 100):.1f}%", end='\r')

print("\nDownsampling complete, starting to plot...")

# Create heatmap
plt.figure(figsize=(10, 10))
# Use log scale to better display non-zero elements in sparse matrix
P_log = np.log10(P_downsampled + 1e-10)  # Add small value to avoid log(0)
im = plt.imshow(P_log, cmap='viridis', aspect='auto', interpolation='nearest')

plt.colorbar(im, label='Log₁₀(Transition Probability + 1e-10)')
plt.title(f'Complete Transition Matrix Heatmap ({N}×{N} downsampled to {target_size}×{target_size})')
plt.xlabel('From State Index (downsampled)')
plt.ylabel('To State Index (downsampled)')

# Add grid lines for better visualization
plt.grid(True, alpha=0.3)

# Set ticks
tick_step = target_size // 10
tick_indices = range(0, target_size, tick_step)
tick_labels = [str(i * downsample_factor) for i in tick_indices]
plt.xticks(tick_indices, tick_labels)
plt.yticks(tick_indices, tick_labels)

plt.tight_layout()

# Save image with lower DPI to save memory
plt.savefig('complete_transition_matrix_heatmap.png', dpi=150, bbox_inches='tight')
print("Complete transition matrix heatmap has been saved to: complete_transition_matrix_heatmap.png")
plt.close()

# Clean up memory
del P_downsampled, P_log

print("\nTransition matrix statistics:")
print(f"Matrix size: {N}×{N}")
print(f"Number of non-zero elements: {P.nnz}")
print(f"Sparsity: {P.nnz / (N*N) * 100:.4f}%")
print(f"Maximum transition probability: {P.data.max():.6f}")
print(f"Minimum non-zero transition probability: {P.data[P.data > 0].min():.6f}")

# =============================
# 7. Calculate Customer Waiting Time Distribution
# =============================
# Assume average service time = 1 / SERVICE_PROB
avg_service_time = 1.0 / SERVICE_PROB


# Compute probability distribution function for queue selection given state and strategy
def compute_choice_probabilities(state, p_smart=CHOOSE_SMART_PROB):
    shortest_idxs = get_shortest_queues(state)
    num_short = len(shortest_idxs)
    num_other = NUM_QUEUES - num_short
    prob_list = np.zeros(NUM_QUEUES)
    for i in range(NUM_QUEUES):
        if i in shortest_idxs:
            prob_list[i] = p_smart / num_short
        else:
            if num_other > 0:
                prob_list[i] = (1 - p_smart) / num_other
            else:
                prob_list[i] = 0
    return prob_list


# Waiting time → probability dictionary, waiting time in multiples of average service time
waiting_time_prob = {}

for idx, state in enumerate(states):
    pi_s = pi[idx]
    choice_probs = compute_choice_probabilities(state)
    for q_idx, prob_q in enumerate(choice_probs):
        if prob_q == 0:
            continue
        # If queue is full, new customers cannot join, skip
        if state[q_idx] >= MAX_QUEUE_LENGTH:
            continue
        wait_time = state[q_idx] * avg_service_time
        waiting_time_prob[wait_time] = waiting_time_prob.get(wait_time,
                                                             0) + pi_s * prob_q

# Normalize (theoretically should sum to 1, but numerical errors may exist)
total_prob = sum(waiting_time_prob.values())
for k in waiting_time_prob:
    waiting_time_prob[k] /= total_prob

# Sort waiting times
wait_times_sorted = sorted(waiting_time_prob.keys())
probs_sorted = [waiting_time_prob[t] for t in wait_times_sorted]

# Output waiting time probability distribution
print("\nWaiting time (in multiples of average service time) and corresponding probabilities:")
for t, p in zip(wait_times_sorted, probs_sorted):
    print(f"Waiting time {t:.2f} : probability {p:.6f}")

# ======================
# 8. Draw Steady-State Distribution π Analysis Plots
# ======================
print("\nStarting to draw steady-state distribution π analysis plots...")

# 8.1 Draw π value histogram distribution
plt.figure(figsize=(12, 8))

# Subplot 1: π value histogram
plt.subplot(2, 2, 1)
pi_nonzero = pi[pi > 0]  # Consider only non-zero probabilities
plt.hist(np.log10(pi_nonzero), bins=50, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('Log₁₀(π)')
plt.ylabel('Frequency')
plt.title('Distribution of Log Steady-State Probabilities')
plt.grid(axis='y', alpha=0.3)

# Subplot 2: π value sorted by index
plt.subplot(2, 2, 2)
sorted_pi = np.sort(pi)[::-1]  # Sort from high to low
plt.plot(range(len(sorted_pi)), sorted_pi, 'b-', linewidth=1)
plt.xlabel('State Rank')
plt.ylabel('Steady-State Probability π')
plt.title('Sorted Steady-State Probabilities')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Subplot 3: Top 50 states with highest probabilities
plt.subplot(2, 2, 3)
top_50_indices = np.argsort(pi)[-50:][::-1]
top_50_probs = pi[top_50_indices]
plt.bar(range(50), top_50_probs, color='coral', alpha=0.8)
plt.xlabel('State Rank (Top 50)')
plt.ylabel('Probability π')
plt.title('Top 50 States by Probability')
plt.yscale('log')
plt.grid(axis='y', alpha=0.3)

# Subplot 4: Cumulative distribution of π values
plt.subplot(2, 2, 4)
cumulative_prob = np.cumsum(sorted_pi)
plt.plot(range(len(cumulative_prob)), cumulative_prob, 'g-', linewidth=2)
plt.xlabel('Number of Top States')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Probability Distribution')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50%')
plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%')
plt.axhline(y=0.99, color='purple', linestyle='--', alpha=0.7, label='99%')
plt.legend()

plt.tight_layout()
plt.savefig('pi_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8.2 Statistical Analysis
print("\nStatistical analysis of steady-state distribution π:")
print(f"Sum of π vector: {pi.sum():.10f}")
print(f"Maximum π value: {pi.max():.6e}")
print(f"Minimum π value: {pi.min():.6e}")
print(f"Mean π value: {pi.mean():.6e}")
print(f"Standard deviation of π: {pi.std():.6e}")

# Calculate how many highest probability states account for 50%, 90%, 99% of total probability
sorted_indices = np.argsort(pi)[::-1]
sorted_probs = pi[sorted_indices]
cumsum = np.cumsum(sorted_probs)

states_for_50 = np.searchsorted(cumsum, 0.5) + 1
states_for_90 = np.searchsorted(cumsum, 0.9) + 1
states_for_99 = np.searchsorted(cumsum, 0.99) + 1

print(f"\nProbability concentration analysis:")
print(f"First {states_for_50} states account for 50% of total probability")
print(f"First {states_for_90} states account for 90% of total probability")
print(f"First {states_for_99} states account for 99% of total probability")
print(f"Total number of states: {N}")

print(f"\nProbability distribution saved to: pi_distribution_analysis.png")

# ======================
# 9. Plot waiting time probability distribution
# ======================
plt.figure(figsize=(8, 5))
plt.bar(wait_times_sorted,
        probs_sorted,
        width=avg_service_time * 0.8,
        color='skyblue',
        edgecolor='black')
plt.xlabel('Waiting Time (multiples of average service time)')
plt.ylabel('Probability')
plt.title('Customer Waiting Time Distribution')
plt.grid(axis='y')
plt.show()
