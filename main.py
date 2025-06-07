import numpy as np
from itertools import product
from scipy.sparse import lil_matrix, csr_matrix

# ====================
# 1. 参数设置（可修改）
# ====================
NUM_QUEUES = 5  # 队列数
MAX_QUEUE_LENGTH = 5  # 每列最大人数
ARRIVAL_PROB = 0.8  # 每单位时间有顾客到达的概率 λ
CHOOSE_SMART_PROB = 0.9  # 顾客选择最短队列的概率 p
SERVICE_PROB = 0.07  # 每列每单位时间服务前端顾客的概率 μ

# ================================
# 2. 枚举所有状态 & 建立索引映射
# ================================
# 每个状态是一个长度为 NUM_QUEUES 的元组 (q1,q2,...,q5)，其中 qi ∈ {0,1,...,MAX_QUEUE_LENGTH}
states = list(product(range(MAX_QUEUE_LENGTH + 1), repeat=NUM_QUEUES))
N = len(states)  # 总状态数 = (MAX_QUEUE_LENGTH+1)^NUM_QUEUES = 6^5 = 7776

state_to_index = {state: idx for idx, state in enumerate(states)}


def get_shortest_queues(state):
    """返回当前状态下最短队列的所有下标列表。"""
    min_len = min(state)
    return [i for i, q in enumerate(state) if q == min_len]


def add_customer(state, queue_idx):
    """在 state 的 queue_idx 列加入一名顾客，如果已满则返回 None。"""
    if state[queue_idx] < MAX_QUEUE_LENGTH:
        lst = list(state)
        lst[queue_idx] += 1
        return tuple(lst)
    return None  # 队列已满


def serve(state):
    """
    返回一个列表，表示在状态 state 下可能的“服务结果”及其概率。
    每列独立以 SERVICE_PROB 服完一名顾客（若该队列非空），否则不服务。
    mask[i] == 1 表示第 i 列发生一次服务（若 qi>0 则减一）。
    mask[i] == 0 表示第 i 列不服务。
    返回 [(next_state_tuple, prob), ...]
    """
    options = []
    # 遍历所有 2^NUM_QUEUES 种“各列是否服务成功”的组合
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
# 3. 构造稀疏转移矩阵 P（N×N）
# =====================================
# P[j, i] 表示从状态 i 转到状态 j 的概率
P = lil_matrix((N, N))

for i_state, state in enumerate(states):
    service_results = serve(state)
    # 对于每一种服务结果 s1，及其概率 p1
    for s1, p1 in service_results:
        # ── 情况 1：本步**没有顾客到达**（概率 = 1 - ARRIVAL_PROB）
        # 保持服务后状态 s1 → s1
        idx_s1 = state_to_index[s1]
        P[idx_s1, i_state] += p1 * (1 - ARRIVAL_PROB)

        # ── 情况 2：本步“有顾客到达”（概率 = ARRIVAL_PROB）
        #   顾客以概率 p 选择所有最短队列中的一个；以 (1-p) 均匀选一个非最短队列。
        shortest_idxs = get_shortest_queues(s1)
        num_short = len(shortest_idxs)
        num_other = NUM_QUEUES - num_short

        # 如果最短队列 = NUM_QUEUES（也就是所有队列都一样短），下文算作“在 shortest_idxs 中随机挑一个”。
        for q_idx in range(NUM_QUEUES):
            if s1[q_idx] == MAX_QUEUE_LENGTH:
                # 如果该列已满，就无法加入顾客，跳过
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
            # 如果 next_state 是 None，说明该列满了，顾客被拒绝 → 概率留空

# 转为 CSR 格式（便于乘法运算）
P = csr_matrix(P)

# =============================================
# 4. 幂迭代计算稳态分布 π：π = π P，π 长度 = N
# =============================================
# 初始化为均匀分布
pi = np.ones(N, dtype=float) / N

max_iters = 5000
tol = 1e-12

for it in range(max_iters):
    pi_next = P.dot(pi)
    pi_next_sum = pi_next.sum()
    if pi_next_sum <= 0:
        raise RuntimeError("迭代过程中概率向量和为 0，可能参数选择不当。")
    pi_next /= pi_next_sum  # 归一化
    diff = np.linalg.norm(pi_next - pi, ord=1)
    pi = pi_next
    if diff < tol:
        print(f"幂迭代在迭代 {it+1} 步后收敛，L1 差值 = {diff:.2e}")
        break
else:
    print(f"警告：迭代达到了最大步数 {max_iters}，但尚未完全收敛，最后 L1 差值 = {diff:.2e}")

# 最终 π 即为近似稳态分布
# ------------------------------------------

# =============================
# 5. 计算并输出一些性能指标
# =============================
# 5.1 系统中平均总排队人数 E[∑ qi]
queue_lengths = np.array([sum(state)
                          for state in states])  # 对每个状态，计算该状态下 5 列之和
average_total_queue = float(np.dot(pi, queue_lengths))
print(f"\n平均系统总排队人数（5 列之和）≈ {average_total_queue:.4f}")

# 5.2 平均每列排队人数
#    计算：E[qi] = ∑_{state} pi[state] * state[i]，然后取平均或逐列输出
avg_each_column = []
for col_idx in range(NUM_QUEUES):
    col_length_vector = np.array([state[col_idx] for state in states])
    avg_each_column.append(float(np.dot(pi, col_length_vector)))

for i, val in enumerate(avg_each_column, start=1):
    print(f"第 {i} 列 平均排队人数 ≈ {val:.4f}")

# 5.3 输出概率最大的若干个状态，看看常见队列配置
top_k = 10
top_indices = np.argsort(pi)[-top_k:][::-1]
print(f"\n概率最高的 {top_k} 个状态（格式： (q1,q2,q3,q4,q5) : π）")
for idx in top_indices:
    print(f"  {states[idx]}  :  {pi[idx]:.6e}")

import matplotlib.pyplot as plt

# =============================
# 6. 绘制完整7776×7776转移矩阵热力图（内存优化版本）
# =============================
print("开始绘制完整转移矩阵热力图...")

# 内存优化策略：分块处理 + 降低分辨率
chunk_size = 500  # 每次处理500x500的块
downsample_factor = 4  # 降采样因子，将7776压缩到约1944
target_size = N // downsample_factor

# 创建降采样后的矩阵
P_downsampled = np.zeros((target_size, target_size), dtype=np.float32)

print(f"原始矩阵大小: {N}×{N}, 降采样后: {target_size}×{target_size}")

# 分块处理并降采样
for i in range(0, target_size, chunk_size // downsample_factor):
    for j in range(0, target_size, chunk_size // downsample_factor):
        # 计算原始矩阵中对应的范围
        i_start = i * downsample_factor
        i_end = min((i + chunk_size // downsample_factor) * downsample_factor, N)
        j_start = j * downsample_factor
        j_end = min((j + chunk_size // downsample_factor) * downsample_factor, N)
        
        # 提取稀疏矩阵块并转为密集矩阵
        P_chunk = P[i_start:i_end, j_start:j_end].toarray()
        
        # 降采样：取每个downsample_factor×downsample_factor块的平均值
        for di in range(0, P_chunk.shape[0], downsample_factor):
            for dj in range(0, P_chunk.shape[1], downsample_factor):
                block = P_chunk[di:di+downsample_factor, dj:dj+downsample_factor]
                if block.size > 0:
                    P_downsampled[i + di//downsample_factor, j + dj//downsample_factor] = np.mean(block)
        
        # 清理内存
        del P_chunk
        
        print(f"处理进度: {((i//chunk_size*downsample_factor + j//chunk_size*downsample_factor + 1) / (target_size//chunk_size*downsample_factor)**2 * 100):.1f}%", end='\r')

print("\n降采样完成，开始绘制...")

# 创建热力图
plt.figure(figsize=(10, 10))
# 使用对数刻度来更好地显示稀疏矩阵的非零元素
P_log = np.log10(P_downsampled + 1e-10)  # 添加小值避免log(0)
im = plt.imshow(P_log, cmap='viridis', aspect='auto', interpolation='nearest')

plt.colorbar(im, label='Log₁₀(Transition Probability + 1e-10)')
plt.title(f'Complete Transition Matrix Heatmap ({N}×{N} downsampled to {target_size}×{target_size})')
plt.xlabel('From State Index (downsampled)')
plt.ylabel('To State Index (downsampled)')

# 添加网格线帮助可视化
plt.grid(True, alpha=0.3)

# 设置刻度
tick_step = target_size // 10
tick_indices = range(0, target_size, tick_step)
tick_labels = [str(i * downsample_factor) for i in tick_indices]
plt.xticks(tick_indices, tick_labels)
plt.yticks(tick_indices, tick_labels)

plt.tight_layout()

# 保存图像，使用较低DPI以节省内存
plt.savefig('complete_transition_matrix_heatmap.png', dpi=150, bbox_inches='tight')
print(f"完整转移矩阵热力图已保存到文件: complete_transition_matrix_heatmap.png")
plt.close()

# 清理内存
del P_downsampled, P_log

print(f"\n转移矩阵统计信息：")
print(f"矩阵大小: {N}×{N}")
print(f"非零元素数量: {P.nnz}")
print(f"稀疏性: {P.nnz / (N*N) * 100:.4f}%")
print(f"最大转移概率: {P.data.max():.6f}")
print(f"最小非零转移概率: {P.data[P.data > 0].min():.6f}")

# =============================
# 7. 计算顾客等待时间分布
# =============================
# 假设平均服务时间 = 1 / SERVICE_PROB
avg_service_time = 1.0 / SERVICE_PROB


# 计算顾客到达时选择某队列的概率分布函数，给定状态和选择策略
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


# 等待时间 → 概率 字典，等待时间单位为平均服务时间的倍数
waiting_time_prob = {}

for idx, state in enumerate(states):
    pi_s = pi[idx]
    choice_probs = compute_choice_probabilities(state)
    for q_idx, prob_q in enumerate(choice_probs):
        if prob_q == 0:
            continue
        # 如果该队列满了，新顾客不能加入，跳过
        if state[q_idx] >= MAX_QUEUE_LENGTH:
            continue
        wait_time = state[q_idx] * avg_service_time
        waiting_time_prob[wait_time] = waiting_time_prob.get(wait_time,
                                                             0) + pi_s * prob_q

# 归一化（理论上应该已经是1，但数值误差可能存在）
total_prob = sum(waiting_time_prob.values())
for k in waiting_time_prob:
    waiting_time_prob[k] /= total_prob

# 排序等待时间
wait_times_sorted = sorted(waiting_time_prob.keys())
probs_sorted = [waiting_time_prob[t] for t in wait_times_sorted]

# 输出部分等待时间概率分布
print("\n等待时间（单位：平均服务时间倍数）及对应概率：")
for t, p in zip(wait_times_sorted, probs_sorted):
    print(f"等待时间 {t:.2f} ：概率 {p:.6f}")

# ======================
# 8. 绘制稳态分布π的分布图
# ======================
print("\n开始绘制稳态分布π的分布图...")

# 8.1 绘制π值的直方图分布
plt.figure(figsize=(12, 8))

# 子图1：π值的直方图
plt.subplot(2, 2, 1)
pi_nonzero = pi[pi > 0]  # 只考虑非零概率
plt.hist(np.log10(pi_nonzero), bins=50, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('Log₁₀(π)')
plt.ylabel('Frequency')
plt.title('Distribution of Log Steady-State Probabilities')
plt.grid(axis='y', alpha=0.3)

# 子图2：π值按索引排序的图
plt.subplot(2, 2, 2)
sorted_pi = np.sort(pi)[::-1]  # 从大到小排序
plt.plot(range(len(sorted_pi)), sorted_pi, 'b-', linewidth=1)
plt.xlabel('State Rank')
plt.ylabel('Steady-State Probability π')
plt.title('Sorted Steady-State Probabilities')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 子图3：最高概率的前50个状态
plt.subplot(2, 2, 3)
top_50_indices = np.argsort(pi)[-50:][::-1]
top_50_probs = pi[top_50_indices]
plt.bar(range(50), top_50_probs, color='coral', alpha=0.8)
plt.xlabel('State Rank (Top 50)')
plt.ylabel('Probability π')
plt.title('Top 50 States by Probability')
plt.yscale('log')
plt.grid(axis='y', alpha=0.3)

# 子图4：π值的累积分布
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

# 8.2 统计分析
print("\n稳态分布π的统计分析：")
print(f"π向量总和: {pi.sum():.10f}")
print(f"π最大值: {pi.max():.6e}")
print(f"π最小值: {pi.min():.6e}")
print(f"π平均值: {pi.mean():.6e}")
print(f"π标准差: {pi.std():.6e}")

# 计算占总概率50%、90%、99%需要多少个最高概率状态
sorted_indices = np.argsort(pi)[::-1]
sorted_probs = pi[sorted_indices]
cumsum = np.cumsum(sorted_probs)

states_for_50 = np.searchsorted(cumsum, 0.5) + 1
states_for_90 = np.searchsorted(cumsum, 0.9) + 1
states_for_99 = np.searchsorted(cumsum, 0.99) + 1

print(f"\n概率集中度分析：")
print(f"前 {states_for_50} 个状态占总概率的50%")
print(f"前 {states_for_90} 个状态占总概率的90%")
print(f"前 {states_for_99} 个状态占总概率的99%")
print(f"总状态数: {N}")

print(f"\n概率分布保存到: pi_distribution_analysis.png")

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
