% 排队系统模拟与分析
% 基于马尔可夫链的多队列系统仿真

% ====================
% 1. 参数设置（可修改）
% ====================
NUM_QUEUES = 5;  % 队列数
MAX_QUEUE_LENGTH = 5;  % 每列最大人数
ARRIVAL_PROB = 0.8;  % 每单位时间有顾客到达的概率 λ
CHOOSE_SMART_PROB = 0.9;  % 顾客选择最短队列的概率 p
SERVICE_PROB = 0.07;  % 每列每单位时间服务前端顾客的概率 μ

% ================================
% 2. 枚举所有状态 & 建立索引映射
% ================================
% 生成所有可能的状态组合
states = {};
index = 1;
for q1 = 0:MAX_QUEUE_LENGTH
    for q2 = 0:MAX_QUEUE_LENGTH
        for q3 = 0:MAX_QUEUE_LENGTH
            for q4 = 0:MAX_QUEUE_LENGTH
                for q5 = 0:MAX_QUEUE_LENGTH
                    states{index} = [q1 q2 q3 q4 q5];
                    index = index + 1;
                end
            end
        end
    end
end

N = length(states);  % 总状态数 = (MAX_QUEUE_LENGTH+1)^NUM_QUEUES = 6^5 = 7776

% ========================
% 辅助函数
% ========================

% 获取最短队列的索引
function shortest_idxs = get_shortest_queues(state)
    min_len = min(state);
    shortest_idxs = find(state == min_len);
end

% 在指定队列添加顾客
function next_state = add_customer(state, queue_idx)
    if state(queue_idx) < MAX_QUEUE_LENGTH
        next_state = state;
        next_state(queue_idx) = next_state(queue_idx) + 1;
    else
        next_state = [];  % 队列已满
    end
end

% 服务函数：返回所有可能的服务结果及其概率
function [next_states, probs] = serve(state)
    masks = dec2bin(0:2^NUM_QUEUES-1) - '0';  % 生成所有可能的服务组合
    next_states = cell(size(masks, 1), 1);
    probs = zeros(size(masks, 1), 1);
    
    for i = 1:size(masks, 1)
        mask = masks(i, :);
        prob = 1.0;
        next_state = state;
        for j = 1:NUM_QUEUES
            if mask(j) == 1
                prob = prob * SERVICE_PROB;
                if next_state(j) > 0
                    next_state(j) = next_state(j) - 1;
                end
            else
                prob = prob * (1 - SERVICE_PROB);
            end
        end
        next_states{i} = next_state;
        probs(i) = prob;
    end
end

% =====================================
% 3. 构造稀疏转移矩阵 P（N×N）
% =====================================
% 创建状态到索引的映射函数
state_to_index = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:N
    state_str = sprintf('%d,%d,%d,%d,%d', states{i});
    state_to_index(state_str) = i;
end

% P(j, i) 表示从状态 i 转到状态 j 的概率
P = sparse(N, N);

for i = 1:N
    state = states{i};
    [service_results, service_probs] = serve(state);
    
    % 对每种服务结果
    for k = 1:length(service_results)
        s1 = service_results{k};
        p1 = service_probs(k);
        
        % ── 情况 1：本步没有顾客到达（概率 = 1 - ARRIVAL_PROB）
        s1_str = sprintf('%d,%d,%d,%d,%d', s1);
        idx_s1 = state_to_index(s1_str);
        P(idx_s1, i) = P(idx_s1, i) + p1 * (1 - ARRIVAL_PROB);
        
        % ── 情况 2：本步有顾客到达（概率 = ARRIVAL_PROB）
        shortest_idxs = get_shortest_queues(s1);
        num_short = length(shortest_idxs);
        num_other = NUM_QUEUES - num_short;
        
        for q_idx = 1:NUM_QUEUES
            if s1(q_idx) == MAX_QUEUE_LENGTH
                continue;  % 该列已满
            end
            
            if ismember(q_idx, shortest_idxs)
                prob_choose = ARRIVAL_PROB * (CHOOSE_SMART_PROB / num_short);
            else
                if num_other > 0
                    prob_choose = ARRIVAL_PROB * ((1 - CHOOSE_SMART_PROB) / num_other);
                else
                    prob_choose = 0.0;
                end
            end
            
            next_state = add_customer(s1, q_idx);
            if ~isempty(next_state)
                next_state_str = sprintf('%d,%d,%d,%d,%d', next_state);
                j = state_to_index(next_state_str);
                P(j, i) = P(j, i) + p1 * prob_choose;
            end
        end
    end
end

% =============================================
% 4. 幂迭代计算稳态分布 π：π = π P，π 长度 = N
% =============================================
% 初始化为均匀分布
pi = ones(N, 1) / N;

max_iters = 5000;
tol = 1e-12;

for it = 1:max_iters
    pi_next = P' * pi;  % 注意MATLAB中矩阵乘法是左乘
    pi_next_sum = sum(pi_next);
    if pi_next_sum <= 0
        error('迭代过程中概率向量和为 0，可能参数选择不当。');
    end
    pi_next = pi_next / pi_next_sum;  % 归一化
    diff = norm(pi_next - pi, 1);
    pi = pi_next;
    if diff < tol
        fprintf('幂迭代在迭代 %d 步后收敛，L1 差值 = %.2e\n', it, diff);
        break;
    end
end

if it == max_iters
    fprintf('警告：迭代达到了最大步数 %d，但尚未完全收敛，最后 L1 差值 = %.2e\n', max_iters, diff);
end

% =============================
% 5. 计算并输出性能指标
% =============================
% 5.1 系统中平均总排队人数
queue_lengths = cellfun(@sum, states);
average_total_queue = dot(pi, queue_lengths);
fprintf('\n平均系统总排队人数（5 列之和）≈ %.4f\n', average_total_queue);

% 5.2 平均每列排队人数
avg_each_column = zeros(NUM_QUEUES, 1);
for col_idx = 1:NUM_QUEUES
    col_length_vector = cellfun(@(x) x(col_idx), states);
    avg_each_column(col_idx) = dot(pi, col_length_vector);
    fprintf('第 %d 列 平均排队人数 ≈ %.4f\n', col_idx, avg_each_column(col_idx));
end

% 5.3 输出概率最大的状态
top_k = 10;
[~, top_indices] = sort(pi, 'descend');
top_indices = top_indices(1:top_k);
fprintf('\n概率最高的 %d 个状态：\n', top_k);
for idx = 1:top_k
    state_str = sprintf('(%d,%d,%d,%d,%d)', states{top_indices(idx)});
    fprintf('  %s  :  %.6e\n', state_str, pi(top_indices(idx)));
end

% =============================
% 6. 绘制转移矩阵热力图
% =============================
fprintf('\n开始绘制转移矩阵热力图...\n');

% 降采样参数
downsample_factor = 4;
target_size = floor(N / downsample_factor);

% 创建降采样后的矩阵
P_downsampled = zeros(target_size);

% 分块处理并降采样
chunk_size = 500;
for i = 1:chunk_size:target_size
    for j = 1:chunk_size:target_size
        i_end = min(i + chunk_size - 1, target_size);
        j_end = min(j + chunk_size - 1, target_size);
        
        % 计算原始矩阵中对应的范围
        i_start_orig = (i-1) * downsample_factor + 1;
        i_end_orig = min(i_end * downsample_factor, N);
        j_start_orig = (j-1) * downsample_factor + 1;
        j_end_orig = min(j_end * downsample_factor, N);
        
        % 提取矩阵块
        P_chunk = full(P(i_start_orig:i_end_orig, j_start_orig:j_end_orig));
        
        % 降采样
        for di = 1:downsample_factor:size(P_chunk, 1)
            for dj = 1:downsample_factor:size(P_chunk, 2)
                di_end = min(di + downsample_factor - 1, size(P_chunk, 1));
                dj_end = min(dj + downsample_factor - 1, size(P_chunk, 2));
                block = P_chunk(di:di_end, dj:dj_end);
                if ~isempty(block)
                    P_downsampled(floor(i_start_orig/downsample_factor) + floor((di-1)/downsample_factor), ...
                                floor(j_start_orig/downsample_factor) + floor((dj-1)/downsample_factor)) = mean(block(:));
                end
            end
        end
        
        fprintf('处理进度: %.1f%%\r', ...
            ((floor(i/chunk_size)*target_size + floor(j/chunk_size)) / (ceil(target_size/chunk_size))^2 * 100));
    end
end

fprintf('\n降采样完成，开始绘制...\n');

% 创建热力图
figure('Position', [100 100 1000 1000]);
P_log = log10(P_downsampled + 1e-10);
imagesc(P_log);
colormap('viridis');
colorbar('Label', 'Log_{10}(Transition Probability + 1e-10)');
title(sprintf('Complete Transition Matrix Heatmap (%d×%d downsampled to %d×%d)', ...
    N, N, target_size, target_size));
xlabel('From State Index (downsampled)');
ylabel('To State Index (downsampled)');
grid on;

% 设置刻度
tick_step = floor(target_size / 10);
tick_indices = 1:tick_step:target_size;
tick_labels = cellstr(num2str((tick_indices-1) * downsample_factor));
set(gca, 'XTick', tick_indices, 'XTickLabel', tick_labels);
set(gca, 'YTick', tick_indices, 'YTickLabel', tick_labels);

saveas(gcf, 'complete_transition_matrix_heatmap.fig');
saveas(gcf, 'complete_transition_matrix_heatmap.png');
fprintf('完整转移矩阵热力图已保存\n');

% 输出统计信息
fprintf('\n转移矩阵统计信息：\n');
fprintf('矩阵大小: %d×%d\n', N, N);
fprintf('非零元素数量: %d\n', nnz(P));
fprintf('稀疏性: %.4f%%\n', nnz(P)/(N*N)*100);
fprintf('最大转移概率: %.6f\n', max(P(:)));
fprintf('最小非零转移概率: %.6f\n', min(P(P>0)));

% =============================
% 7. 计算顾客等待时间分布
% =============================
avg_service_time = 1.0 / SERVICE_PROB;

% 计算选择概率分布函数
function probs = compute_choice_probabilities(state, p_smart)
    shortest_idxs = get_shortest_queues(state);
    num_short = length(shortest_idxs);
    num_other = NUM_QUEUES - num_short;
    probs = zeros(1, NUM_QUEUES);
    
    for i = 1:NUM_QUEUES
        if ismember(i, shortest_idxs)
            probs(i) = p_smart / num_short;
        elseif num_other > 0
            probs(i) = (1 - p_smart) / num_other;
        end
    end
end

% 计算等待时间分布
waiting_time_prob = containers.Map('KeyType', 'double', 'ValueType', 'double');

for idx = 1:N
    state = states{idx};
    pi_s = pi(idx);
    choice_probs = compute_choice_probabilities(state, CHOOSE_SMART_PROB);
    
    for q_idx = 1:NUM_QUEUES
        if choice_probs(q_idx) == 0 || state(q_idx) >= MAX_QUEUE_LENGTH
            continue;
        end
        
        wait_time = state(q_idx) * avg_service_time;
        if ~isKey(waiting_time_prob, wait_time)
            waiting_time_prob(wait_time) = 0;
        end
        waiting_time_prob(wait_time) = waiting_time_prob(wait_time) + pi_s * choice_probs(q_idx);
    end
end

% 归一化
wait_times = cell2mat(keys(waiting_time_prob));
probs = cell2mat(values(waiting_time_prob));
total_prob = sum(probs);
for k = wait_times
    waiting_time_prob(k) = waiting_time_prob(k) / total_prob;
end

% 排序并输出等待时间分布
[wait_times_sorted, idx] = sort(wait_times);
probs_sorted = probs(idx);

fprintf('\n等待时间（单位：平均服务时间倍数）及对应概率：\n');
for i = 1:length(wait_times_sorted)
    fprintf('等待时间 %.2f ：概率 %.6f\n', wait_times_sorted(i), probs_sorted(i));
end

% ======================
% 8. 绘制稳态分布π的分析图
% ======================
fprintf('\n开始绘制稳态分布π的分析图...\n');

figure('Position', [100 100 1200 800]);

% 子图1：π值的直方图
subplot(2, 2, 1);
pi_nonzero = pi(pi > 0);
histogram(log10(pi_nonzero), 50, 'FaceColor', [0.7 0.9 1], 'EdgeColor', 'black');
xlabel('Log_{10}(π)');
ylabel('Frequency');
title('Distribution of Log Steady-State Probabilities');
grid on;

% 子图2：π值按索引排序的图
subplot(2, 2, 2);
sorted_pi = sort(pi, 'descend');
plot(1:length(sorted_pi), sorted_pi, 'b-', 'LineWidth', 1);
xlabel('State Rank');
ylabel('Steady-State Probability π');
title('Sorted Steady-State Probabilities');
set(gca, 'YScale', 'log');
grid on;

% 子图3：最高概率的前50个状态
subplot(2, 2, 3);
bar(1:50, sorted_pi(1:50), 'FaceColor', [1 0.6 0.4]);
xlabel('State Rank (Top 50)');
ylabel('Probability π');
title('Top 50 States by Probability');
set(gca, 'YScale', 'log');
grid on;

% 子图4：π值的累积分布
subplot(2, 2, 4);
cumulative_prob = cumsum(sorted_pi);
plot(1:length(cumulative_prob), cumulative_prob, 'g-', 'LineWidth', 2);
hold on;
plot([1 length(cumulative_prob)], [0.5 0.5], 'r--', 'LineWidth', 1);
plot([1 length(cumulative_prob)], [0.9 0.9], '--', 'Color', [1 0.5 0], 'LineWidth', 1);
plot([1 length(cumulative_prob)], [0.99 0.99], '--', 'Color', [0.5 0 0.5], 'LineWidth', 1);
xlabel('Number of Top States');
ylabel('Cumulative Probability');
title('Cumulative Probability Distribution');
grid on;
legend('Cumulative', '50%', '90%', '99%', 'Location', 'southeast');

saveas(gcf, 'pi_distribution_analysis.fig');
saveas(gcf, 'pi_distribution_analysis.png');

% 统计分析输出
fprintf('\n稳态分布π的统计分析：\n');
fprintf('π向量总和: %.10f\n', sum(pi));
fprintf('π最大值: %.6e\n', max(pi));
fprintf('π最小值: %.6e\n', min(pi));
fprintf('π平均值: %.6e\n', mean(pi));
fprintf('π标准差: %.6e\n', std(pi));

% 计算概率集中度
states_for_50 = find(cumulative_prob >= 0.5, 1);
states_for_90 = find(cumulative_prob >= 0.9, 1);
states_for_99 = find(cumulative_prob >= 0.99, 1);

fprintf('\n概率集中度分析：\n');
fprintf('前 %d 个状态占总概率的50%%\n', states_for_50);
fprintf('前 %d 个状态占总概率的90%%\n', states_for_90);
fprintf('前 %d 个状态占总概率的99%%\n', states_for_99);
fprintf('总状态数: %d\n', N);

% ======================
% 9. 绘制等待时间概率分布
% ======================
figure('Position', [100 100 800 500]);
bar(wait_times_sorted, probs_sorted, avg_service_time * 0.8, ...
    'FaceColor', [0.7 0.9 1], 'EdgeColor', 'black');
xlabel('Waiting Time (multiples of average service time)');
ylabel('Probability');
title('Customer Waiting Time Distribution');
grid on;
saveas(gcf, 'waiting_time_distribution.fig');
saveas(gcf, 'waiting_time_distribution.png');