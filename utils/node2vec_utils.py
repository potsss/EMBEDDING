import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from config import Config
import multiprocessing # 引入多进程库
import pickle
import os
import hashlib

# 顶层辅助函数，用于 defaultdict 的工厂
def _create_default_dict_int():
    return defaultdict(int)

def build_graph_from_sequences(user_sequences, directed=False):
    """
    从用户序列构建物品-物品图。
    图的节点是物品ID，边表示物品在序列中相邻出现。
    可以构建有向图或无向图，边可以带权重（共现频率）。

    Args:
        user_sequences (dict): {user_id: [item_id_1, item_id_2, ...]}
        directed (bool): 是否构建有向图。默认为 False (无向图)。

    Returns:
        defaultdict: 图的邻接表表示 {node: {neighbor: weight}}
    """
    print("Building item-item graph from user sequences...")
    graph = defaultdict(_create_default_dict_int) # 使用顶层辅助函数
    
    for user_id, sequence in tqdm(user_sequences.items(), desc="Processing user sequences for graph"):
        if not sequence or len(sequence) < 2:
            continue
        for i in range(len(sequence) - 1):
            current_item = sequence[i]
            next_item = sequence[i+1]
            
            if current_item == next_item: # 忽略自环，或根据需要处理
                continue

            graph[current_item][next_item] += 1
            if not directed:
                graph[next_item][current_item] += 1
                
    print(f"Graph built with {len(graph)} nodes.")
    return graph

def get_alias_edge(graph, src, dst, p, q, dst_neighbors_sorted, src_neighbors_set):
    """
    计算Node2Vec中从src到dst的转移概率的未归一化权重。
    Args:
        graph: 图的邻接表。
        src: 当前游走路径中的前一个节点 (t)。
        dst: 当前游走路径中的当前节点 (v)。
        p: Node2Vec的返回参数。
        q: Node2Vec的进出参数。
        dst_neighbors_sorted: 节点 dst 的已排序邻居列表。
        src_neighbors_set: 节点 src 的邻居集合。
    Returns:
        list: [(neighbor, unnormalized_probability), ...]
    """
    unnormalized_probs = []
    # dst是当前节点v，我们要决定下一个节点x
    for dst_neighbor in dst_neighbors_sorted: # 遍历v的所有邻居x
        edge_weight = graph[dst][dst_neighbor] # 获取v-x边的权重

        if dst_neighbor == src:  # 如果x是t (返回)
            unnormalized_probs.append((dst_neighbor, edge_weight / p))
        elif dst_neighbor in src_neighbors_set:  # 如果x和t直接相连 (d_tx = 1)
            unnormalized_probs.append((dst_neighbor, edge_weight))
        else:  # 如果x和t不直接相连 (d_tx = 2)
            unnormalized_probs.append((dst_neighbor, edge_weight / q))
            
    return unnormalized_probs

def alias_setup(probs):
    """
    计算用于高效采样的别名表。
    Args:
        probs: 概率列表。
    Returns:
        tuple: (J, q) J是别名表，q是概率表。
    """
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    """
    从别名表中进行一次采样。
    Args:
        J: 别名表。
        q: 概率表。
    Returns:
        int: 采样得到的索引。
    """
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

# 为多进程优化的辅助函数
def _calculate_alias_edge_for_worker(args):
    graph, t_node, v_node, p, q, v_node_neighbors_sorted, t_node_neighbors_set = args
    if not graph[v_node]: # v没有出度
        return (t_node, v_node), None

    unnormalized_trans_probs = get_alias_edge(graph, t_node, v_node, p, q, v_node_neighbors_sorted, t_node_neighbors_set)
    
    probs_val = [prob for neighbor, prob in unnormalized_trans_probs]
    neighbors_for_alias = [neighbor for neighbor, prob in unnormalized_trans_probs]

    norm_const = sum(probs_val)
    if norm_const > 0:
        normalized_probs = [float(prob_val)/norm_const for prob_val in probs_val]
        return (t_node, v_node), (alias_setup(normalized_probs), neighbors_for_alias)
    else:
        return (t_node, v_node), None

def generate_node2vec_walks_dynamic(graph, num_walks, walk_length, p, q, start_nodes=None):
    """
    动态计算版本的 Node2Vec 随机游走生成器。
    不预计算边转移概率，而是在游走时动态计算，大幅减少预处理时间。

    Args:
        graph: defaultdict, 图的邻接表 {node: {neighbor: weight}}。
        num_walks (int): 每个节点开始的游走次数。
        walk_length (int): 每次游走的长度。
        p (float): Node2Vec的返回参数。
        q (float): Node2Vec的进出参数。
        start_nodes (list, optional): 指定开始游走的节点列表。如果为None，则从所有节点开始。

    Returns:
        list: 包含所有生成的游走路径的列表，每个路径是一个节点ID列表。
    """
    print(f"生成Node2Vec随机游走 (动态计算版本) (num_walks={num_walks}, walk_length={walk_length}, p={p}, q={q})...")
    walks = []
    
    if start_nodes is None:
        nodes = list(graph.keys())
    else:
        nodes = start_nodes
        
    if not nodes:
        print("警告：图或起始节点中没有节点。不会生成随机游走。")
        return []

    # 为每个节点预计算排序的邻居列表和邻居集合
    print("预计算邻居信息...")
    precomputed_sorted_neighbors = {node: sorted(graph[node].keys()) for node in tqdm(graph.keys(), desc="排序邻居")}
    precomputed_neighbor_sets = {node: set(graph[node].keys()) for node in tqdm(graph.keys(), desc="邻居集合")}

    # 为每个节点预处理第一步转移概率（这个数量相对较少）
    alias_nodes = {}
    for node in tqdm(nodes, desc="预处理节点转移概率"):
        node_sorted_neighbors = precomputed_sorted_neighbors.get(node, [])
        unnormalized_probs = [graph[node][neighbor] for neighbor in node_sorted_neighbors]
        norm_const = sum(unnormalized_probs)
        if norm_const > 0:
            normalized_probs = [float(prob)/norm_const for prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        else:
            alias_nodes[node] = None

    def _get_next_node_dynamic(prev_node, current_node):
        """动态计算下一个节点"""
        current_neighbors = precomputed_sorted_neighbors.get(current_node, [])
        if not current_neighbors:
            return None
            
        prev_neighbors_set = precomputed_neighbor_sets.get(prev_node, set())
        
        # 动态计算转移概率
        unnormalized_probs = []
        for neighbor in current_neighbors:
            edge_weight = graph[current_node][neighbor]
            
            if neighbor == prev_node:  # 返回
                prob = edge_weight / p
            elif neighbor in prev_neighbors_set:  # 距离为1
                prob = edge_weight
            else:  # 距离为2
                prob = edge_weight / q
                
            unnormalized_probs.append(prob)
        
        # 归一化
        total_prob = sum(unnormalized_probs)
        if total_prob == 0:
            return None
            
        normalized_probs = [prob / total_prob for prob in unnormalized_probs]
        
        # 使用轮盘赌选择（对于小规模邻居，这比别名方法更快）
        rand_val = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(normalized_probs):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return current_neighbors[i]
        
        # 防止浮点精度问题
        return current_neighbors[-1]

    # 开始生成随机游走
    for walk_iter in range(num_walks):
        random.shuffle(nodes)
        for start_node in tqdm(nodes, desc=f"生成随机游走 (iteration {walk_iter + 1}/{num_walks})", leave=False):
            walk = [start_node]
            
            while len(walk) < walk_length:
                current_node = walk[-1]
                current_neighbors = precomputed_sorted_neighbors.get(current_node, [])
                
                if not current_neighbors:
                    break

                if len(walk) == 1:  # 第一步转移
                    if alias_nodes[current_node] is None:
                        break
                    J, q_probs = alias_nodes[current_node]
                    next_node_idx = alias_draw(J, q_probs)
                    next_node = current_neighbors[next_node_idx]
                else:  # 后续转移，动态计算
                    prev_node = walk[-2]
                    next_node = _get_next_node_dynamic(prev_node, current_node)
                    if next_node is None:
                        break
                
                walk.append(next_node)
            
            walks.append(walk)
    
    print(f"生成 {len(walks)} 条随机游走路径.")
    return walks

# 保留原函数作为备选方案，重命名
def generate_node2vec_walks_precompute(graph, num_walks, walk_length, p, q, start_nodes=None):
    """
    预计算版本的 Node2Vec（需要大量预处理时间，但游走速度快）
    """
    print(f"生成Node2Vec随机游走 (预计算版本) (num_walks={num_walks}, walk_length={walk_length}, p={p}, q={q})...")
    walks = []
    
    if start_nodes is None:
        nodes = list(graph.keys())
    else:
        nodes = start_nodes
        
    if not nodes:
        print("警告：图或起始节点中没有节点。不会生成随机游走。")
        return []

    # 为每个节点预计算排序的邻居列表和邻居集合
    print("预计算邻居信息...")
    precomputed_sorted_neighbors = {node: sorted(graph[node].keys()) for node in tqdm(graph.keys(), desc="排序邻居")}
    precomputed_neighbor_sets = {node: set(graph[node].keys()) for node in tqdm(graph.keys(), desc="邻居集合")}

    # 为每个节点预处理转移概率
    alias_nodes = {}
    for node in tqdm(nodes, desc="预处理节点转移概率 (alias_nodes)"):
        node_sorted_neighbors = precomputed_sorted_neighbors.get(node, [])
        unnormalized_probs = [graph[node][neighbor] for neighbor in node_sorted_neighbors]
        norm_const = sum(unnormalized_probs)
        if norm_const > 0:
            normalized_probs = [float(prob)/norm_const for prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        else:
            alias_nodes[node] = None
            
    alias_edges = {}
    # 预计算边转移概率 (t, v) -> x
    edge_processing_args = []
    print("准备边转移概率计算参数...")
    for t_node in tqdm(graph.keys(), desc="准备边转移参数"):
        t_node_neighbors_set = precomputed_neighbor_sets.get(t_node, set())
        for v_node in precomputed_sorted_neighbors.get(t_node, []): # v是t的邻居
            v_node_neighbors_sorted = precomputed_sorted_neighbors.get(v_node, [])
            if not v_node_neighbors_sorted: # v 没有出度
                continue
            edge_processing_args.append(
                (graph, t_node, v_node, p, q, v_node_neighbors_sorted, t_node_neighbors_set)
            )
    
    print(f"使用多进程计算 {len(edge_processing_args)} 条边的转移概率...")
    # 使用 multiprocessing.Pool 来并行计算
    try:
        num_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        num_cores = 4
    
    pool_size = min(num_cores, len(edge_processing_args)) if len(edge_processing_args) > 0 else 1

    if len(edge_processing_args) > 0:
        with multiprocessing.Pool(processes=pool_size) as pool:
            results = list(tqdm(pool.imap(_calculate_alias_edge_for_worker, edge_processing_args), total=len(edge_processing_args), desc="计算边转移概率 (alias_edges)"))
        
        for key, value in results:
            alias_edges[key] = value
    else:
        print("没有边需要预处理。")

    for _ in range(num_walks):
        random.shuffle(nodes)
        for start_node in tqdm(nodes, desc=f"生成随机游走 (iteration {_ + 1}/{num_walks})", leave=False):
            walk = [start_node]
            
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = precomputed_sorted_neighbors.get(cur, [])
                
                if not cur_nbrs:
                    break

                if len(walk) == 1:
                    if alias_nodes[cur] is None:
                        break
                    J, q_probs = alias_nodes[cur]
                    next_node_idx_in_cur_nbrs = alias_draw(J, q_probs)
                    walk.append(cur_nbrs[next_node_idx_in_cur_nbrs])
                else:
                    prev = walk[-2]
                    
                    edge_alias_data = alias_edges.get((prev, cur))
                    if edge_alias_data is None:
                        break
                    
                    (J_edge, q_probs_edge), actual_neighbors = edge_alias_data
                    
                    if not actual_neighbors:
                        break

                    sampled_idx_in_actual_neighbors = alias_draw(J_edge, q_probs_edge)
                    next_node = actual_neighbors[sampled_idx_in_actual_neighbors]
                    walk.append(next_node)
            
            walks.append(walk)
            
    print(f"生成 {len(walks)} 条随机游走路径.")
    return walks

# 将动态版本设为默认版本
def generate_node2vec_walks(graph, num_walks, walk_length, p, q, start_nodes=None):
    """
    默认使用动态计算版本的 Node2Vec 随机游走
    """
    return generate_node2vec_walks_dynamic(graph, num_walks, walk_length, p, q, start_nodes)

def _get_walks_cache_key(graph, num_walks, walk_length, p, q):
    """
    生成随机游走缓存的唯一键值
    基于图结构和参数生成哈希值
    """
    # 创建图的简化表示用于哈希
    graph_repr = []
    for node in sorted(graph.keys()):
        neighbors = sorted([(neighbor, weight) for neighbor, weight in graph[node].items()])
        graph_repr.append((node, tuple(neighbors)))
    
    # 将图结构和参数组合成字符串
    key_string = f"graph_{hash(tuple(graph_repr))}_walks_{num_walks}_length_{walk_length}_p_{p}_q_{q}"
    
    # 生成MD5哈希值作为缓存键
    return hashlib.md5(key_string.encode()).hexdigest()

def save_walks_to_cache(walks, cache_key, cache_dir):
    """
    保存随机游走到缓存文件
    
    Args:
        walks: 随机游走列表
        cache_key: 缓存键值
        cache_dir: 缓存目录
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"node2vec_walks_{cache_key}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(walks, f)
        print(f"随机游走已保存到缓存: {cache_file}")
        return cache_file
    except Exception as e:
        print(f"保存随机游走缓存时出错: {e}")
        return None

def load_walks_from_cache(cache_key, cache_dir):
    """
    从缓存文件加载随机游走
    
    Args:
        cache_key: 缓存键值
        cache_dir: 缓存目录
    
    Returns:
        walks: 随机游走列表，如果缓存不存在则返回None
    """
    cache_file = os.path.join(cache_dir, f"node2vec_walks_{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                walks = pickle.load(f)
            print(f"从缓存加载随机游走: {cache_file}")
            print(f"加载了 {len(walks)} 条随机游走")
            return walks
        except Exception as e:
            print(f"加载随机游走缓存时出错: {e}")
            return None
    else:
        print(f"缓存文件不存在: {cache_file}")
        return None

def generate_node2vec_walks_with_cache(graph, num_walks, walk_length, p, q, start_nodes=None, 
                                     use_cache=True, cache_dir=None, force_regenerate=False):
    """
    带缓存功能的 Node2Vec 随机游走生成器
    
    Args:
        graph: 图的邻接表
        num_walks: 每个节点的游走次数
        walk_length: 游走长度
        p: 返回参数
        q: 进出参数
        start_nodes: 起始节点列表
        use_cache: 是否使用缓存
        cache_dir: 缓存目录，默认使用Config中的路径
        force_regenerate: 是否强制重新生成（忽略缓存）
    
    Returns:
        walks: 随机游走列表
    """
    if cache_dir is None:
        # 使用Config中的处理数据路径作为缓存目录
        from config import Config
        cache_dir = Config.PROCESSED_DATA_PATH
    
    # 生成缓存键
    cache_key = _get_walks_cache_key(graph, num_walks, walk_length, p, q)
    
    # 尝试从缓存加载（如果启用缓存且不强制重新生成）
    if use_cache and not force_regenerate:
        cached_walks = load_walks_from_cache(cache_key, cache_dir)
        if cached_walks is not None:
            return cached_walks
    
    # 生成新的随机游走
    print("生成新的随机游走...")
    
    # 根据配置选择使用哪个版本的 Node2Vec
    if Config.NODE2VEC_PRECOMPUTE:
        print("使用预计算版本 (更快的游走，但需要更多预处理时间)")
        walks = generate_node2vec_walks_precompute(graph, num_walks, walk_length, p, q, start_nodes)
    else:
        print("使用动态计算版本 (更快的启动，但游走时稍慢)")
        walks = generate_node2vec_walks_dynamic(graph, num_walks, walk_length, p, q, start_nodes)
    
    # 保存到缓存（如果启用缓存）
    if use_cache and walks:
        save_walks_to_cache(walks, cache_key, cache_dir)
    
    return walks

if __name__ == '__main__':
    # 示例用法
    sample_user_sequences = {
        'user1': [1, 2, 3, 4, 2, 5],
        'user2': [1, 2, 6, 4, 7],
        'user3': [3, 2, 5, 1],
        'user4': [8, 9] # 孤立的子图
    }
    
    # 1. 构建图
    item_graph = build_graph_from_sequences(sample_user_sequences, directed=False)
    print("Graph:", dict(item_graph))
    for node, neighbors in item_graph.items():
        print(f"Node {node}: {dict(neighbors)}")

    # 2. 生成Node2Vec游走
    # 使用Config中的默认参数
    node2vec_walks = generate_node2vec_walks(
        graph=item_graph,
        num_walks=Config.NUM_WALKS,
        walk_length=Config.WALK_LENGTH,
        p=Config.P_PARAM,
        q=Config.Q_PARAM
    )
    
    print(f"\nFirst 5 Node2Vec walks (example):")
    for i, walk in enumerate(node2vec_walks[:5]):
        print(f"Walk {i+1}: {walk}")
        
    # 测试有向图
    print("\n--- Testing Directed Graph ---")
    directed_item_graph = build_graph_from_sequences(sample_user_sequences, directed=True)
    print("Directed Graph:", dict(directed_item_graph))
    for node, neighbors in directed_item_graph.items():
        print(f"Node {node}: {dict(neighbors)}")

    directed_node2vec_walks = generate_node2vec_walks(
        graph=directed_item_graph,
        num_walks=5, # 减少数量以便快速测试
        walk_length=10,
        p=Config.P_PARAM,
        q=Config.Q_PARAM
    )
    print(f"\nFirst 5 Directed Node2Vec walks (example):")
    for i, walk in enumerate(directed_node2vec_walks[:5]):
        print(f"Walk {i+1}: {walk}")

    # 测试包含孤立节点的图
    print("\n--- Testing Graph with Isolated Nodes ---")
    graph_with_isolated = {
        1: {2: 1, 3: 1},
        2: {1: 1, 4: 1},
        3: {1: 1},
        4: {2: 1},
        5: {}  # 孤立节点
    }
    isolated_walks = generate_node2vec_walks(
        graph=graph_with_isolated,
        num_walks=2,
        walk_length=5,
        p=1.0, q=1.0,
        start_nodes=[1, 5] # 测试包含孤立节点的起始点
    )
    print(f"\nWalks from graph with isolated node (node 5 should not produce long walks):")
    for walk in isolated_walks:
        print(walk) 