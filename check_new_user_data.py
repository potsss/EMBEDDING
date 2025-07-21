#!/usr/bin/env python3
"""
新用户数据兼容性检查工具

该脚本用于检查新用户数据与训练数据的兼容性，
提供详细的过滤报告和数据质量分析。

使用方法:
    python check_new_user_data.py --experiment_name your_experiment
    python check_new_user_data.py --experiment_name your_experiment --new_user_behavior_path data/custom_behavior.csv
"""

import argparse
import os
import pandas as pd
import pickle
from collections import defaultdict
import sys

def load_training_entities(experiment_path):
    """加载训练实体记录"""
    entities_path = os.path.join(experiment_path, 'processed_data', 'training_entities.pkl')
    if os.path.exists(entities_path):
        with open(entities_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"❌ 错误：未找到训练实体记录文件: {entities_path}")
        return None

def analyze_behavior_data(behavior_path, training_entities):
    """分析行为数据兼容性"""
    print("🔍 分析行为数据...")
    
    if not os.path.exists(behavior_path):
        print(f"❌ 行为数据文件不存在: {behavior_path}")
        return None
    
    df = pd.read_csv(behavior_path)
    print(f"📁 加载行为数据: {behavior_path}")
    print(f"📊 数据形状: {df.shape}")
    
    # 统计URL
    all_urls = set(df['url'].unique())
    training_urls = training_entities['urls']
    
    known_urls = all_urls.intersection(training_urls)
    unknown_urls = all_urls - training_urls
    
    # 按用户统计
    user_stats = {}
    for user_id, group in df.groupby('user_id'):
        user_urls = set(group['url'])
        user_known = user_urls.intersection(training_urls)
        user_unknown = user_urls - training_urls
        
        user_stats[user_id] = {
            'total_records': len(group),
            'total_urls': len(user_urls),
            'known_urls': len(user_known),
            'unknown_urls': len(user_unknown),
            'known_records': len(group[group['url'].isin(training_urls)]),
            'unknown_records': len(group[~group['url'].isin(training_urls)]),
            'coverage': len(user_known) / len(user_urls) if user_urls else 0
        }
    
    return {
        'total_urls': len(all_urls),
        'known_urls': len(known_urls),
        'unknown_urls': len(unknown_urls),
        'unknown_url_list': sorted(unknown_urls),
        'user_stats': user_stats,
        'coverage': len(known_urls) / len(all_urls) if all_urls else 0
    }

def analyze_location_data(location_path, training_entities):
    """分析位置数据兼容性"""
    print("🔍 分析位置数据...")
    
    if not os.path.exists(location_path):
        print(f"❌ 位置数据文件不存在: {location_path}")
        return None
    
    if 'base_stations' not in training_entities:
        print("⚠️ 训练数据中没有基站信息")
        return None
    
    df = pd.read_csv(location_path, sep='\t')
    print(f"📁 加载位置数据: {location_path}")
    print(f"📊 数据形状: {df.shape}")
    
    # 统计基站
    all_base_stations = set(df['base_station_id'].unique())
    training_base_stations = training_entities['base_stations']
    
    known_base_stations = all_base_stations.intersection(training_base_stations)
    unknown_base_stations = all_base_stations - training_base_stations
    
    # 按用户统计
    user_stats = {}
    for user_id, group in df.groupby('user_id'):
        user_base_stations = set(group['base_station_id'])
        user_known = user_base_stations.intersection(training_base_stations)
        user_unknown = user_base_stations - training_base_stations
        
        user_stats[user_id] = {
            'total_records': len(group),
            'total_base_stations': len(user_base_stations),
            'known_base_stations': len(user_known),
            'unknown_base_stations': len(user_unknown),
            'known_records': len(group[group['base_station_id'].isin(training_base_stations)]),
            'unknown_records': len(group[~group['base_station_id'].isin(training_base_stations)]),
            'coverage': len(user_known) / len(user_base_stations) if user_base_stations else 0
        }
    
    return {
        'total_base_stations': len(all_base_stations),
        'known_base_stations': len(known_base_stations),
        'unknown_base_stations': len(unknown_base_stations),
        'unknown_base_station_list': sorted(unknown_base_stations),
        'user_stats': user_stats,
        'coverage': len(known_base_stations) / len(all_base_stations) if all_base_stations else 0
    }

def print_compatibility_report(behavior_analysis, location_analysis):
    """打印兼容性报告"""
    print("\n" + "="*60)
    print("📋 新用户数据兼容性报告")
    print("="*60)
    
    # 总体兼容性评分
    total_score = 0
    max_score = 0
    
    # 行为数据分析
    if behavior_analysis:
        print(f"\n🌐 行为数据分析:")
        print(f"  📊 URL总数: {behavior_analysis['total_urls']}")
        print(f"  ✅ 已知URL: {behavior_analysis['known_urls']}")
        print(f"  ❌ 未知URL: {behavior_analysis['unknown_urls']}")
        print(f"  📈 覆盖率: {behavior_analysis['coverage']*100:.1f}%")
        
        total_score += behavior_analysis['coverage'] * 50
        max_score += 50
        
        if behavior_analysis['unknown_urls'] > 0:
            print(f"  📋 未知URL列表 (前10个): {behavior_analysis['unknown_url_list'][:10]}")
            if len(behavior_analysis['unknown_url_list']) > 10:
                print(f"      ... 还有 {len(behavior_analysis['unknown_url_list'])-10} 个")
        
        # 用户级别统计
        print(f"\n👥 用户级别分析:")
        for user_id, stats in behavior_analysis['user_stats'].items():
            print(f"  {user_id}: {stats['known_records']}/{stats['total_records']} 记录可用 "
                  f"({stats['coverage']*100:.1f}% URL覆盖率)")
    
    # 位置数据分析
    if location_analysis:
        print(f"\n📡 位置数据分析:")
        print(f"  📊 基站总数: {location_analysis['total_base_stations']}")
        print(f"  ✅ 已知基站: {location_analysis['known_base_stations']}")
        print(f"  ❌ 未知基站: {location_analysis['unknown_base_stations']}")
        print(f"  📈 覆盖率: {location_analysis['coverage']*100:.1f}%")
        
        total_score += location_analysis['coverage'] * 50
        max_score += 50
        
        if location_analysis['unknown_base_stations'] > 0:
            print(f"  📋 未知基站列表: {location_analysis['unknown_base_station_list']}")
        
        # 用户级别统计
        print(f"\n👥 用户级别分析:")
        for user_id, stats in location_analysis['user_stats'].items():
            print(f"  {user_id}: {stats['known_records']}/{stats['total_records']} 记录可用 "
                  f"({stats['coverage']*100:.1f}% 基站覆盖率)")
    
    # 总体评分
    if max_score > 0:
        final_score = total_score / max_score
        print(f"\n🎯 总体兼容性评分: {final_score*100:.1f}%")
        
        if final_score >= 0.8:
            print("✅ 优秀 - 数据兼容性很好，可以直接使用")
        elif final_score >= 0.6:
            print("⚠️ 良好 - 数据基本兼容，但建议检查未知实体")
        elif final_score >= 0.4:
            print("⚠️ 一般 - 数据兼容性较差，建议补充训练数据")
        else:
            print("❌ 较差 - 数据兼容性很差，需要重新准备数据")
    
    # 建议
    print(f"\n💡 改进建议:")
    if behavior_analysis and behavior_analysis['unknown_urls'] > 0:
        print(f"  • 考虑将重要的未知URL添加到训练数据中")
        print(f"  • 或者将未知URL映射到相似的已知URL")
    
    if location_analysis and location_analysis['unknown_base_stations'] > 0:
        print(f"  • 检查基站ID的命名规范是否一致")
        print(f"  • 考虑将重要的未知基站添加到训练数据中")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='新用户数据兼容性检查工具')
    parser.add_argument('--experiment_name', type=str, required=True, 
                       help='实验名称')
    parser.add_argument('--experiment_path', type=str, 
                       help='实验路径（如果不指定，将使用experiments/{experiment_name}）')
    parser.add_argument('--new_user_behavior_path', type=str, 
                       default='data/new_user_behavior.csv',
                       help='新用户行为数据路径')
    parser.add_argument('--new_user_location_path', type=str, 
                       default='data/new_user_base_stations.tsv',
                       help='新用户位置数据路径')
    
    args = parser.parse_args()
    
    # 确定实验路径
    if args.experiment_path:
        experiment_path = args.experiment_path
    else:
        experiment_path = f'experiments/{args.experiment_name}'
    
    if not os.path.exists(experiment_path):
        print(f"❌ 错误：实验路径不存在: {experiment_path}")
        sys.exit(1)
    
    print(f"🔍 检查实验: {args.experiment_name}")
    print(f"📁 实验路径: {experiment_path}")
    print(f"📱 行为数据: {args.new_user_behavior_path}")
    print(f"📍 位置数据: {args.new_user_location_path}")
    
    # 加载训练实体记录
    training_entities = load_training_entities(experiment_path)
    if training_entities is None:
        print("❌ 无法加载训练实体记录，请确保已完成数据预处理")
        sys.exit(1)
    
    print(f"✅ 成功加载训练实体记录:")
    print(f"  📊 训练URL数量: {len(training_entities['urls'])}")
    if 'base_stations' in training_entities:
        print(f"  📡 训练基站数量: {len(training_entities['base_stations'])}")
    
    # 分析行为数据
    behavior_analysis = analyze_behavior_data(args.new_user_behavior_path, training_entities)
    
    # 分析位置数据
    location_analysis = analyze_location_data(args.new_user_location_path, training_entities)
    
    # 打印报告
    print_compatibility_report(behavior_analysis, location_analysis)

if __name__ == "__main__":
    main() 