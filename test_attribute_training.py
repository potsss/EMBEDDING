#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
属性向量训练测试脚本
用于验证生成的用户属性数据是否能正常进行属性向量训练
"""

import os
import sys
import pandas as pd
from pathlib import Path

def test_data_format():
    """测试数据格式是否正确"""
    print("=== 测试数据格式 ===")
    
    # 检查文件是否存在
    behavior_file = "data/edu.csv"
    attribute_file = "data/user_attributes.tsv"
    
    if not os.path.exists(behavior_file):
        print(f"❌ 行为数据文件不存在: {behavior_file}")
        return False
    
    if not os.path.exists(attribute_file):
        print(f"❌ 属性数据文件不存在: {attribute_file}")
        return False
    
    print(f"✅ 数据文件存在")
    
    # 读取并检查行为数据
    try:
        behavior_df = pd.read_csv(behavior_file, sep='\t', encoding='utf-8')
        print(f"✅ 行为数据读取成功: {len(behavior_df)} 行")
        print(f"   列: {list(behavior_df.columns)}")
        
        # 检查必需的列
        required_cols = ['user_id', 'url', 'timestamp_str', 'weight']
        missing_cols = [col for col in required_cols if col not in behavior_df.columns]
        if missing_cols:
            print(f"❌ 行为数据缺少必需列: {missing_cols}")
            return False
        
        unique_users_behavior = set(behavior_df['user_id'].unique())
        print(f"   唯一用户数: {len(unique_users_behavior)}")
        
    except Exception as e:
        print(f"❌ 行为数据读取失败: {str(e)}")
        return False
    
    # 读取并检查属性数据
    try:
        attribute_df = pd.read_csv(attribute_file, sep='\t', encoding='utf-8')
        print(f"✅ 属性数据读取成功: {len(attribute_df)} 行")
        print(f"   列: {list(attribute_df.columns)}")
        
        # 检查必需的列
        if 'user_id' not in attribute_df.columns:
            print(f"❌ 属性数据缺少 user_id 列")
            return False
        
        unique_users_attribute = set(attribute_df['user_id'].unique())
        print(f"   唯一用户数: {len(unique_users_attribute)}")
        
        # 检查用户ID一致性
        common_users = unique_users_behavior & unique_users_attribute
        print(f"   共同用户数: {len(common_users)}")
        
        if len(common_users) == 0:
            print(f"❌ 行为数据和属性数据没有共同用户")
            return False
        
        coverage = len(common_users) / len(unique_users_behavior) * 100
        print(f"   属性覆盖率: {coverage:.1f}%")
        
        if coverage < 90:
            print(f"⚠️ 属性覆盖率较低，可能影响训练效果")
        
    except Exception as e:
        print(f"❌ 属性数据读取失败: {str(e)}")
        return False
    
    print("✅ 数据格式检查通过")
    return True

def test_config_setup():
    """测试配置设置"""
    print("\n=== 测试配置设置 ===")
    
    try:
        from config import Config
        
        print(f"✅ 配置文件导入成功")
        print(f"   ENABLE_ATTRIBUTES: {Config.ENABLE_ATTRIBUTES}")
        print(f"   ATTRIBUTE_DATA_PATH: {Config.ATTRIBUTE_DATA_PATH}")
        print(f"   ATTRIBUTE_EMBEDDING_DIM: {Config.ATTRIBUTE_EMBEDDING_DIM}")
        print(f"   FUSION_HIDDEN_DIM: {Config.FUSION_HIDDEN_DIM}")
        print(f"   FINAL_USER_EMBEDDING_DIM: {Config.FINAL_USER_EMBEDDING_DIM}")
        
        if not Config.ENABLE_ATTRIBUTES:
            print("⚠️ ENABLE_ATTRIBUTES 设置为 False，需要在命令行中启用")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件导入失败: {str(e)}")
        return False

def test_import_modules():
    """测试模块导入"""
    print("\n=== 测试模块导入 ===")
    
    modules_to_test = [
        ('data_preprocessing', 'DataPreprocessor'),
        ('model', 'Item2Vec'),
        ('trainer', 'Trainer'),
        ('evaluator', 'Evaluator'),
        ('visualizer', 'Visualizer'),
    ]
    
    success = True
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name} 导入成功")
            else:
                print(f"❌ {module_name} 中找不到 {class_name}")
                success = False
        except Exception as e:
            print(f"❌ {module_name} 导入失败: {str(e)}")
            success = False
    
    return success

def run_quick_test():
    """运行快速测试"""
    print("\n=== 运行快速属性预处理测试 ===")
    
    try:
        from data_preprocessing import AttributeProcessor
        from config import Config
        
        # 创建临时配置
        original_enable = Config.ENABLE_ATTRIBUTES
        original_path = Config.ATTRIBUTE_DATA_PATH
        
        Config.ENABLE_ATTRIBUTES = True
        Config.ATTRIBUTE_DATA_PATH = "data/user_attributes.tsv"
        
        # 测试属性处理器
        processor = AttributeProcessor()
        
        # 测试属性数据加载和处理
        user_attributes, attribute_info = processor.process_attributes(Config.ATTRIBUTE_DATA_PATH)
        
        if user_attributes is not None and attribute_info is not None:
            print(f"✅ 属性数据处理成功")
            print(f"   用户数: {len(user_attributes)}")
            print(f"   属性数量: {len(attribute_info)}")
            print(f"   属性信息: {list(attribute_info.keys())}")
            
            # 显示一个用户的属性示例
            sample_user_id = list(user_attributes.keys())[0]
            sample_attributes = user_attributes[sample_user_id]
            print(f"   示例用户 {sample_user_id}: {sample_attributes}")
        else:
            print(f"❌ 属性数据处理失败")
            return False
        
        # 恢复配置
        Config.ENABLE_ATTRIBUTES = original_enable
        Config.ATTRIBUTE_DATA_PATH = original_path
        
        return True
        
    except Exception as e:
        print(f"❌ 快速测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试属性向量训练功能\n")
    
    all_passed = True
    
    # 测试数据格式
    if not test_data_format():
        all_passed = False
    
    # 测试配置设置
    if not test_config_setup():
        all_passed = False
    
    # 测试模块导入
    if not test_import_modules():
        all_passed = False
    
    # 运行快速测试
    if not run_quick_test():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 所有测试通过！")
        print("\n建议的运行命令:")
        print("python main.py --mode all --data_path data/edu.csv --enable_attributes --attribute_data_path data/user_attributes.tsv")
    else:
        print("❌ 部分测试失败，请检查上述错误信息")
        print("\n请确保:")
        print("1. 所有依赖包已正确安装")
        print("2. 数据文件格式正确")
        print("3. 代码文件没有语法错误")

if __name__ == "__main__":
    main() 