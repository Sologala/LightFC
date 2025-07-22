import onnx
import argparse
from collections import Counter

def get_onnx_operators(model_path):
    """提取 ONNX 模型中使用的所有算子"""
    try:
        # 加载 ONNX 模型
        model = onnx.load(model_path)
        # 检查模型是否有效
        onnx.checker.check_model(model)
        
        # 遍历所有节点，收集算子类型
        operators = []
        for node in model.graph.node:
            operators.append(node.op_type)
        
        return operators
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='提取 ONNX 模型中使用的所有算子')
    parser.add_argument('--model', required=True, help='ONNX 模型文件路径')
    parser.add_argument('--count', action='store_true', help='显示每个算子的使用次数')
    args = parser.parse_args()
    
    # 获取算子列表
    operators = get_onnx_operators(args.model)
    
    if not operators:
        print("未找到算子或模型加载失败。")
        return
    
    # 输出结果
    if args.count:
        # 统计每个算子的使用次数
        operator_counts = Counter(operators)
        print("模型中使用的算子及其出现次数:")
        for op, count in operator_counts.most_common():
            print(f"{op}: {count}")
    else:
        # 去重并排序
        unique_operators = sorted(list(set(operators)))
        print("模型中使用的算子列表:")
        for op in unique_operators:
            print(op)

if __name__ == "__main__":
    main()    
