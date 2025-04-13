import re
import numpy as np
import matplotlib.pyplot as plt

def calculate_statistics(filename):
    first_numbers = []
    second_numbers = []
    
    # 优化正则表达式匹配模式
    pattern = re.compile(
        r'\[\s*([+-]?\d+\.?\d*|\.?\d+)(?:[eE][+-]?\d+)?'  # 第一个数字
        r'\s+([+-]?\d+\.?\d*|\.?\d+)(?:[eE][+-]?\d+)?'    # 第二个数字
    )
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                try:
                    num1 = float(match.group(1))
                    num2 = float(match.group(2))
                    first_numbers.append(num1)
                    second_numbers.append(num2)
                except ValueError:
                    continue
    
    # 计算统计量
    stats = {
        'first': {
            'mean': np.mean(first_numbers) if first_numbers else None,
            'var': np.var(first_numbers) if first_numbers else None
        },
        'second': {
            'mean': np.mean(second_numbers) if second_numbers else None,
            'var': np.var(second_numbers) if second_numbers else None
        }
    }
    
    # 绘制分布图
    plt.figure(figsize=(12, 6))
    
    # 第一个数分布
    plt.subplot(1, 2, 1)
    plt.hist(first_numbers, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(stats['first']['mean'], color='red', linestyle='dashed', linewidth=2, 
                label=f"均值: {stats['first']['mean']:.4f}")
    plt.title('第一个数分布（方差={:.6f}）'.format(stats['first']['var']))
    plt.xlabel('数值')
    plt.ylabel('频数')
    plt.legend()
    
    # 第二个数分布
    plt.subplot(1, 2, 2)
    plt.hist(second_numbers, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(stats['second']['mean'], color='blue', linestyle='dashed', linewidth=2, 
                label=f"均值: {stats['second']['mean']:.4f}")
    plt.title('第二个数分布（方差={:.6f}）'.format(stats['second']['var']))
    plt.xlabel('数值')
    plt.ylabel('频数')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return stats

# 执行分析
stats = calculate_statistics(r"C:\Users\qixin\Desktop\result-det.txt")

# 打印结果
print("第一个数统计：")
print(f"平均值: {stats['first']['mean']:.6f}")
print(f"方差: {stats['first']['var']:.6f}\n")

print("第二个数统计：")
print(f"平均值: {stats['second']['mean']:.6f}")
print(f"方差: {stats['second']['var']:.6f}")