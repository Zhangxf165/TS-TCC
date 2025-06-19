import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

def load_emg_samples(mat_file_path: str, var_name: str) -> np.ndarray:
    """
    加载EMG样本数据，不进行帧划分，保持连续时间序列
    """
    mat_data = sio.loadmat(mat_file_path)
    if var_name not in mat_data:
        raise KeyError(f"变量名 '{var_name}' 不存在：可选 {list(mat_data.keys())}")

    data = mat_data[var_name]
    labels = data[:, 0].astype(int)
    signals = data[:, 1:]

    unique_labels, counts = np.unique(labels, return_counts=True)
    expected_labels = np.array([0, 1, 2, 3])
    if not np.array_equal(np.sort(unique_labels), expected_labels) or len(set(counts)) != 1:
        raise ValueError(f"标签必须为0-3且行数一致，当前：{dict(zip(unique_labels, counts))}")

    n_samples = counts[0]
    T = signals.shape[1]
    
    # 构建 (n_samples, 4, T)
    samples = np.zeros((n_samples, 4, T), dtype=signals.dtype)
    for lab in expected_labels:
        idx = np.where(labels == lab)[0]
        samples[:, lab, :] = signals[idx]
    
    # === 3个样本拼接（如果需要）===
    if n_samples >= 3:
        n_groups = n_samples // 3
        concatenated_samples = []
        
        for group_idx in range(n_groups):
            start_idx = group_idx * 3
            end_idx = start_idx + 3
            group_samples = samples[start_idx:end_idx]
            
            # 在时间维度拼接：(3, 4, T) -> (4, 3*T)
            concat_sample = np.concatenate(group_samples, axis=-1)
            concatenated_samples.append(concat_sample)
        
        samples = np.stack(concatenated_samples, axis=0)
        n_samples = n_groups
        T = 3 * T
        
        print(f"拼接后：{n_samples} 个样本，每个样本形状: (4, {T})")
    
    
    return samples.astype(np.float32)

class SpanningConv2D(nn.Module):
    """
    Spanning Convolution 2D layer for sEMG signal processing
    Modified for 4-channel muscle configuration with long sequences
    """

    def __init__(self, in_channels, out_channels, kernel_size, spanning_type='2x2',
                 stride=1, padding=0, bias=True):
        super(SpanningConv2D, self).__init__()

        self.spanning_type = spanning_type
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Define spanning configurations for 4-muscle layout
        if spanning_type == '2x2':
            # 4肌肉分组: 左侧(0,1) vs 右侧(2,3)
            self.muscle_pairs = [(0, 1), (2, 3)]
            self.num_muscle_groups = 2
        else:
            raise ValueError(f"Unsupported spanning_type: {spanning_type}. Only '2x2' supported for 4-muscle configuration.")

        # Create separate convolution layers for each muscle group
        self.conv_groups = nn.ModuleList()
        for _ in range(self.num_muscle_groups):
            self.conv_groups.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding, bias=bias)
            )

    def forward(self, x):
        """
        Forward pass for spanning convolution
        x shape: (batch_size, channels, 4_muscles, time_steps)
        """
        # 确保输入数据类型正确
        x = x.float()
        
        batch_size, channels, muscles, time_steps = x.shape
        assert muscles == 4, f"Expected 4 muscle channels, got {muscles}"

        outputs = []

        if self.spanning_type == '2x2':
            # 左侧肌肉群: [0, 1]
            left_muscles = x[:, :, [0, 1], :]  # (batch_size, channels, 2, time_steps)
            left_output = self.conv_groups[0](left_muscles)
            outputs.append(left_output)
            
            # 右侧肌肉群: [2, 3]
            right_muscles = x[:, :, [2, 3], :]  # (batch_size, channels, 2, time_steps)
            right_output = self.conv_groups[1](right_muscles)
            outputs.append(right_output)

        # Concatenate outputs along muscle dimension
        output = torch.cat(outputs, dim=2)
        return output

class SCNN2D_LongSequence(nn.Module):
    """
    Complete 2D Spanning CNN for 4-channel sEMG signal classification
    Optimized for long sequences (9 seconds, ~18000 time steps)
    """

    def __init__(self, num_classes=5, spanning_type='2x2'):
        super(SCNN2D_LongSequence, self).__init__()
        
        # 第一层: 跨越卷积 + 大幅降采样
        self.spanning_conv1 = SpanningConv2D(
            in_channels=1, out_channels=32,
            kernel_size=(2, 128),     # 增大时间卷积核以适应长序列
            spanning_type=spanning_type,
            padding=(0, 64)           # 对应padding
        )
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 8))  # 大幅池化降采样
        
        # 第二层: 中等降采样
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 64), padding=(0, 32))
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4))
        
        # 第三层: 小幅降采样
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 32), padding=(0, 16))
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # 第四层: 最终特征提取
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 16), padding=(0, 8))
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # 全局平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类层
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 确保输入维度正确和数据类型
        x = x.float()
        
        # 如果输入是4D (batch, 4, frames, frame_size)，需要增加channel维度
        if len(x.shape) == 4:
            batch_size, muscles, frames, frame_size = x.shape
            # 重塑为 (batch, 1, 4, frames*frame_size) 用于2D卷积
            x = x.view(batch_size, 1, muscles, frames * frame_size)
        
        batch_size, channels, muscles, time_steps = x.shape
        assert muscles == 4, f"Expected 4 muscle channels, got {muscles}"
        
        print(f"Input shape: {x.shape}")
        
        # 跨越卷积层
        x = self.spanning_conv1(x)
        print(f"After spanning conv: {x.shape}")
        x = F.relu(self.batch_norm1(x))
        x = self.pool1(x)
        print(f"After pool1: {x.shape}")
        
        # 第二层
        x = self.conv2(x)
        x = F.relu(self.batch_norm2(x))
        x = self.pool2(x)
        print(f"After pool2: {x.shape}")
        
        # 第三层
        x = self.conv3(x)
        x = F.relu(self.batch_norm3(x))
        x = self.pool3(x)
        print(f"After pool3: {x.shape}")
        
        # 第四层
        x = self.conv4(x)
        x = F.relu(self.batch_norm4(x))
        x = self.pool4(x)
        print(f"After pool4: {x.shape}")
        
        # 全局池化和分类
        x = self.adaptive_pool(x)
        print(f"After adaptive pool: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"After flatten: {x.shape}")
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def process_data(mat_path: str, var_name: str, model: SCNN2D_LongSequence, batch_size: int = 8) -> np.ndarray:
    """
    加载并处理单个 .mat 文件数据，返回提取的特征向量矩阵
    """
    samples = load_emg_samples(mat_path, var_name)
    print(f"加载 {mat_path}，输入形状: {samples.shape}")
    
    # 转换为PyTorch张量并添加channel维度
    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    samples_tensor = samples_tensor.unsqueeze(1)  # (n_samples, 1, 4, 18000)
    
    model.eval()
    all_features = []
    
    # 批处理推理
    with torch.no_grad():
        for i in range(0, len(samples_tensor), batch_size):
            batch = samples_tensor[i:i+batch_size]
            features = model(batch)
            all_features.append(features.cpu().numpy())
    
    # 合并所有批次的特征
    all_features = np.concatenate(all_features, axis=0)
    print(f"提取特征: {all_features.shape}")
    return all_features

# Example usage and testing
if __name__ == "__main__":
    print("=== 4-Channel SCNN2D Long Sequence Model Test ===")
    
    # 先加载第一个文件
    datasets = [
        ('healthy_men_before.mat', 'combined_data'),
        ('healthy_men_after_add.mat', 'healthy_men_after')
    ]
    
    first_path, first_var = datasets[0]
    example_samples = load_emg_samples(first_path, first_var)
    print(f"示例样本形状: {example_samples.shape}")  # (810, 4, 18000)
    
    # 创建模型
    model = SCNN2D_LongSequence(num_classes=5, spanning_type='2x2')
    
    # 正确的维度处理
    test_sample = torch.tensor(example_samples[:8], dtype=torch.float32)  # (8, 4, 18000)
    
    # 关键修改：正确添加channel维度
    # 原来: (8, 4, 18000) -> unsqueeze(1) -> (8, 1, 4, 18000)
    # 但model期望: (batch, channels, muscles, time)
    
    # 方法1: 在第1个位置插入channel维度
    test_sample = test_sample.unsqueeze(1)  # (8, 1, 4, 18000)
    print(f"测试输入形状: {test_sample.shape}")
    print(f"维度解析: batch={test_sample.shape[0]}, channels={test_sample.shape[1]}, muscles={test_sample.shape[2]}, time={test_sample.shape[3]}")
    
    output = model(test_sample)
    print(f"测试输出形状: {output.shape}")
    
    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 处理所有数据集
    for mat_path, var_name in datasets:
        try:
            features = process_data(mat_path, var_name, model)
            out_name_npy = mat_path.replace('.mat', '_scnn_features.npy')
            
            # 保存为 .npy 文件
            np.save(out_name_npy, features)
            print(f"保存特征到 {out_name_npy}")
            
        except Exception as e:
            print(f"处理 {mat_path} 时出错: {e}")
    
    print("\n=== 处理完成 ===")