import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from scipy.signal import spectrogram

# 参数设置 - 与maxed2.py保持一致
Fs = 2000
window_size = 200
overlap = 100        # 帧移=200-100=100个点=0.05秒
nfft = 200

def process_emg_data(mat_file: str, var_name: str) -> np.ndarray:
    """
    数据预处理函数，与maxed2.py完全一致的处理方式
    """
    print(f"\n处理数据集: {mat_file} - {var_name}")
    mat_data = scipy.io.loadmat(mat_file)
    data_matrix = mat_data[var_name]
    labels = data_matrix[:, 0].astype(int)
    raw = data_matrix[:, 1:]
    
    # 按标签分组整理数据
    channel_data = {}
    for lab, row in zip(labels, raw):
        channel_data.setdefault(lab, []).append(row)
    for ch in channel_data:
        channel_data[ch] = np.stack(channel_data[ch], axis=0)
    
    num_samples = channel_data[0].shape[0]
    print(f"原始样本数: {num_samples}")
    
    # === 新增：将3个样本拼接成1个样本 ===
    if num_samples >= 3:
        # 每3个样本为一组进行拼接
        n_groups = num_samples // 3
        print(f"将 {num_samples} 个样本拼接成 {n_groups} 个长样本")
        
        # 为每个通道创建拼接后的数据
        concatenated_channel_data = {}
        for ch in range(4):
            concatenated_samples = []
            for group_idx in range(n_groups):
                start_idx = group_idx * 3
                end_idx = start_idx + 3
                # 取3个样本并在时间维度拼接
                group_samples = channel_data[ch][start_idx:end_idx]  # (3, signal_length)
                concat_sample = np.concatenate(group_samples, axis=0)  # (3*signal_length,)
                concatenated_samples.append(concat_sample)
            concatenated_channel_data[ch] = np.stack(concatenated_samples, axis=0)
        
        # 更新通道数据和样本数
        channel_data = concatenated_channel_data
        num_samples = n_groups
        
        print(f"拼接后样本数: {num_samples}")
        print(f"每个样本长度: {channel_data[0].shape[1]}")
    
    # 计算新的时间帧数
    signal_length = channel_data[0].shape[1]  # 拼接后的信号长度 = 18000
    frame_shift = window_size - overlap  # 100个点
    estimated_frames = (signal_length - window_size) // frame_shift + 1
    print(f"预估时间帧数: {estimated_frames}")
    print(f"每帧时间: {frame_shift / Fs:.3f}秒")
    
    # 生成频谱图
    spectrogram_list = []
    for i in range(num_samples):
        sample = np.zeros((4, channel_data[0].shape[1]))  # 使用拼接后的长度
        for ch in range(4):
            sample[ch] = channel_data[ch][i]
        
        # 动态分配谱图矩阵大小
        spec_mat = np.zeros((estimated_frames, 4, 20))
        
        for ch in range(4):
            f, t, Sxx = spectrogram(
                sample[ch], fs=Fs, window='hamming',
                nperseg=window_size, noverlap=overlap, nfft=nfft
            )
            idx = np.where((f>=10)&(f<=200))[0]
            Ssel = Sxx[idx, :]
            
            # 调整到预估帧数
            actual_frames = Ssel.shape[1]
            if actual_frames > estimated_frames:
                Ssel = Ssel[:, :estimated_frames]  # 截断
            elif actual_frames < estimated_frames:
                # 补零
                padding = np.zeros((20, estimated_frames - actual_frames))
                Ssel = np.concatenate([Ssel, padding], axis=1)
            
            spec_mat[:, ch, :] = Ssel.T
        
        spectrogram_list.append(spec_mat)
    
    print(f"生成 {len(spectrogram_list)} 个谱图矩阵，每个形状为 {spec_mat.shape}")
    
    # 转换为PyTorch格式: (n_samples, frames, 4, 20)
    spectrogram_array = np.stack(spectrogram_list, axis=0)
    
    return spectrogram_array.astype(np.float32)


class SpanningConv2D(nn.Module):
    """
    Spanning Convolution 2D layer for frequency domain sEMG processing
    Modified for 4-channel muscle configuration with fusion
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
            self.muscles_per_group = 2
        else:
            raise ValueError(f"Unsupported spanning_type: {spanning_type}")

        # 🔧 关键修改：每个组输出out_channels//2，这样拼接后正好是out_channels
        channels_per_group = out_channels // self.num_muscle_groups
        
        # Create separate convolution layers for each muscle group
        self.conv_groups = nn.ModuleList()
        for _ in range(self.num_muscle_groups):
            self.conv_groups.append(
                nn.Conv2d(in_channels, channels_per_group, kernel_size, 
                         stride=stride, padding=padding, bias=bias)
            )

    def forward(self, x):
        """
        Forward pass for spanning convolution with fusion
        x shape: (batch_size, channels, 4_muscles, 20_freq)
        """
        batch_size, channels, muscles, freq_bins = x.shape
        assert muscles == 4, f"Expected 4 muscle channels, got {muscles}"

        outputs = []

        if self.spanning_type == '2x2':
            # 左侧肌肉群: [0, 1]
            left_muscles = x[:, :, [0, 1], :]  # (batch_size, channels, 2, 20)
            left_output = self.conv_groups[0](left_muscles)  # (batch, out_channels//2, 1, 16)
            outputs.append(left_output)
            
            # 右侧肌肉群: [2, 3]
            right_muscles = x[:, :, [2, 3], :]  # (batch_size, channels, 2, 20)
            right_output = self.conv_groups[1](right_muscles)  # (batch, out_channels//2, 1, 16)
            outputs.append(right_output)

        # 🔥 融合：在通道维度拼接，而不是肌肉维度
        output = torch.cat(outputs, dim=1)  # (batch, out_channels, 1, 16)
        
        return output


class FrequencyFrameEncoder(nn.Module):
    """
    频域帧编码器，处理单帧频谱图数据 (4, 20)
    """
    
    def __init__(self, spanning_type='2x2'):
        super(FrequencyFrameEncoder, self).__init__()
        
        # 第一层: Spanning convolution - 输出32通道（融合后）
        self.spanning_conv1 = SpanningConv2D(
            in_channels=1, out_channels=32,  # 直接输出32通道
            kernel_size=(2, 5), spanning_type=spanning_type,
            padding=(0, 0)
        )
        
        # 池化层
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        # ✅ 现在输入通道数正确匹配了
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 0))
        
    def forward(self, x):
        """
        x shape: (batch_size, 4, 20) -> 需要添加channel维度
        """
        # 添加channel维度: (batch_size, 4, 20) -> (batch_size, 1, 4, 20)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        batch_size, channels, muscles, freq_bins = x.shape
        print(f"Frame input shape: {x.shape}")
        
        # 🔥 Spanning convolution with fusion: (batch, 1, 4, 20) -> (batch, 32, 1, 16)
        x = self.spanning_conv1(x)
        print(f"After spanning conv (fused): {x.shape}")
        x = F.relu(x)
        
        # Average pooling: (batch, 32, 1, 16) -> (batch, 32, 1, 8)
        x = self.pool1(x)
        print(f"After pool1: {x.shape}")
        
        # Second convolution: (batch, 32, 1, 8) -> (batch, 64, 1, 4)
        x = self.conv2(x)
        print(f"After conv2: {x.shape}")
        x = F.relu(x)
        
        # Flatten: (batch, 64, 1, 4) -> (batch, 256)
        x = x.view(x.size(0), -1)
        print(f"After flatten: {x.shape}")
        
        return x


class FrequencyDomainSCNN(nn.Module):
    """
    频域SCNN模型，结合TimeDistributed + LSTM
    """
    
    def __init__(self, spanning_type='2x2', lstm_hidden=256):
        super(FrequencyDomainSCNN, self).__init__()
        
        self.frame_encoder = FrequencyFrameEncoder(spanning_type=spanning_type)
        
        # ✅ LSTM输入维度恢复为256
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden, 
                           batch_first=True)
        
    def forward(self, x):
        """
        x shape: (batch_size, frames, 4, 20)
        """
        batch_size, frames, muscles, freq_bins = x.shape
        print(f"Input shape: {x.shape}")
        
        # 处理每一帧
        frame_features = []
        for frame_idx in range(frames):
            frame_data = x[:, frame_idx, :, :]  # (batch, 4, 20)
            frame_feat = self.frame_encoder(frame_data)  # (batch, 256)
            frame_features.append(frame_feat)
        
        # 堆叠帧特征: (batch, frames, 256)
        sequence_features = torch.stack(frame_features, dim=1)
        print(f"Sequence features shape: {sequence_features.shape}")
        
        # LSTM处理时序信息
        lstm_out, (h_n, c_n) = self.lstm(sequence_features)
        
        # 使用最后的隐藏状态作为最终特征
        final_features = h_n[-1]  # (batch, lstm_hidden)
        print(f"Final features shape: {final_features.shape}")
        
        return final_features


def process_data_with_model(mat_path: str, var_name: str, model: FrequencyDomainSCNN, batch_size: int = 8) -> np.ndarray:
    """
    使用频域SCNN模型处理数据并提取特征
    """
    spectrogram_data = process_emg_data(mat_path, var_name)
    print(f"加载 {mat_path}，输入形状: {spectrogram_data.shape}")
    
    # 转换为PyTorch张量
    data_tensor = torch.tensor(spectrogram_data, dtype=torch.float32)
    
    model.eval()
    all_features = []
    
    # 批处理推理
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size]
            features = model(batch)
            all_features.append(features.cpu().numpy())
    
    # 合并所有批次的特征
    all_features = np.concatenate(all_features, axis=0)
    print(f"提取特征: {all_features.shape}")
    return all_features


# Example usage and testing
if __name__ == "__main__":
    print("=== 频域SCNN with LSTM Model Test ===")
    
    # 计算拼接后的时间帧数
    estimated_frames = (18000 - window_size) // (window_size - overlap) + 1
    print(f"预估时间帧数: {estimated_frames}")
    print(f"每帧时间: {(window_size - overlap) / Fs:.3f}秒")
    
    # 数据集列表
    datasets = [
        ('healthy_men_before.mat', 'combined_data'),
        ('healthy_men_after_add.mat', 'healthy_men_after')
    ]
    
    # 创建模型
    model = FrequencyDomainSCNN(spanning_type='2x2', lstm_hidden=256)
    
    # 测试第一个数据集
    first_path, first_var = datasets[0]
    test_data = process_emg_data(first_path, first_var)
    print(f"测试数据形状: {test_data.shape}")
    
    # 测试模型
    test_tensor = torch.tensor(test_data[:4], dtype=torch.float32)  # 取前4个样本测试
    test_features = model(test_tensor)
    print(f"测试输出形状: {test_features.shape}")
    
    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 处理所有数据集并保存特征
    for mat_path, var_name in datasets:
        try:
            print(f"\n处理文件: {mat_path}")
            features = process_data_with_model(mat_path, var_name, model, batch_size=4)
            
            # 保存特征到 .npy 文件
            out_name_npy = mat_path.replace('.mat', '_freq_scnn_features.npy')
            np.save(out_name_npy, features)
            print(f"保存特征到: {out_name_npy}")
            print(f"特征维度: {features.shape}")
            
        except Exception as e:
            print(f"处理 {mat_path} 时出错: {e}")
    
    print("\n=== 频域特征提取完成 ===")