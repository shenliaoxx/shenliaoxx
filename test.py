import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pandas as pd
from datetime import datetime
import seaborn as sns

class DataViewer:
    def __init__(self, file_path):
        """初始化数据查看器"""
        self.file_path = Path(file_path)
        self.data = None
        self.metadata = None
        self.load_data()

    def load_data(self):
        """加载HDF5文件数据"""
        try:
            with h5py.File(self.file_path, 'r') as f:
                # 读取元数据
                self.metadata = {
                    'subject_id': f['metadata'].attrs['subject_id'],
                    'dominant_hand': f['metadata'].attrs['dominant_hand'],
                    'action': f['metadata'].attrs['action'],
                    'trial_timestamp': f['metadata'].attrs['trial_timestamp'],
                    'recording_start_time': f['metadata'].attrs['recording_start_time'],
                    'recording_end_time': f['metadata'].attrs['recording_end_time']
                }

                # 读取数据
                self.data = {
                    'timestamps': np.array(f['data/timestamps']),
                    'emg_raw': np.array(f['data/emg_raw']),
                    'emg_filtered': np.array(f['data/emg_filtered']),
                    'hand_angles': np.array(f['data/hand_angles'])
                }

        except Exception as e:
            print(f"读取文件错误: {e}")
            raise

    def print_summary(self):
        """打印数据摘要信息"""
        print("\n=== 数据文件摘要 ===")
        print(f"文件路径: {self.file_path}")
        print("\n--- 元数据 ---")
        print(f"受试者ID: {self.metadata['subject_id']}")
        print(f"惯用手: {self.metadata['dominant_hand']}")
        print(f"动作类型: {self.metadata['action']}")
        
        # 转换时间戳为可读格式
        start_time = datetime.fromtimestamp(self.metadata['recording_start_time'])
        end_time = datetime.fromtimestamp(self.metadata['recording_end_time'])
        duration = end_time - start_time
        
        print(f"记录开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"记录结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"记录持续时间: {duration.total_seconds():.2f} 秒")
        
        print("\n--- 数据统计 ---")
        print(f"采样点数: {len(self.data['timestamps'])}")
        print(f"采样率: {(len(self.data['timestamps'])-1)/duration.total_seconds():.2f} Hz")
        
        # EMG数据统计
        print("\nEMG数据统计:")
        emg_stats = pd.DataFrame({
            'Raw_Mean': np.mean(self.data['emg_raw'], axis=0),
            'Raw_Std': np.std(self.data['emg_raw'], axis=0),
            'Filtered_Mean': np.mean(self.data['emg_filtered'], axis=0),
            'Filtered_Std': np.std(self.data['emg_filtered'], axis=0)
        })
        print(emg_stats.round(2))

    def plot_emg_data(self, channel=None, filtered=True):
        """绘制EMG数据
        
        Args:
            channel: 要绘制的通道号（0-7），None表示绘制所有通道
            filtered: 是否显示滤波后的数据
        """
        emg_data = self.data['emg_filtered'] if filtered else self.data['emg_raw']
        time_points = self.data['timestamps'] - self.data['timestamps'][0]
        
        if channel is not None:
            # 绘制单个通道
            plt.figure(figsize=(12, 4))
            plt.plot(time_points, emg_data[:, channel])
            plt.title(f"通道 {channel+1} EMG数据 ({'滤波后' if filtered else '原始'})")
            plt.xlabel("时间 (秒)")
            plt.ylabel("幅值")
            plt.grid(True)
        else:
            # 绘制所有通道
            fig, axes = plt.subplots(8, 1, figsize=(12, 16), sharex=True)
            fig.suptitle(f"所有通道EMG数据 ({'滤波后' if filtered else '原始'})")
            
            for i in range(8):
                axes[i].plot(time_points, emg_data[:, i])
                axes[i].set_title(f"通道 {i+1}")
                axes[i].grid(True)
            
            axes[-1].set_xlabel("时间 (秒)")
            plt.tight_layout()

    def plot_hand_angles(self, joint_type=None):
        """绘制手部关节角度数据
        
        Args:
            joint_type: 要绘制的关节类型（'mcp', 'pip', 'dip'），None表示所有关节
        """
        time_points = self.data['timestamps'] - self.data['timestamps'][0]
        hand_data = self.data['hand_angles']
        
        # 提取所有关节名称
        joint_names = []
        for data_point in hand_data:
            if isinstance(data_point, dict):  # 确保数据点是字典格式
                joint_names.extend(data_point.keys())
        joint_names = list(set(joint_names))  # 去重
        
        if not joint_names:
            print("未找到关节角度数据")
            return
            
        if joint_type:
            # 筛选指定类型的关节
            selected_joints = [j for j in joint_names if joint_type in j]
            
            plt.figure(figsize=(12, 6))
            for joint in selected_joints:
                angles = [d.get(joint, 0) for d in hand_data]
                plt.plot(time_points, angles, label=joint)
            
            plt.title(f"{joint_type.upper()} 关节角度")
            plt.xlabel("时间 (秒)")
            plt.ylabel("角度 (度)")
            plt.legend()
            plt.grid(True)
        else:
            # 绘制所有关节数据
            joint_types = list(set(j.split('_')[1] for j in joint_names))
            fig, axes = plt.subplots(len(joint_types), 1, figsize=(12, 4*len(joint_types)))
            
            for i, jt in enumerate(joint_types):
                selected_joints = [j for j in joint_names if f"_{jt}_" in j]
                for joint in selected_joints:
                    angles = [d.get(joint, 0) for d in hand_data]
                    axes[i].plot(time_points, angles, label=joint)
                
                axes[i].set_title(f"{jt.upper()} 关节角度")
                axes[i].legend()
                axes[i].grid(True)
            
            axes[-1].set_xlabel("时间 (秒)")
            plt.tight_layout()

    def plot_data_overview(self):
        """绘制数据总览图"""
        plt.figure(figsize=(15, 10))
        
        # EMG数据热图
        plt.subplot(2, 1, 1)
        sns.heatmap(self.data['emg_filtered'].T, 
                    cmap='viridis',
                    xticklabels=100,  # 每100个采样点显示一个刻度
                    yticklabels=[f"Channel {i+1}" for i in range(8)])
        plt.title("EMG数据热图 (滤波后)")
        plt.xlabel("采样点")
        plt.ylabel("EMG通道")
        
        # 手部关节角度数据
        plt.subplot(2, 1, 2)
        hand_data = self.data['hand_angles']
        if isinstance(hand_data[0], dict):
            # 获取所有关节名称
            joint_names = list(hand_data[0].keys())
            angles_data = np.array([[d.get(joint, 0) for joint in joint_names] for d in hand_data])
            
            sns.heatmap(angles_data.T,
                        cmap='coolwarm',
                        xticklabels=100,
                        yticklabels=joint_names)
            plt.title("手部关节角度数据热图")
            plt.xlabel("采样点")
            plt.ylabel("关节")
        
        plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description='HDF5数据文件查看器')
    parser.add_argument(r'E:\multimodel_collection\data\raw\1\1_manual_recording_20250311_123808.h5', help='HDF5文件路径')
    parser.add_argument('--0', type=int, choices=range(8), help='要查看的EMG通道号(0-7)')
    parser.add_argument('--joint', choices=['mcp', 'pip', 'dip'], help='要查看的关节类型')
    parser.add_argument('--raw', action='store_true', help='显示原始EMG数据')
    parser.add_argument('--overview', action='store_true', help='显示数据总览')
    
    args = parser.parse_args()
    
    viewer = DataViewer(args.file_path)
    viewer.print_summary()
    
    if args.overview:
        viewer.plot_data_overview()
    else:
        if args.channel is not None:
            viewer.plot_emg_data(channel=args.channel, filtered=not args.raw)
        else:
            viewer.plot_emg_data(filtered=not args.raw)
            
        if args.joint:
            viewer.plot_hand_angles(joint_type=args.joint)
        else:
            viewer.plot_hand_angles()
    
    plt.show()

if __name__ == '__main__':
    main()