from flask import Flask, render_template, jsonify, request,send_from_directory
from collectors.myo_collector import MyoManager
from collectors.realsense_collector import RealSenseCollector
from threading import Thread, Lock
import time
import base64
import cv2
import json
import os
import h5py
import numpy as np
from datetime import datetime

app = Flask(__name__)

# 设置视频文件目录路径
VIDEO_FOLDER = os.path.join(app.static_folder, 'videos')
# 确保视频目录存在
os.makedirs(VIDEO_FOLDER, exist_ok=True)


# 全局变量
myo_manager = None
realsense_collector = None
experiment_lock = Lock()
experiment_config = None
experiment_state = {
    'is_recording': False,
    'current_action': None,
    'current_trial': 0,
    'recording_start_time': None,
    'recorded_data': [],
    'subject_id': None
}



@app.route('/save_experiment_config', methods=['POST'])
def save_experiment_config():
    global experiment_config
    try:
        config = request.json
        
        # 验证配置
        if not validate_experiment_config(config):
            return jsonify({
                'status': 'error',
                'message': '实验配置无效'
            })
        
        # 创建数据存储目录
        data_path = os.path.join('data', 'raw', config['subjectId'])
        os.makedirs(data_path, exist_ok=True)
        
        # 保存配置
        experiment_config = config
        experiment_config['data_path'] = data_path
        
        return jsonify({
            'status': 'success',
            'message': '配置已保存'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

def validate_experiment_config(config):
    """验证实验配置"""
    required_fields = ['subjectId', 'dominantHand', 'actionDuration', 
                      'repeatTimes', 'restDuration', 'selectedActions']
    
    # 检查必填字段
    if not all(field in config for field in required_fields):
        return False
    
    # 验证数值范围
    if not (5 <= config['actionDuration'] <= 10):
        return False
    if not (3 <= config['repeatTimes'] <= 5):
        return False
    if not (3 <= config['restDuration'] <= 5):
        return False
    if not config['selectedActions']:
        return False
        
    return True

@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    global experiment_state, experiment_config
    try:
        if not experiment_config:
            return jsonify({
                'status': 'error',
                'message': '未找到实验配置'
            })
        
        # 初始化实验状态
        experiment_state = {
            'is_recording': False,
            'current_action': None,
            'current_trial': 0,
            'recording_start_time': None,
            'recorded_data': [],
            'subject_id': experiment_config['subjectId']
        }
        
        return jsonify({
            'status': 'success',
            'message': '实验已启动'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/record_action', methods=['POST'])
def record_action():
    global experiment_state
    try:
        data = request.json
        
        # 初始化记录状态
        with experiment_lock:
            experiment_state = {
                'is_recording': True,
                'current_action': data.get('action', 'manual_recording'),
                'current_trial': data.get('trial', int(time.time())),
                'recording_start_time': time.time(),
                'recorded_data': [],
                'subject_id': data.get('subject_id'),
                'dominant_hand': data.get('dominant_hand')
            }
        
        # 创建数据存储目录
        data_path = os.path.join('data', 'raw', experiment_state['subject_id'])
        os.makedirs(data_path, exist_ok=True)
        
        return jsonify({
            'status': 'success',
            'message': '开始记录'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global experiment_state
    try:
        with experiment_lock:
            if experiment_state['is_recording']:
                # 保存当前trial的数据
                save_trial_data()
                experiment_state['is_recording'] = False
        
        return jsonify({
            'status': 'success',
            'message': '记录已停止'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

def save_trial_data():
    """保存单次试验数据为HDF5格式"""
    global experiment_state
    
    if not experiment_state['recorded_data']:
        return
    
    try:
        # 构建文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{experiment_state['subject_id']}_{experiment_state['current_action']}_{timestamp}.h5"
        filepath = os.path.join('data', 'raw', experiment_state['subject_id'], filename)
        
        # 保存数据到HDF5文件
        with h5py.File(filepath, 'w') as f:
            # 创建元数据组
            metadata = f.create_group('metadata')
            metadata.attrs['subject_id'] = experiment_state['subject_id']
            metadata.attrs['dominant_hand'] = experiment_state['dominant_hand']
            metadata.attrs['action'] = experiment_state['current_action']
            metadata.attrs['trial_timestamp'] = experiment_state['current_trial']
            metadata.attrs['recording_start_time'] = experiment_state['recording_start_time']
            metadata.attrs['recording_end_time'] = time.time()
            
            # 创建数据组
            data = f.create_group('data')
            
            # 提取数据
            timestamps = []
            emg_raw_data = []
            emg_filtered_data = []
            hand_data_list = []
            
            # 添加调试输出
            print("开始处理记录数据...")
            print(f"记录数据点数: {len(experiment_state['recorded_data'])}")
            
            for frame in experiment_state['recorded_data']:
                timestamps.append(frame['timestamp'])
                emg_raw_data.append(frame['emg_data']['raw_emg'])
                emg_filtered_data.append(frame['emg_data']['filtered_emg'])
                
                # 处理手部数据，确保数据格式正确
                hand_data = frame.get('hand_data', {})
                if hand_data:
                    print("手部数据示例:", hand_data)  # 调试输出
                hand_data_list.append(hand_data)
            
            # 保存时间戳
            data.create_dataset('timestamps', data=np.array(timestamps))
            
            # 保存EMG数据
            data.create_dataset('emg_raw', data=np.array(emg_raw_data))
            data.create_dataset('emg_filtered', data=np.array(emg_filtered_data))
            
            # 保存手部数据
            # 首先获取所有可能的关节角度键
            all_keys = set()
            for hand_data in hand_data_list:
                all_keys.update(hand_data.keys())
            
            # 创建结构化的手部数据数组
            hand_angles = np.zeros((len(hand_data_list), len(all_keys)))
            key_list = sorted(list(all_keys))  # 确保键的顺序一致
            
            # 填充数据
            for i, hand_data in enumerate(hand_data_list):
                for j, key in enumerate(key_list):
                    hand_angles[i, j] = hand_data.get(key, 0)
            
            # 保存手部数据和关节名称
            hand_angles_dataset = data.create_dataset('hand_angles', data=hand_angles)
            hand_angles_dataset.attrs['joint_names'] = [str(k) for k in key_list]
            
            print(f"数据保存完成: {filepath}")
            print(f"保存的关节数量: {len(key_list)}")
            print("关节名称:", key_list)
            
        return True
        
    except Exception as e:
        print(f"保存数据错误: {e}")
        import traceback
        print(traceback.format_exc())
        raise

@app.route('/get_experiment_status')
def get_experiment_status():
    """获取实验状态"""
    global experiment_state, experiment_config
    
    if not experiment_config:
        return jsonify({
            'status': 'error',
            'message': '未找到实验配置'
        })
    
    return jsonify({
        'status': 'success',
        'is_recording': experiment_state['is_recording'],
        'current_action': experiment_state['current_action'],
        'current_trial': experiment_state['current_trial'],
        'config': experiment_config
    })


@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    """
    提供视频文件服务的路由
    :param filename: 视频文件名
    :return: 视频文件响应
    """
    try:
        return send_from_directory(VIDEO_FOLDER, filename)
    except Exception as e:
        app.logger.error(f"视频文件访问错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': '视频文件不存在或无法访问'
        }), 404
    
    
def process_realsense():
    global realsense_collector
    print("开始处理RealSense数据流")
    while realsense_collector.is_running:
        try:
            realsense_collector.process_frame()
            # time.sleep(0.01)
        except Exception as e:
            print(f"RealSense处理错误: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_data')
def get_data():
    global myo_manager, realsense_collector, experiment_state
    try:
        # 获取EMG数据
        emg_data = myo_manager.get_latest_data() if myo_manager else {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8}
        
        # 获取手部数据和相机帧
        hand_data = realsense_collector.get_hand_data() if realsense_collector else {}
        frame = realsense_collector.get_frame()
        
        # 获取相机统计信息
        camera_stats = {}
        if realsense_collector:
            try:
                camera_stats = realsense_collector.get_camera_stats()
            except Exception as e:
                print(f"获取相机统计信息错误: {e}")
                camera_stats = {'camera_fps': 0, 'camera_total_frames': 0}
        else:
            camera_stats = {'camera_fps': 0, 'camera_total_frames': 0}

        # 转换相机帧为base64
        frame_base64 = ''
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 准备响应数据
        response_data = {
            'status': 'success',
            'emg_data': emg_data,
            'hand_data': hand_data,
            'frame': frame_base64,
            'timestamp': time.time(),
            'camera_fps': camera_stats.get('camera_fps', 0),
            'camera_total_frames': camera_stats.get('camera_total_frames', 0)
        }
        
        # 如果正在记录，保存数据
        if experiment_state['is_recording']:
            experiment_state['recorded_data'].append({
                'timestamp': response_data['timestamp'],
                'emg_data': emg_data,
                'hand_data': hand_data
            })
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"获取数据错误: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e),
            'emg_data': {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8},
            'hand_data': {},
            'frame': '',
            'camera_fps': 0,
            'camera_total_frames': 0
        })

@app.route('/myo_status')
def myo_status():
    global myo_manager
    if myo_manager:
        status = myo_manager.get_status()
        return jsonify({
            'status': 'success',
            'connected': status['connected'],
            'synced': status['synced'],
            'frame_rate': status['frame_rate'],
            'total_frames': status['total_frames'],
        })
    return jsonify({
        'status': 'error',
        'message': 'Myo管理器未初始化'
    })

if __name__ == '__main__':
    try:
        # 创建数据存储目录
        os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
        
        # 初始化RealSense
        realsense_collector = RealSenseCollector()
        realsense_collector.start()
        
        # 启动RealSense处理线程
        realsense_thread = Thread(target=process_realsense)
        realsense_thread.daemon = True
        realsense_thread.start()
        
        # 初始化并启动Myo线程
        myo_manager = MyoManager()
        myo_thread = Thread(target=myo_manager.run_collection)
        myo_thread.daemon = True
        myo_thread.start()
        
        # 启动Flask服务器
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"启动错误: {e}")
    finally:
        if realsense_collector:
            realsense_collector.stop()