from flask import Flask, render_template, jsonify
from collectors.myo_collector import MyoManager
from collectors.realsense_collector import RealSenseCollector
from threading import Thread
import time
import base64
import cv2

app = Flask(__name__)

# 全局变量
myo_manager = None
realsense_collector = None

def process_realsense():
    global realsense_collector
    print("开始处理RealSense数据流")
    while realsense_collector.is_running:
        try:
            realsense_collector.process_frame()
            time.sleep(0.01)
        except Exception as e:
            print(f"RealSense处理错误: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    global myo_manager, realsense_collector
    try:
        # 获取EMG数据
        emg_data = myo_manager.get_latest_data() if myo_manager else {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8}
        
        # 获取手部数据和相机帧
        hand_data = realsense_collector.get_hand_data() if realsense_collector else {}
        frame = realsense_collector.get_frame()
        
        # 转换相机帧为base64
        frame_base64 = ''
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'emg_data': emg_data,
            'hand_data': hand_data,
            'frame': frame_base64,
            'timestamp': time.time()
        })
    except Exception as e:
        print(f"获取数据错误: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'emg_data': {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8},
            'hand_data': {},
            'frame': ''
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