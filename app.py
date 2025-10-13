from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import threading
from werkzeug.utils import secure_filename
import pandas as pd
from io import BytesIO
import uuid
import time as time_module
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'recordings'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a'}

# 確保必要資料夾存在
os.makedirs('recordings', exist_ok=True)
os.makedirs('recordings_processed', exist_ok=True)

# 資料庫檔案
DB_FILE = 'processing_records.json'
FILENAME_MAP_FILE = 'filename_map.json'
ANALYTICS_FILE = 'analytics.json'


def load_filename_map():
    """載入文件名映射表"""
    if os.path.exists(FILENAME_MAP_FILE):
        with open(FILENAME_MAP_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_filename_map(mapping):
    """保存文件名映射表"""
    with open(FILENAME_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_records():
    """載入處理紀錄"""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_records(records):
    """儲存處理紀錄"""
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_analytics():
    """載入分析數據"""
    if os.path.exists(ANALYTICS_FILE):
        with open(ANALYTICS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'button_clicks': {},
        'daily_uploads': {},
        'hourly_uploads': {}
    }


def save_analytics(analytics):
    """保存分析數據"""
    with open(ANALYTICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2)


def track_button_click(button_name):
    """記錄按鈕點擊"""
    analytics = load_analytics()
    if 'button_clicks' not in analytics:
        analytics['button_clicks'] = {}
    analytics['button_clicks'][button_name] = analytics['button_clicks'].get(button_name, 0) + 1
    save_analytics(analytics)


def track_upload():
    """記錄上傳事件"""
    analytics = load_analytics()
    now = datetime.now()
    
    # 記錄每日上傳
    date_key = now.strftime('%Y-%m-%d')
    if 'daily_uploads' not in analytics:
        analytics['daily_uploads'] = {}
    analytics['daily_uploads'][date_key] = analytics['daily_uploads'].get(date_key, 0) + 1
    
    # 記錄每小時上傳
    hour_key = now.strftime('%Y-%m-%d %H:00')
    if 'hourly_uploads' not in analytics:
        analytics['hourly_uploads'] = {}
    analytics['hourly_uploads'][hour_key] = analytics['hourly_uploads'].get(hour_key, 0) + 1
    
    save_analytics(analytics)


def allowed_file(filename):
    """檢查檔案格式"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_audio_async(internal_filename, original_filename):
    """背景處理音檔"""
    from main import process_audio_file
    
    start_time = time_module.time()

    # 更新狀態為處理中
    records = load_records()
    for record in records:
        if record['音檔檔名'] == original_filename:
            record['處理狀態'] = '處理中'
            break
    save_records(records)

    try:
        # 執行處理
        audio_path = Path(app.config['UPLOAD_FOLDER']) / internal_filename
        result = process_audio_file(audio_path, enable_denoise=True, original_name=original_filename)

        # 計算處理時間
        processing_time = time_module.time() - start_time

        # 更新處理結果
        records = load_records()
        for record in records:
            if record['音檔檔名'] == original_filename:
                record['處理狀態'] = '完成'
                record['處理時間'] = f"{processing_time:.2f}秒"
                record['分類'] = result.get('class', '其他')
                record['摘要'] = result.get('abstract', '')
                record['意圖'] = ', '.join(result.get('intents', []))

                # 獲取主要情緒
                emotions = result.get('emotions', {})
                if emotions:
                    main_emotion = max(emotions.items(), key=lambda x: x[1])
                    record['主要情緒'] = f"{main_emotion[0]} ({main_emotion[1]}次)"

                # 找到逐字稿檔案和情緒分析檔案
                class_folder = {
                    "傷害健康保險": "class_disease",
                    "旅平險": "class_travel",
                    "車險": "class_car",
                    "其他": "class_other"
                }.get(result.get('class', '其他'), 'class_other')

                base_name = Path(original_filename).stem
                
                transcript_file = f"{class_folder}/voice_text/{base_name}.txt"
                if os.path.exists(transcript_file):
                    record['逐字稿連結'] = transcript_file

                emotion_file = f"{class_folder}/voice_emo/{base_name}_emotion.txt"
                if os.path.exists(emotion_file):
                    record['情緒分析連結'] = emotion_file

                break
        save_records(records)

    except Exception as e:
        # 處理失敗
        processing_time = time_module.time() - start_time
        records = load_records()
        for record in records:
            if record['音檔檔名'] == original_filename:
                record['處理狀態'] = '失敗'
                record['處理時間'] = f"{processing_time:.2f}秒"
                record['錯誤訊息'] = str(e)
                break
        save_records(records)


@app.route('/')
def index():
    """首頁"""
    records = load_records()

    # 計算統計資料
    stats = {
        'total': len(records),
        'completed': sum(1 for r in records if r['處理狀態'] == '完成'),
        'processing': sum(1 for r in records if r['處理狀態'] == '處理中'),
        'failed': sum(1 for r in records if r['處理狀態'] == '失敗')
    }

    # 按時間排序（最新的在前）
    records.sort(key=lambda x: x['上傳時間'], reverse=True)

    return render_template('index.html', records=records, stats=stats)


@app.route('/upload', methods=['POST'])
def upload_file():
    """上傳音檔（保留中文檔名）"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '沒有選擇檔案'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': '檔案名稱為空'})

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '不支援的檔案格式'})

    # 保留原始檔名（含中文）
    original_filename = file.filename
    
    # 生成唯一的內部檔名（用於存儲）
    file_ext = original_filename.rsplit('.', 1)[1].lower()
    internal_filename = f"{uuid.uuid4().hex}.{file_ext}"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], internal_filename)

    # 檢查原始檔名是否已存在
    records = load_records()
    if any(r['音檔檔名'] == original_filename for r in records):
        return jsonify({'success': False, 'error': '檔案已存在'})

    # 保存檔案
    file.save(filepath)

    # 保存檔名映射
    filename_map = load_filename_map()
    filename_map[original_filename] = internal_filename
    save_filename_map(filename_map)

    # 記錄上傳事件
    track_upload()

    # 新增處理紀錄
    records.append({
        '音檔檔名': original_filename,
        '上傳時間': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '處理時間': '-',
        '處理狀態': '等待處理',
        '分類': '-',
        '摘要': '-',
        '意圖': '-',
        '主要情緒': '-',
        '逐字稿連結': None,
        '情緒分析連結': None,
        '錯誤訊息': None
    })
    save_records(records)

    # 啟動背景處理
    thread = threading.Thread(target=process_audio_async, args=(internal_filename, original_filename))
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'message': f'檔案 {original_filename} 上傳成功，開始處理中...'
    })


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    """刪除檔案"""
    try:
        track_button_click('刪除')
        
        # 獲取內部檔名
        filename_map = load_filename_map()
        internal_filename = filename_map.get(filename)
        
        if internal_filename:
            # 刪除原始音檔
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], internal_filename)
            if os.path.exists(original_path):
                os.remove(original_path)

            # 從映射表中刪除
            del filename_map[filename]
            save_filename_map(filename_map)

        # 刪除處理後的音檔
        base_name = Path(filename).stem
        processed_path = Path('recordings_processed') / f"{base_name}_processed_denoised.wav"
        if processed_path.exists():
            processed_path.unlink()

        processed_path_no_denoise = Path('recordings_processed') / f"{base_name}_processed.wav"
        if processed_path_no_denoise.exists():
            processed_path_no_denoise.unlink()

        # 刪除逐字稿和情緒分析檔案
        for class_folder in ['class_disease', 'class_travel', 'class_car', 'class_other']:
            text_file = Path(class_folder) / 'voice_text' / f"{base_name}.txt"
            if text_file.exists():
                text_file.unlink()

            emo_file = Path(class_folder) / 'voice_emo' / f"{base_name}_emotion.txt"
            if emo_file.exists():
                emo_file.unlink()

        # 更新紀錄
        records = load_records()
        records = [r for r in records if r['音檔檔名'] != filename]
        save_records(records)

        return jsonify({'success': True, 'message': f'已刪除 {filename}'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/transcript/<path:filepath>')
def view_transcript(filepath):
    """查看逐字稿"""
    track_button_click('查看逐字稿')
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f'<pre style="font-family: monospace; white-space: pre-wrap; padding: 20px;">{content}</pre>'
    except Exception as e:
        return f'無法載入逐字稿: {str(e)}', 404


@app.route('/download/transcript/<path:filepath>')
def download_transcript(filepath):
    """下載逐字稿檔案"""
    track_button_click('下載逐字稿')
    try:
        filename = Path(filepath).name
        return send_file(
            filepath,
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return f'無法下載逐字稿: {str(e)}', 404


@app.route('/emotion/<path:filepath>')
def view_emotion(filepath):
    """查看情緒分析"""
    track_button_click('查看情緒')
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f'<pre style="font-family: monospace; white-space: pre-wrap; padding: 20px;">{content}</pre>'
    except Exception as e:
        return f'無法載入情緒分析: {str(e)}', 404


@app.route('/download/emotion/<path:filepath>')
def download_emotion(filepath):
    """下載情緒分析檔案"""
    track_button_click('下載情緒')
    try:
        filename = Path(filepath).name
        return send_file(
            filepath,
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return f'無法下載情緒分析: {str(e)}', 404


@app.route('/download/excel')
def download_excel():
    """下載Excel報表"""
    track_button_click('下載Excel')
    records = load_records()

    if not records:
        return '沒有資料可下載', 404

    # 轉換為DataFrame
    df = pd.DataFrame(records)

    # 移除不需要的欄位
    columns_to_remove = ['逐字稿連結', '情緒分析連結', '錯誤訊息']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # 建立Excel檔案
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='處理紀錄')

        # 調整欄寬
        worksheet = writer.sheets['處理紀錄']
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'音檔處理紀錄_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    )


@app.route('/api/stats')
def get_stats():
    """取得統計資料API"""
    records = load_records()

    stats = {
        'total': len(records),
        'completed': sum(1 for r in records if r['處理狀態'] == '完成'),
        'processing': sum(1 for r in records if r['處理狀態'] == '處理中'),
        'failed': sum(1 for r in records if r['處理狀態'] == '失敗'),
        'class_distribution': {}
    }

    # 分類統計
    for record in records:
        if record['處理狀態'] == '完成':
            class_name = record.get('分類', '其他')
            stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1

    return jsonify(stats)


@app.route('/api/analytics')
def get_analytics():
    """取得詳細分析數據"""
    analytics = load_analytics()
    records = load_records()
    
    # 意圖統計
    intent_stats = {}
    for record in records:
        if record['處理狀態'] == '完成' and record.get('意圖') != '-':
            intents = record['意圖'].split(', ')
            for intent in intents:
                intent_stats[intent] = intent_stats.get(intent, 0) + 1
    
    # 每日上傳統計（最近7天）
    daily_data = []
    for i in range(6, -1, -1):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        count = analytics.get('daily_uploads', {}).get(date, 0)
        daily_data.append({'date': date, 'count': count})
    
    # 每小時上傳統計（今天）
    hourly_data = []
    today = datetime.now().strftime('%Y-%m-%d')
    for hour in range(24):
        hour_key = f"{today} {hour:02d}:00"
        count = analytics.get('hourly_uploads', {}).get(hour_key, 0)
        hourly_data.append({'hour': f"{hour:02d}:00", 'count': count})
    
    return jsonify({
        'button_clicks': analytics.get('button_clicks', {}),
        'intent_stats': intent_stats,
        'daily_uploads': daily_data,
        'hourly_uploads': hourly_data
    })

@app.route('/api/reset_analytics', methods=['POST'])
def reset_analytics():
    """重置數據分析"""
    try:
        # 清空分析數據
        analytics = {
            'button_clicks': {},
            'daily_uploads': {},
            'hourly_uploads': {}
        }
        save_analytics(analytics)
        
        return jsonify({
            'success': True,
            'message': '數據分析已重置'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)