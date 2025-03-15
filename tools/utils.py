import os
import sys
import logging
import platform
import subprocess
import json
import time
from datetime import datetime

logger = logging.getLogger("utils")

def save_json(data, filepath):
    """保存JSON数据到文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")
        return False

def load_json(filepath):
    """从文件加载JSON数据"""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载JSON文件失败: {e}")
        return None

def get_timestamp():
    """获取当前时间戳"""
    return int(time.time())

def get_formatted_time():
    """获取格式化的时间字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def open_file(filepath):
    """使用系统默认程序打开文件"""
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', filepath])
        else:  # Linux
            subprocess.call(['xdg-open', filepath])
        return True
    except Exception as e:
        logger.error(f"打开文件失败: {e}")
        return False

def restart_application():
    """重启应用程序"""
    try:
        python = sys.executable
        os.execl(python, python, *sys.argv)
    except Exception as e:
        logger.error(f"重启应用程序失败: {e}")
        return False

def create_directory_if_not_exists(directory):
    """如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return True
    return False 