import threading
import time
import random
import pyautogui
import logging

from main import BasePlugin

logger = logging.getLogger("automove")

class AutomovePlugin(BasePlugin):
    def __init__(self, config):
        super().__init__(config)
        # 获取配置
        self.interval = config.getfloat('automove', 'interval', fallback=60.0)
        self.max_distance = config.getint('automove', 'max_distance', fallback=10)
        self.running = False
        self.thread = None
    
    def start(self):
        """启动自动移动"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._move_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("自动移动已启动")
    
    def stop(self):
        """停止自动移动"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("自动移动已停止")
    
    def _move_loop(self):
        """移动循环"""
        while self.running:
            try:
                # 获取当前鼠标位置
                current_x, current_y = pyautogui.position()
                
                # 随机偏移
                offset_x = random.randint(-self.max_distance, self.max_distance)
                offset_y = random.randint(-self.max_distance, self.max_distance)
                
                # 计算新位置
                new_x = current_x + offset_x
                new_y = current_y + offset_y
                
                # 移动鼠标
                pyautogui.moveTo(new_x, new_y, duration=0.5)
                logger.info(f"鼠标移动到 ({new_x}, {new_y})")
                
                # 等待
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"自动移动出错: {e}")
                time.sleep(5)  # 出错后等待一段时间再重试 