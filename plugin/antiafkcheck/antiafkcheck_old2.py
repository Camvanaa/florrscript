from main import BasePlugin
import cv2
import numpy as np
import pyautogui
import time
import logging
from threading import Thread
from scipy.ndimage import binary_dilation
import math
import mss  # 新增导入
import traceback
from cv2 import ximgproc
from collections import deque

Islog=True
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AFKCheckSolver:
    def __init__(self, check_interval=1):
        """初始化AFK检测求解器"""
        self.check_interval = check_interval
        self.running = False
        self.last_check_time = 0
        # 定义游戏中不同稀有度的小球颜色的HSV范围
        self.ball_color_ranges = [
            # 普通 #7ce96b - 绿色 finish HSV: H=56, S=138, V=236
            {"name": "普通(绿色)", "lower": np.array([55, 120, 200]), "upper": np.array([65, 140, 255])},
            # 罕见 #fbe25c - 黄色 finish HSV值: H=25, S=162, V=255
            {"name": "罕见(黄色)", "lower": np.array([20, 150, 200]), "upper": np.array([30, 175, 255])},
            # 稀有 #4d52e3 - 蓝色 finish HSV: H=119, S=168, V=227
            {"name": "稀有(蓝色)", "lower": np.array([110, 150, 200]), "upper": np.array([130, 175, 255])},
            # 史诗 #861fde - 紫色 finish HSV: H=136, S=219, V=222
            {"name": "史诗(紫色)", "lower": np.array([130, 200, 200]), "upper": np.array([150, 220, 255])},
            # 传奇 #db1f1f - 红色 (两个范围) finish HSV值: H=0, S=219, V=222
            {"name": "传奇(红色1)", "lower": np.array([0, 200, 200]), "upper": np.array([15, 255, 255])},
            {"name": "传奇(红色2)", "lower": np.array([160, 200, 200]), "upper": np.array([180, 255, 255])},
            # 神话 #1fdbde - 青色 finish HSV值: H=90, S=219, V=222
            {"name": "神话(青色)", "lower": np.array([80, 210, 200]), "upper": np.array([100, 220, 255])},
            # 究极 #fc2a74 - 粉红色 finish HSV: H=170, S=212, V=255
            {"name": "究极(粉红色)", "lower": np.array([160, 200, 200]), "upper": np.array([180, 220, 255])},
            # 超级 #2bffa3 - 绿松石色 finish HSV: H=77, S=212, V=255
            {"name": "超级(绿松石色)", "lower": np.array([60, 200, 200]), "upper": np.array([80, 220, 255])},
        ]
        
    def start(self):
        """启动AFK检测监控"""
        self.running = True
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("AFK检测监控已启动")
        
    def stop(self):
        """停止AFK检测监控"""
        self.running = False
        logger.info("AFK检测监控已停止")
        
    def _screenshot(self):
        """使用MSS库捕获屏幕，每次创建新实例避免线程问题"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # 通常是主显示器
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            logger.error(f"截图出错: {e}")
            # 备选方案：使用pyautogui截图
            try:
                screenshot = pyautogui.screenshot()
                screen = np.array(screenshot)
                return cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            except Exception as e2:
                logger.error(f"备选截图也出错: {e2}")
                return None
        
    def _screenshot_region(self, region):
        """捕获屏幕指定区域 (x, y, width, height)，每次创建新实例"""
        try:
            x, y, width, height = region
            with mss.mss() as sct:
                monitor = {"top": y, "left": x, "width": width, "height": height}
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            logger.error(f"区域截图出错: {e}")
            # 备选方案：使用pyautogui截图
            try:
                screenshot = pyautogui.screenshot(region=region)
                screen = np.array(screenshot)
                return cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            except Exception as e2:
                logger.error(f"备选区域截图也出错: {e2}")
                return None
    
    def _monitor_loop(self):
        """持续监控屏幕寻找AFK检测窗口"""
        while self.running:
            try:
                # 使用改进的截图方法
                screen = self._screenshot()
                if screen is None:
                    logger.error("获取屏幕截图失败，跳过此轮检测")
                    time.sleep(self.check_interval)
                    continue
                    
                # 检测AFK窗口
                afk_box = self._detect_afk_box(screen)
                if afk_box is not None:
                    logger.info("检测到AFK窗口，开始处理...")
                    self._solve_afk_check(afk_box)
                    
            except Exception as e:
                logger.error(f"监控过程出错: {e}")
                
            time.sleep(self.check_interval)
    
    def _detect_afk_box(self, screen):
        """检测屏幕上的黑色半透明AFK检测窗口"""
        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
            # 固定使用阈值130检测半透明黑色窗口
            _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
            
            # 使用形态学操作去除凸起
            # 1. 使用非常大的水平kernel进行开运算，专门去除水平方向的凸起
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_horizontal)
            
            # 2. 使用非常大的垂直kernel进行开运算，专门去除垂直方向的凸起
            kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)
            
            # 3. 使用方形kernel进行闭运算，填充主要区域
            kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_square)
            
            # 4. 最后再次使用开运算确保去除所有凸起
            kernel_final = np.ones((10, 10), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_final)
            
            # 保存处理后的阈值图像用于调试
            # cv2.imwrite(f"processed_thresh_{int(time.time())}.png", thresh)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选潜在的半透明黑色窗口
            best_box = None
            max_confidence = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                area = w * h
                
                # 特别针对AFK窗口的形状和大小
                if (150 < w < 400 and 150 < h < 400 and 
                    0.7 < aspect_ratio < 1.3 and area > 30000):
                    
                    # 计算矩形内部像素的平均值和标准差
                    roi = gray[y:y+h, x:x+w]
                    mean_val = np.mean(roi)
                    std_val = np.std(roi)
                    
                    # 评估该区域是否为半透明黑色
                    is_dark = 60 < mean_val < 140
                    is_uniform = std_val < 50
                    
                    confidence = 0
                    if is_dark:
                        confidence += 1
                    if is_uniform:
                        confidence += 1
                        
                    # 评估是否包含滑块元素
                    roi_color = screen[y:y+h, x:x+w]
                    slider_score = self._evaluate_slider_presence(roi_color)
                    confidence += slider_score
                    
                    # 选择置信度最高的窗口
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_box = (x, y, w, h)

            if best_box:
                logger.info(f"检测到半透明黑色AFK窗口，置信度={max_confidence}: x={best_box[0]}, y={best_box[1]}, w={best_box[2]}, h={best_box[3]}")
                return best_box
            
            logger.error("未检测到半透明黑色AFK窗口")
            return None
            
        except Exception as e:
            logger.error(f"检测AFK窗口时出错: {e}")
            return None
    
    def _evaluate_slider_presence(self, roi):
        """评估ROI中滑块元素的存在程度，返回0-3的分数"""
        try:
            # 转为HSV色彩空间以便检测特定颜色
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 检测红色小球 (扩大范围)
            red_lower1 = np.array([0, 30, 30])
            red_upper1 = np.array([20, 255, 255])
            red_lower2 = np.array([160, 30, 30])
            red_upper2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pixels = cv2.countNonZero(red_mask)
            
            # 检测灰色轨道
            lower_gray = np.array([0, 0, 80])
            upper_gray = np.array([180, 40, 200])
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
            gray_pixels = cv2.countNonZero(gray_mask)
            
            # 计算得分
            score = 0
            total_pixels = roi.shape[0] * roi.shape[1]
                
            # 灰色轨道检测
            gray_ratio = gray_pixels / total_pixels
            logger.info(f"灰色轨道检测 - 像素数: {gray_pixels}, 比例: {gray_ratio:.5f}")
            if gray_ratio > 0.03:  # 超过3%的像素是灰色
                score += 1
            
            logger.info(f"滑块元素评估得分: {score}")
            return score
        except Exception as e:
            logger.error(f"评估滑块存在时出错: {e}")
            return 0
    
    def _extract_track_path(self, roi):
        """提取滑动轨道并规划路径"""
        try:
            # 裁剪图像四周1像素
            h, w = roi.shape[:2]
            roi = roi[3:h-3, 3:w-3]
            
            # 转为灰度图并二值化
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
            
            # 保存原始二值化图像
            cv2.imwrite(f"binary_thresh_original_{int(time.time())}.png", thresh)
            
            # 首先获取小球位置
            ball_pos, ball_color = self._find_ball_position(roi)
            if not ball_pos:
                logger.error("无法检测到小球位置，无法规划路径")
                return []
            
            ball_x, ball_y = ball_pos
            logger.info(f"找到小球位置: ({ball_x}, {ball_y})")
            
            # 在二值化图像中填充起始小球区域为白色
            # 定义一个填充起始小球的函数（使用1.2倍大小）
            def fill_ball_area(img, center, base_radius=10, scale=1.4):
                cx, cy = center
                # 检查中心点是否已经是白色
                if img[cy, cx] == 255:
                    logger.info("小球位置已经是白色，无需填充")
                    return img
                radius = int(base_radius * scale)
                cv2.circle(img, (cx, cy), radius, 255, -1)  # 填充白色圆形
                return img
            
            # 填充起始点小球（1.2倍大小）
            thresh = fill_ball_area(thresh, (ball_x, ball_y))
            
            # 保存填充小球后的二值化图像
            cv2.imwrite(f"binary_thresh_filled_ball_{int(time.time())}.png", thresh)
            
            # 尝试找到并填充黑色小球（同样使用1.2倍大小）
            black_regions = cv2.bitwise_not(thresh)
            black_contours, _ = cv2.findContours(black_regions, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            end_ball = None
            for contour in black_contours:
                area = cv2.contourArea(contour)
                if 20 < area < 200:  # 调整大小范围
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # 估计小球半径
                        radius = int(np.sqrt(area / np.pi) * 1.2)  # 使用1.2倍大小
                        # 将黑色小球区域涂白（使用圆形而不是轮廓）
                        cv2.circle(thresh, (cx, cy), radius, 255, -1)
                        end_ball = (cx, cy)
                        logger.info(f"找到终点黑色小球位置: ({cx}, {cy})，填充半径: {radius}")
                        break
            
            # 保存填充后的二值化图像
            cv2.imwrite(f"binary_thresh_filled_{int(time.time())}.png", thresh)
            
            # 1. 计算轨道的平均宽度
            def calculate_track_width(binary_img):
                # 获取白色连通区域
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
                
                # 计算每个连通区域的平均宽度
                widths = []
                for i in range(1, num_labels):  # 跳过背景（标签0）
                    # 获取当前连通区域
                    current_region = (labels == i).astype(np.uint8) * 255
                    # 计算距离变换
                    dist = cv2.distanceTransform(current_region, cv2.DIST_L2, 5)
                    # 获取最大距离（即最大半径）
                    max_dist = np.max(dist)
                    # 宽度是半径的两倍
                    widths.append(max_dist * 2)
                
                # 返回中位数宽度
                return int(np.median(widths)) if widths else 3
            
            track_width = calculate_track_width(thresh)
            logger.info(f"检测到轨道宽度: {track_width}")
            
            # 创建小kernel用于细节处理
            kernel_small = np.ones((2,2), np.uint8)  # 使用2x2的kernel
            
            # 处理白色区域内部的黑色
            white_regions = thresh.copy()
            
            # 先保存原始轨道末端位置
            endpoints = white_regions.copy()
            
            # 使用小kernel进行闭运算，只处理小的黑点
            white_regions = cv2.morphologyEx(white_regions, cv2.MORPH_CLOSE, kernel_small)
            
            # 保存处理后的白色区域
            cv2.imwrite(f"white_regions_processed_{int(time.time())}.png", white_regions)
            
            # 使用flood fill处理内部的黑色区域
            h, w = white_regions.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            
            # 从边界开始flood fill，标记外部区域为128
            fill_img = white_regions.copy()
            for x in range(w):
                if fill_img[0, x] == 0:  # 顶边
                    cv2.floodFill(fill_img, mask, (x, 0), 128)
                if fill_img[h-1, x] == 0:  # 底边
                    cv2.floodFill(fill_img, mask, (x, h-1), 128)
            
            for y in range(h):
                if fill_img[y, 0] == 0:  # 左边
                    cv2.floodFill(fill_img, mask, (0, y), 128)
                if fill_img[y, w-1] == 0:  # 右边
                    cv2.floodFill(fill_img, mask, (w-1, y), 128)
            
            # 将所有非128的黑色区域（内部黑色）填充为白色
            internal_black = (fill_img == 0)
            white_regions[internal_black] = 255
            
            # 确保轨道末端不会被意外填充
            # 使用膨胀操作来稍微扩展轨道末端
            kernel_endpoint = np.ones((3,3), np.uint8)
            endpoints = cv2.dilate(endpoints, kernel_endpoint, iterations=1)
            
            # 合并原始末端
            white_regions = cv2.bitwise_or(white_regions, endpoints)
            
            # 保存最终处理结果
            cv2.imwrite(f"white_regions_final_{int(time.time())}.png", white_regions)
            
            # 更新thresh用于后续处理
            thresh = white_regions.copy()
            
            # 骨架化
            skeleton = ximgproc.thinning(thresh)
            cv2.imwrite(f"skeleton_{int(time.time())}.png", skeleton)
            
            # 获取所有轨道点
            track_points = np.where(skeleton > 0)
            track_points = list(zip(track_points[1], track_points[0]))  # 转换为(x,y)坐标列表
            track_points = set(track_points)  # 转换为集合以加快查找
            
            if not track_points:
                logger.error("未找到有效轨道点")
                return []
            
            # 获取骨架点
            skeleton_points = []
            for y in range(h):
                for x in range(w):
                    if skeleton[y, x] > 0:
                        skeleton_points.append((x, y))
            
            if not skeleton_points:
                logger.error("未找到骨架点，无法继续规划路径")
                return []
            
            # 找到距离起点最近的骨架点作为开始
            start_point = min(skeleton_points, key=lambda p: np.sqrt((p[0] - ball_x)**2 + (p[1] - ball_y)**2))
            
            def follow_track(skeleton, start_point):
                """严格按照骨架上连续的点行走"""
                path = []
                h, w = skeleton.shape[:2]
                
                # 找到距离起点最近的骨架点作为实际起点
                min_dist = float('inf')
                current = None
                for y in range(h):
                    for x in range(w):
                        if skeleton[y, x] > 0:
                            dist = abs(x - start_point[0]) + abs(y - start_point[1])
                            if dist < min_dist:
                                min_dist = dist
                                current = (x, y)
                
                if not current:
                    return [start_point]
                
                # 标记已访问的点
                visited = np.zeros_like(skeleton)
                
                # 从起点开始，严格按照相邻点行走
                while current:
                    path.append(current)
                    visited[current[1], current[0]] = 1
                    
                    # 在当前点的8邻域中找下一个未访问的骨架点
                    next_point = None
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (-1,-1), (1,-1)]:  # 优先直线方向
                        nx, ny = current[0] + dx, current[1] + dy
                        if (0 <= nx < w and 0 <= ny < h and 
                            skeleton[ny, nx] > 0 and 
                            visited[ny, nx] == 0):
                            next_point = (nx, ny)
                            break
                    
                    current = next_point
                
                logger.info(f"找到路径，长度: {len(path)}")
                return path
            
            # 获取完整路径
            track_path = follow_track(skeleton, start_point)
            
            # 验证路径的连续性
            for i in range(1, len(track_path)):
                prev = track_path[i-1]
                curr = track_path[i]
                dx = abs(curr[0] - prev[0])
                dy = abs(curr[1] - prev[1])
                if dx > 1 or dy > 1:
                    logger.error(f"检测到不连续点: 从 {prev} 到 {curr}")
                    # 在这里可以break或抛出异常
            
            # 构建最终路径，确保平滑过渡
            path = []
            
            # 添加从起始小球到轨道的平滑过渡
            dx = track_path[0][0] - ball_x
            dy = track_path[0][1] - ball_y
            steps = max(abs(dx), abs(dy))
            if steps > 0:
                for i in range(steps + 1):
                    t = i / steps
                    x = int(ball_x + dx * t)
                    y = int(ball_y + dy * t)
                    path.append((x, y))
            
            # 添加轨道路径
            path.extend(track_path)
            
            # 如果有终点小球，添加到终点的平滑过渡
            if end_ball:
                # 找到最接近终点小球的轨道点
                end_track_point = min(track_path, 
                                    key=lambda p: np.sqrt((p[0] - end_ball[0])**2 + (p[1] - end_ball[1])**2))
                # 如果最后一个轨道点不是最接近终点的点，就不添加终点
                if end_track_point == track_path[-1]:
                    dx = end_ball[0] - track_path[-1][0]
                    dy = end_ball[1] - track_path[-1][1]
                    steps = max(abs(dx), abs(dy))
                    if steps > 0:
                        for i in range(1, steps + 1):
                            t = i / steps
                            x = int(track_path[-1][0] + dx * t)
                            y = int(track_path[-1][1] + dy * t)
                            path.append((x, y))
            
            # 绘制调试图像
            debug_img = roi.copy()
            # 绘制骨架
            debug_img[skeleton > 0] = [0, 255, 0]
            # 绘制路径点
            for point in path:
                cv2.circle(debug_img, point, 1, (0, 165, 255), -1)
            # 标记起点和终点
            cv2.circle(debug_img, (ball_x, ball_y), 5, (255, 0, 0), -1)  # 蓝色起点
            cv2.circle(debug_img, path[-1], 5, (0, 0, 255), -1)  # 红色终点
            cv2.imwrite(f"track_path_{int(time.time())}.png", debug_img)
            
            return path
            
        except Exception as e:
            logger.error(f"轨道路径提取出错: {e}\n{traceback.format_exc()}")
            return []
    
    def _solve_afk_check(self, box):
        """解决AFK检测，完成滑动操作"""
        try:
            logger.info("检测到AFK窗口，开始处理...")
            x, y, w, h = box
            logger.info(f"AFK窗口位置: x={x}, y={y}, w={w}, h={h}")
            
            screen = self._screenshot()
            roi = screen[y:y+h, x:x+w]
            
            # 保存ROI图像用于调试
            try:
                cv2.imwrite(f"afk_roi_{int(time.time())}.png", roi)
                logger.info("已保存ROI图像用于调试")
            except Exception as e:
                logger.error(f"保存ROI图像失败: {e}")
            
            # 提取轨道路径
            path = self._extract_track_path(roi)
            
            if path:
                # 将相对坐标转换为屏幕坐标
                screen_path = [(x + px + 1, y + py + 1) for px, py in path]  # 补偿裁剪的1像素
                
                # 添加小球位置作为起点
                start_pos = screen_path[0]
                
                logger.info(f"开始从 {start_pos} 沿轨道拖动小球，路径点数: {len(screen_path)}")
                
                # 移动到小球位置
                pyautogui.moveTo(start_pos[0], start_pos[1], duration=0.05)
                
                # 按下鼠标
                pyautogui.mouseDown()
                
                # 沿路径快速移动，每5个点取一个点
                for point in screen_path[::5]:  # 每5个点取一个点
                    pyautogui.moveTo(point[0], point[1], duration=0.001)  # 极短的移动时间
                
                # 确保移动到终点
                pyautogui.moveTo(screen_path[-1][0], screen_path[-1][1], duration=0.05)
                
                # 释放鼠标
                pyautogui.mouseUp()
                
                logger.info("AFK检测已完成")
                time.sleep(0.2)  # 短暂等待
            else:
                logger.error("无法生成有效路径")
        except Exception as e:
            logger.error(f"解决AFK检测时出错: {e}")
            logger.error(traceback.format_exc())
    
    def _find_ball_position(self, roi):
        """查找小球位置，支持所有颜色"""
        try:
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            best_ball = None
            best_circularity = 0
            ball_color = None
            
            # 遍历所有可能的小球颜色
            for color_range in self.ball_color_ranges:
                # 创建颜色掩码
                mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
                
                # 形态学操作，去除噪点
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # 找到所有轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # 计算轮廓面积和周长
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter == 0:
                        continue
                    
                    # 计算圆度
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # 输出每个轮廓的信息
                    logger.info(f"{color_range['name']} 轮廓 - 面积: {area}, 圆度: {circularity:.2f}")
                    
                    # 面积需要在合理范围内，且形状接近圆形
                    if 50 < area < 500 and circularity > 0.7:
                        if circularity > best_circularity:
                            best_circularity = circularity
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                best_ball = (cx, cy)
                                ball_color = color_range["name"]
                                # 输出找到的小球的具体HSV值
                                ball_hsv = hsv[cy, cx]
                                logger.info(f"找到{ball_color}小球的HSV值: H={ball_hsv[0]}, S={ball_hsv[1]}, V={ball_hsv[2]}")
                
            if best_ball:
                logger.info(f"检测到{ball_color}小球，位置: {best_ball}, 圆度: {best_circularity:.2f}")
                return best_ball, ball_color
                
            logger.error("未找到任何颜色的小球")
            return None, None
            
        except Exception as e:
            logger.error(f"查找小球位置时出错: {e}")
            return None, None

class AntiafkcheckPlugin(BasePlugin):
    """AFK检测自动处理插件"""
    def __init__(self, config):
        super().__init__(config)
        self.check_interval = config.getint('antiafkcheck', 'check_interval', fallback=2)
        self.solver = AFKCheckSolver(check_interval=self.check_interval)
    
    def start(self):
        """启动插件"""
        self.solver.start()
        logger.info(f"AFK检测处理插件已启动，检查间隔: {self.check_interval}秒")
    
    def stop(self):
        """停止插件"""
        self.solver.stop()
        logger.info("AFK检测处理插件已停止")
