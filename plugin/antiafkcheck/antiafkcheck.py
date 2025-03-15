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
        """检测屏幕上的半透明黑色AFK检测窗口"""
        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
            # 使用更宽松的阈值
            _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
            
            # 保存原始二值化图像
            cv2.imwrite(f"thresh_original_{int(time.time())}.png", thresh)
            
            # 1. 使用形态学操作去除噪点
            kernel_small = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
            
            # 2. 寻找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建调试图像
            debug_img = screen.copy()
            
            # 筛选潜在的AFK窗口
            potential_boxes = []
            
            for contour in contours:
                # 获取轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算面积
                area = cv2.contourArea(contour)
                
                # 计算区域内像素的平均灰度值
                roi = gray[y:y+h, x:x+w]
                mean_val = np.mean(roi)
                
                # 计算纵横比
                aspect_ratio = w / h if h > 0 else 0
                
                # 计算置信度分数
                confidence = 0
                
                # 放宽基本筛选条件
                if (100 < w < 600 and 100 < h < 600 and  # 更宽松的尺寸范围
                    0.5 < aspect_ratio < 2.0 and  # 更宽松的纵横比
                    10000 < area < 300000):  # 更宽松的面积范围
                    
                    # 1. 尺寸评分
                    if (200 < w < 400 and 200 < h < 400):
                        confidence += 2
                    elif (150 < w < 450 and 150 < h < 450):
                        confidence += 1
                    
                    # 2. 纵横比评分
                    if 0.8 < aspect_ratio < 1.2:
                        confidence += 2
                    
                    # 3. 灰度值评分（AFK窗口通常是半透明黑色）
                    if 60 < mean_val < 140:
                        confidence += 2
                    
                    # 4. 检查是否有文本"AFK Check"
                    # 裁剪上部区域进行文本检测
                    top_region = thresh[y:y+int(h*0.2), x:x+w]
                    if np.count_nonzero(top_region) > top_region.size * 0.1:
                        confidence += 2
                    
                    # 在调试图像上绘制轮廓和分数
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(debug_img, f"Score: {confidence}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 保存潜在窗口信息
                    potential_boxes.append((x, y, w, h, confidence))
            
            # 保存调试图像
            cv2.imwrite(f"afk_detection_debug_{int(time.time())}.png", debug_img)
            
            # 选择置信度最高的窗口
            if potential_boxes:
                best_box = max(potential_boxes, key=lambda x: x[4])
                x, y, w, h, confidence = best_box
                
                logger.info(f"检测到可能的AFK窗口: x={x}, y={y}, w={w}, h={h}, 置信度={confidence}")
                
                # 降低置信度要求
                if confidence >= 2:  # 从3降低到2
                    # 调整窗口位置，排除左侧和上方突起
                    new_x = int(x + w * 0.05)
                    new_y = int(y + h * 0.05)
                    new_w = int(w * 0.9)
                    new_h = int(h * 0.9)
                    
                    return (new_x, new_y, new_w, new_h)
            
            logger.error("未检测到AFK窗口")
            return None
            
        except Exception as e:
            logger.error(f"检测AFK窗口时出错: {e}")
            traceback.print_exc()
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
            roi = roi[1:h-1, 1:w-1]
            
            # 转为灰度图并二值化
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
            
            # 保存原始二值化图像
            cv2.imwrite(f"binary_thresh_original_{int(time.time())}.png", thresh)
            
            # 首先获取小球位置
            ball_pos, _ = self._find_ball_position(roi)
            if not ball_pos:
                logger.error("无法检测到小球位置，无法规划路径")
                return []
            
            ball_x, ball_y = ball_pos
            logger.info(f"找到小球位置: ({ball_x}, {ball_y})")
            
            # 尝试找到并填充黑色小球
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
                        # 将黑色小球区域涂白
                        cv2.drawContours(thresh, [contour], -1, 255, -1)
                        end_ball = (cx, cy)
                        logger.info(f"找到终点黑色小球位置: ({cx}, {cy})")
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
            
            # 2. 根据轨道宽度创建kernel
            kernel_size = max(3, track_width - 2)  # 确保最小是3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # 3. 处理白色区域内部的黑色
            white_regions = thresh.copy()
            # 先用小kernel
            kernel_small = np.ones((3,3), np.uint8)
            white_regions = cv2.morphologyEx(white_regions, cv2.MORPH_CLOSE, kernel_small)
            # 再用与轨道宽度匹配的kernel
            white_regions = cv2.morphologyEx(white_regions, cv2.MORPH_CLOSE, kernel)
            
            # 4. 处理外部黑色区域
            # 使用floodfill从边界填充，得到外部区域
            h, w = thresh.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            external_black = np.zeros((h, w), np.uint8)
            external_black[:] = 255
            cv2.floodFill(external_black, mask, (0,0), 0)
            
            # 获取外部黑色区域
            black_mask = cv2.bitwise_not(thresh)
            external_black = cv2.bitwise_and(black_mask, external_black)
            
            # 处理外部黑色区域
            external_black = cv2.morphologyEx(external_black, cv2.MORPH_CLOSE, kernel_small)
            external_black = cv2.morphologyEx(external_black, cv2.MORPH_OPEN, kernel)
            external_black = cv2.morphologyEx(external_black, cv2.MORPH_CLOSE, kernel_small)
            
            # 5. 合并结果
            # 使用处理后的白色区域
            thresh = white_regions.copy()
            # 添加处理后的外部黑色区域
            thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(external_black))
            
            # 保存清理后的二值化图像
            cv2.imwrite(f"binary_thresh_cleaned_{int(time.time())}.png", thresh)
            
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
            
            # 找到最近的轨道点作为起点
            start_point = min(track_points, 
                             key=lambda p: np.sqrt((p[0] - ball_x)**2 + (p[1] - ball_y)**2))
            
            # 改进的路径追踪算法
            def follow_track(track_points, start):
                """改进的路径追踪算法，只允许相邻像素移动"""
                path = [start]
                current = start
                visited = {start}
                track_points = set(track_points)  # 转换为集合以提高查找效率
                
                # 八个相邻方向
                directions = [
                    (-1, -1), (0, -1), (1, -1),
                    (-1,  0),          (1,  0),
                    (-1,  1), (0,  1), (1,  1)
                ]
                
                while True:
                    # 只在8个相邻位置寻找下一个点
                    neighbors = []
                    for dx, dy in directions:
                        nx, ny = current[0] + dx, current[1] + dy
                        if (nx, ny) in track_points and (nx, ny) not in visited:
                            neighbors.append((nx, ny))
                    
                    if not neighbors:
                        # 如果没有未访问的相邻点，尝试回溯
                        backtrack_found = False
                        for prev_point in reversed(path[:-1]):  # 从倒数第二个点开始回溯
                            # 检查这个回溯点的邻居
                            for dx, dy in directions:
                                nx, ny = prev_point[0] + dx, prev_point[1] + dy
                                if (nx, ny) in track_points and (nx, ny) not in visited:
                                    # 找到新的可行路径，从这个点重新开始
                                    path = path[:path.index(prev_point) + 1]  # 截断到回溯点
                                    current = prev_point
                                    backtrack_found = True
                                    break
                            if backtrack_found:
                                break
                        
                        if not backtrack_found:
                            break  # 如果回溯也找不到路径，则结束
                        continue
                    
                    # 选择最佳的下一个点
                    next_point = None
                    min_cost = float('inf')
                    
                    # 计算当前路径的主要方向（使用最后几个点）
                    path_direction = [0, 0]
                    if len(path) >= 3:
                        for i in range(1, 4):
                            if i >= len(path):
                                break
                            path_direction[0] += path[-i][0] - path[-i-1][0]
                            path_direction[1] += path[-i][1] - path[-i-1][1]
                    
                    for nx, ny in neighbors:
                        dx = nx - current[0]
                        dy = ny - current[1]
                        
                        # 计算与当前路径方向的一致性
                        direction_cost = 0
                        if path_direction != [0, 0]:
                            # 归一化方向向量
                            path_mag = math.sqrt(path_direction[0]**2 + path_direction[1]**2)
                            if path_mag > 0:
                                dot_product = (dx * path_direction[0] + dy * path_direction[1]) / path_mag
                                direction_cost = -dot_product  # 负点积作为方向代价
                        
                        # 总代价是距离和方向变化的加权和
                        cost = math.sqrt(dx*dx + dy*dy) + direction_cost * 0.5
                        
                        if cost < min_cost:
                            min_cost = cost
                            next_point = (nx, ny)
                    
                    if next_point is None:  # 不应该发生，但以防万一
                        break
                    
                    current = next_point
                    path.append(current)
                    visited.add(current)
                
                return path
            
            # 获取完整路径
            track_path = follow_track(track_points, start_point)
            
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
