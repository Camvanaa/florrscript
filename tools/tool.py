import cv2
import numpy as np
import pyautogui
import mss
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def screenshot():
    """使用MSS库捕获屏幕"""
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

def get_hsv_value(event, x, y, flags, param):
    """鼠标回调函数，用于获取HSV值"""
    if event == cv2.EVENT_LBUTTONDOWN:
        img, hsv = param
        # 获取BGR和HSV值
        bgr_color = img[y, x]
        hsv_color = hsv[y, x]
        
        # 输出颜色信息
        logger.info(f"位置 ({x}, {y}):")
        logger.info(f"HSV: H={hsv_color[0]}, S={hsv_color[1]}, V={hsv_color[2]}")
        logger.info(f"BGR: B={bgr_color[0]}, G={bgr_color[1]}, R={bgr_color[2]}")
        
        # 将BGR转换为十六进制颜色代码
        hex_color = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"
        logger.info(f"HEX: {hex_color}")

def main():
    print("HSV颜色提取工具")
    print("1. 点击图像获取HSV值")
    print("2. 按'q'键退出")
    print("3. 按's'键保存当前图像")
    print("4. 按'p'键暂停/继续更新")
    
    last_update = 0
    update_interval = 0.1  # 100ms更新一次
    is_paused = False
    last_img = None
    last_hsv = None
    
    while True:
        current_time = time.time()
        
        # 如果没有暂停且达到更新间隔，更新图像
        if not is_paused and (current_time - last_update) >= update_interval:
            img = screenshot()
            if img is not None:
                last_img = img
                last_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                last_update = current_time
        
        # 如果有图像，显示它
        if last_img is not None:
            cv2.imshow('HSV Color Picker', last_img)
            cv2.setMouseCallback('HSV Color Picker', get_hsv_value, (last_img, last_hsv))
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and last_img is not None:
            # 保存图像
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, last_img)
            logger.info(f"图像已保存为: {filename}")
        elif key == ord('p'):
            # 切换暂停状态
            is_paused = not is_paused
            logger.info("图像更新已" + ("暂停" if is_paused else "继续"))
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 