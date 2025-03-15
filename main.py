import os
import importlib
import logging
import threading
from typing import Dict, List
import configparser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 终端输出
    ]
)
logger = logging.getLogger("MainFramework")

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, 'BasePlugin'] = {}
        self.active_plugins: Dict[str, 'BasePlugin'] = {}
        self.config = self._load_config()
        
    def _load_config(self) -> configparser.ConfigParser:
        """加载配置文件"""
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        return config
    
    def load_plugins(self):
        """加载所有插件"""
        plugin_dir = "plugin"
        for folder in os.listdir(plugin_dir):
            folder_path = os.path.join(plugin_dir, folder)
            if os.path.isdir(folder_path) and not folder.startswith('__'):
                try:
                    # 导入插件模块
                    module = importlib.import_module(f"plugin.{folder}.{folder}")
                    # 获取插件类
                    plugin_class = getattr(module, f"{folder.capitalize()}Plugin")
                    # 实例化插件
                    plugin = plugin_class(self.config)
                    self.plugins[folder] = plugin
                    logger.info(f"加载插件 {folder} 成功")
                except Exception as e:
                    logger.error(f"加载插件 {folder} 失败: {e}")
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """启用插件"""
        if plugin_name in self.plugins and plugin_name not in self.active_plugins:
            plugin = self.plugins[plugin_name]
            try:
                plugin.start()
                self.active_plugins[plugin_name] = plugin
                logger.info(f"启用插件 {plugin_name} 成功")
                return True
            except Exception as e:
                logger.error(f"启用插件 {plugin_name} 失败: {e}")
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """禁用插件"""
        if plugin_name in self.active_plugins:
            plugin = self.active_plugins[plugin_name]
            try:
                plugin.stop()
                del self.active_plugins[plugin_name]
                logger.info(f"禁用插件 {plugin_name} 成功")
                return True
            except Exception as e:
                logger.error(f"禁用插件 {plugin_name} 失败: {e}")
        return False
    
    def get_active_plugins(self) -> List[str]:
        """获取所有活动的插件"""
        return list(self.active_plugins.keys())
    
    def update_config(self):
        """更新配置文件"""
        # 更新插件启用状态
        for plugin_name in self.plugins:
            self.config.set(plugin_name, 'enabled', 
                          str(plugin_name in self.active_plugins).lower())
        
        # 写入配置文件
        with open('config.ini', 'w', encoding='utf-8') as f:
            self.config.write(f)

class BasePlugin:
    """插件基类"""
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.priority = self._get_priority()
        self.name = self.__class__.__name__.lower().replace('plugin', '')
        self.logger = logging.getLogger(self.name)
    
    def _get_priority(self) -> int:
        """从配置文件获取优先级"""
        try:
            return self.config.getint(self.name, 'priority', fallback=100)
        except:
            return 100
    
    def start(self):
        """启动插件"""
        raise NotImplementedError
    
    def stop(self):
        """停止插件"""
        raise NotImplementedError

def main():
    # 创建插件管理器
    manager = PluginManager()
    
    # 加载所有插件
    manager.load_plugins()
    
    # 根据配置启用插件
    for plugin_name in manager.plugins:
        if manager.config.getboolean(plugin_name, 'enabled', fallback=False):
            manager.enable_plugin(plugin_name)
    
    # 启动UI
    try:
        from ui.main_window import start_ui
        # 在单独的线程中启动UI
        ui_thread = threading.Thread(target=start_ui, args=(manager,))
        ui_thread.daemon = True
        ui_thread.start()
        
        # 保持程序运行
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        # 停止所有活动的插件
        for plugin_name in manager.get_active_plugins():
            manager.disable_plugin(plugin_name)
        logger.info("程序已终止")
    except Exception as e:
        logger.error(f"程序出错: {e}")
    finally:
        # 更新配置
        manager.update_config()

if __name__ == "__main__":
    main()
