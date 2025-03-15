import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QGroupBox, QScrollArea, QTabWidget, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import logging

class PluginControlPanel(QWidget):
    def __init__(self, plugin_manager):
        super().__init__()
        self.plugin_manager = plugin_manager
        self.init_ui()
        
    def init_ui(self):
        # 创建布局
        layout = QVBoxLayout()
        
        # 加载所有插件
        for plugin_name in self.plugin_manager.plugins:
            plugin = self.plugin_manager.plugins[plugin_name]
            
            # 创建插件控制组
            group_box = QGroupBox(plugin_name.capitalize())
            group_layout = QVBoxLayout()
            
            # 添加启用/禁用复选框
            enabled_checkbox = QCheckBox("启用")
            enabled_checkbox.setChecked(plugin_name in self.plugin_manager.active_plugins)
            enabled_checkbox.toggled.connect(
                lambda checked, name=plugin_name: self.toggle_plugin(name, checked)
            )
            group_layout.addWidget(enabled_checkbox)
            
            # 添加优先级信息
            priority_label = QLabel(f"优先级: {plugin.priority}")
            group_layout.addWidget(priority_label)
            
            # 其他插件特定控件可以在这里添加
            
            group_box.setLayout(group_layout)
            layout.addWidget(group_box)
        
        # 添加一个伸展因子，使控件靠上排列
        layout.addStretch(1)
        
        self.setLayout(layout)
    
    def toggle_plugin(self, plugin_name, enabled):
        if enabled:
            self.plugin_manager.enable_plugin(plugin_name)
        else:
            self.plugin_manager.disable_plugin(plugin_name)

class LogHandler(logging.Handler, QObject):
    """自定义日志处理器，将日志发送到UI"""
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class LogViewerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # 创建并设置日志处理器
        self.log_handler = LogHandler()
        self.log_handler.log_signal.connect(self.append_log)
        
        # 获取根日志记录器并添加处理器
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 使用QTextEdit替代QLabel以获得更好的滚动支持
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.WidgetWidth)
        
        # 创建清除按钮
        clear_button = QPushButton("清除日志")
        clear_button.clicked.connect(self.clear_log)
        
        layout.addWidget(self.log_text)
        layout.addWidget(clear_button)
        self.setLayout(layout)
    
    def append_log(self, text):
        self.log_text.append(text)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def clear_log(self):
        self.log_text.clear()

class MainWindow(QMainWindow):
    def __init__(self, plugin_manager):
        super().__init__()
        self.plugin_manager = plugin_manager
        self.init_ui()
        
    def init_ui(self):
        # 设置窗口属性
        self.setWindowTitle('脚本管理器')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中央小部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 创建选项卡小部件
        tab_widget = QTabWidget()
        
        # 添加插件控制面板
        plugin_panel = PluginControlPanel(self.plugin_manager)
        tab_widget.addTab(plugin_panel, "插件管理")
        
        # 添加日志查看器
        log_viewer = LogViewerWidget()
        tab_widget.addTab(log_viewer, "日志")
        
        main_layout.addWidget(tab_widget)
        
        # 添加状态栏
        self.statusBar().showMessage('就绪')
        
        # 显示窗口
        self.show()

def start_ui(plugin_manager):
    app = QApplication(sys.argv)
    window = MainWindow(plugin_manager)
    sys.exit(app.exec_()) 