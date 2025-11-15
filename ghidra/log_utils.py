"""
日志工具模块
统一封装日志功能，支持全局使用和配置
"""
import sys
import time
import logging
from loguru import logger

class LogWrapper:
    """日志系统包装类，统一对外提供日志接口"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """确保只创建一个实例（单例模式）"""
        if cls._instance is None:
            cls._instance = super(LogWrapper, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        """初始化日志系统，只执行一次"""
        if not LogWrapper._initialized:
            self.logger = None
            self.log_file = None
            self._configure_logging()
            LogWrapper._initialized = True
            
    def _configure_logging(self):
        """配置底层日志系统"""
        # 移除loguru默认处理器
        logger.remove()
        
        # 添加控制台输出
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # 添加文件输出
        # current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # self.log_file = f"decompiler_{current_time}.log"
        # logger.add(
        #     self.log_file,
        #     level="INFO",
        #     rotation="500 MB",  # 日志文件达到500MB时自动分割
        #     retention="7 days",  # 保留7天的日志
        #     encoding="utf-8",
        #     format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
        # )
        
        # 配置标准logging转发到loguru
        class LoguruHandler(logging.Handler):
            """将标准logging的日志转发到loguru"""
            def emit(self, record):
                try:
                    # 将标准logging的级别转换为loguru的级别
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                
                # 获取调用栈信息，确保日志显示正确的调用位置
                frame, depth = logging.currentframe(), 2
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                
                # 将日志传递给loguru
                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )
        
        # 配置根日志记录器，捕获所有标准logging日志
        logging.basicConfig(handlers=[LoguruHandler()], level=logging.DEBUG)
        
        # 设置第三方库的日志级别
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

        self._configVllmLogging()

        self.logger = logger


    def _configVllmLogging(self):
        # 这行导包必不可少，触发 vllm 初始化自己的日志系统（调用 init_logger）
        from vllm import LLM, SamplingParams

        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.CRITICAL)
        # vllm_logger.propagate = False

        # 移除所有处理器
        # for handler in vllm_logger.handlers[:]:
        #     vllm_logger.removeHandler(handler)

        # 添加 NullHandler 确保完全不输出
        # vllm_logger.addHandler(logging.NullHandler())
        
    def debug(self, msg, *args, **kwargs):
        """调试级别日志"""
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        """信息级别日志"""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        """警告级别日志"""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        """错误级别日志"""
        self.logger.error(msg, *args, **kwargs)
        
    def exception(self, msg, *args, **kwargs):
        """异常日志，自动记录堆栈信息"""
        self.logger.exception(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """严重错误级别日志"""
        self.logger.critical(msg, *args, **kwargs)
    
    def get_log_file(self):
        """获取当前日志文件路径"""
        return self.log_file
    
    # 兼容原有的日志调用方式
    def setup_logging(self):
        """初始化日志系统，保持与原有代码的兼容性"""
        return global_logger

# 提供全局单例实例，方便直接导入使用
global_logger = LogWrapper()

# 直接导出常用方法，方便使用
debug = global_logger.debug
info = global_logger.info
warning = global_logger.warning
error = global_logger.error
exception = global_logger.exception
critical = global_logger.critical
