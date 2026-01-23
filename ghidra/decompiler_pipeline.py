import time
from typing import Any, Dict, List

from exceptions import DecompilerError
from log_utils import global_logger as logger
from resource_monitor import MonitorLevel, create_simple_monitor


class DecompilerPipeline:
    """反编译器模块化流水线"""

    def __init__(self, config):
        self.config = config
        self.logger = logger
        self._modules: Dict[str, Any] = {}
        self._execution_order: List[str] = []  # 执行顺序

        self.results = {}
        self.performance_monitor = create_simple_monitor(
            sampling_interval=1.0,
            enable_gpu=True,
            monitor_level=MonitorLevel.EXTENDED
        )

    #todo: name 与 module 中的名字是否重复了？
    def register_module(self, name, module):
        """注册处理模块"""
        # name = module.name()
        if name in self._modules:  # O(1)检查
            raise ValueError(f"Module {name} already registered")

        self._modules[name] = module
        self._execution_order.append(name)
        self.logger.debug(f"注册模块: {name}")

    def execute_pipeline(self):
        """执行处理流水线"""
        self.logger.info("开始执行反编译流水线")

        for module_name in self._execution_order:
            module = self._modules[module_name]
            if module:
                try:
                    self.logger.info(f"🎯 开始执行模块: {module_name}")

                    # 记录模块开始时间
                    module_start_time = time.time()

                    # 性能监控
                    with self.performance_monitor.time_block(module_name):
                        result = module.process(self.results)

                    # 计算模块执行时间
                    module_duration = time.time() - module_start_time

                    self.results[module_name] = result
                    self.logger.info(f"✅ 模块 {module_name} 执行完成 (耗时: {module_duration:.3f}秒)")

                except Exception as e:
                    self.logger.error(f"❌ 模块 {module_name} 执行失败: {str(e)}")
                    raise DecompilerError(f"模块 {module_name} 执行失败") from e
            else:
                self.logger.warning(f"⚠️ 未找到模块: {module_name}，跳过")

        self.logger.info("🎉 反编译流水线执行完成")
        return self.results
