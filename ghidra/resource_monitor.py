import time
import psutil
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json
import threading
from enum import Enum, IntEnum
from collections import deque
import os
import sys
from log_utils import global_logger as logger

# 尝试导入可选依赖
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==================== 监控级别枚举 ====================
class MonitorLevel(IntEnum):
    """监控级别枚举"""
    NONE = 0      # 不监控
    BASIC = 1     # 基本监控（CPU、内存）
    EXTENDED = 2  # 扩展监控（增加GPU、磁盘）
    DETAILED = 3  # 详细监控（增加进程级监控）
    DEBUG = 4     # 调试监控（最高频率，最详细数据）

MonitorLevel.current_level = MonitorLevel.BASIC  # 默认监控级别

# ==================== 数据类 ====================
@dataclass
class ResourceMetrics:
    """资源指标数据类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_power_usage: Optional[float] = None
    disk_read_mb: Optional[float] = None
    disk_write_mb: Optional[float] = None
    threads_count: Optional[int] = None

# ==================== 指标收集器接口 ====================
class MetricsCollector:
    """指标收集器基类"""
    def collect(self) -> Dict[str, Any]:
        """收集指标数据"""
        raise NotImplementedError

# ==================== 基于 Torch 的 GPU 监控收集器 ====================
class TorchGpuMetricsCollector(MetricsCollector):
    """基于 PyTorch 的 GPU 指标收集器"""
    
    def __init__(self):
        self.has_torch = HAS_TORCH
        self.initialized = False
        self.logger = logger
        self.gpu_count = 0
        self.last_memory_allocated = {}
        self.last_collection_time = time.time()
        self.memory_change_history = deque(maxlen=10)
        self.utilization_smoothing_factor = 0.7
        self.collector_name = "Torch GPU Collector"
        
        if self.has_torch:
            self.initialize()
    
    def initialize(self) -> bool:
        """初始化 GPU 监控"""
        if not self.has_torch:
            self.logger.warning("PyTorch 未安装，无法进行 GPU 监控")
            return False
            
        try:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA 不可用，无法进行 GPU 监控")
                return False
            
            self.gpu_count = torch.cuda.device_count()
            if self.gpu_count == 0:
                self.logger.warning("未检测到 GPU 设备")
                return False
            
            # 初始化内存记录
            for i in range(self.gpu_count):
                self.last_memory_allocated[i] = torch.cuda.memory_allocated(i)
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.logger.info(f"Torch检测到 GPU {i}: {device_name} ({total_memory:.1f} GB)")
            
            self.initialized = True
            self.logger.info(f"Torch GPU 监控初始化完成，找到 {self.gpu_count} 个 GPU 设备")
            return True
            
        except Exception as e:
            self.logger.error(f"Torch GPU 监控初始化失败: {str(e)}")
            return False
    
    def _estimate_gpu_utilization(self, gpu_index: int, memory_allocated: int, time_diff: float) -> float:
        """估算 GPU 利用率"""
        try:
            # 计算内存变化率
            memory_diff = memory_allocated - self.last_memory_allocated.get(gpu_index, 0)
            memory_change_rate = abs(memory_diff) / (1024 * 1024)  # MB/s
            
            # 基于内存变化率估算利用率
            if time_diff > 0:
                base_utilization = min(100.0, memory_change_rate / time_diff * 2.0)
            else:
                base_utilization = 0.0
            
            # 考虑当前内存使用率
            if hasattr(torch.cuda, 'get_device_properties'):
                total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
                memory_ratio = memory_allocated / total_memory if total_memory > 0 else 0
                memory_based_utilization = memory_ratio * 50.0
            else:
                memory_based_utilization = 0.0
            
            # 综合计算利用率
            utilization = min(100.0, base_utilization + memory_based_utilization)
            
            # 使用平滑滤波减少波动
            self.memory_change_history.append(utilization)
            if len(self.memory_change_history) > 1:
                smoothed_utilization = sum(self.memory_change_history) / len(self.memory_change_history)
                utilization = (self.utilization_smoothing_factor * utilization + 
                             (1 - self.utilization_smoothing_factor) * smoothed_utilization)
            
            return utilization
            
        except Exception as e:
            self.logger.debug(f"GPU 利用率估算失败: {str(e)}")
            return 0.0
    
    def collect(self) -> Dict[str, Any]:
        """收集 GPU 指标"""
        if not self.initialized:
            return {}
        
        try:
            current_time = time.time()
            time_diff = current_time - self.last_collection_time
            result = {}
            
            # 目前只监控第一个 GPU
            gpu_index = 0
            
            # 获取内存使用情况
            memory_allocated = torch.cuda.memory_allocated(gpu_index)
            
            # 获取设备属性
            if hasattr(torch.cuda, 'get_device_properties'):
                device_props = torch.cuda.get_device_properties(gpu_index)
                total_memory = device_props.total_memory
            else:
                total_memory = None
            
            # 估算 GPU 利用率
            utilization = self._estimate_gpu_utilization(gpu_index, memory_allocated, time_diff)
            
            result.update({
                'gpu_utilization': utilization,
                'gpu_memory_used_mb': memory_allocated / (1024 * 1024),
                'gpu_memory_total_mb': total_memory / (1024 * 1024) if total_memory else None,
            })
            
            # 更新最后记录
            self.last_memory_allocated[gpu_index] = memory_allocated
            self.last_collection_time = current_time
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Torch GPU 指标采集失败: {str(e)}")
            # 如果连续失败，尝试重新初始化
            if "CUDA" in str(e) or "cuda" in str(e):
                self.logger.info("检测到 CUDA 错误，尝试重新初始化 Torch GPU 监控")
                self.initialized = False
                time.sleep(1.0)
                self.initialize()
            return {}
    
    def shutdown(self):
        """关闭 GPU 监控"""
        self.initialized = False
        self.memory_change_history.clear()

# ==================== 其他收集器保持不变 ====================
class CpuMetricsCollector(MetricsCollector):
    """CPU指标收集器"""
    def __init__(self, process: psutil.Process):
        self.process = process
        self.cpu_initialized = False
    
    def collect(self) -> Dict[str, Any]:
        try:
            if not self.cpu_initialized:
                cpu_percent = self.process.cpu_percent(interval=0.05)
                self.cpu_initialized = True
            else:
                cpu_percent = self.process.cpu_percent(interval=None)
            
            threads_count = len(self.process.threads())
            
            return {
                'cpu_percent': cpu_percent,
                'threads_count': threads_count
            }
        except Exception as e:
            logger.warning(f"CPU指标采集失败: {str(e)}")
            return {
                'cpu_percent': 0.0,
                'threads_count': 0
            }

class MemoryMetricsCollector(MetricsCollector):
    """内存指标收集器"""
    def __init__(self, process: psutil.Process):
        self.process = process
    
    def collect(self) -> Dict[str, Any]:
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            memory_used_mb = memory_info.rss / 1024 / 1024
            
            return {
                'memory_percent': memory_percent,
                'memory_used_mb': memory_used_mb
            }
        except Exception as e:
            logger.warning(f"内存指标采集失败: {str(e)}")
            return {
                'memory_percent': 0.0,
                'memory_used_mb': 0.0
            }

class DiskMetricsCollector(MetricsCollector):
    """磁盘指标收集器"""
    def __init__(self):
        self.last_io_counters = psutil.disk_io_counters()
        self.last_time = time.time()
    
    def collect(self) -> Dict[str, Any]:
        if MonitorLevel.current_level < MonitorLevel.EXTENDED:
            return {}
        
        try:
            current_io_counters = psutil.disk_io_counters()
            current_time = time.time()
            
            time_diff = current_time - self.last_time
            if time_diff <= 0:
                return {}
            
            read_mb = (current_io_counters.read_bytes - self.last_io_counters.read_bytes) / (1024 * 1024)
            write_mb = (current_io_counters.write_bytes - self.last_io_counters.write_bytes) / (1024 * 1024)
            
            disk_read_mb = read_mb / time_diff
            disk_write_mb = write_mb / time_diff
            
            self.last_io_counters = current_io_counters
            self.last_time = current_time
            
            return {
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb
            }
        except Exception as e:
            logger.warning(f"磁盘指标采集失败: {str(e)}")
            return {
                'disk_read_mb': 0.0,
                'disk_write_mb': 0.0
            }

# ==================== 主监控类 ====================
class ResourceMonitor:
    """
    实时资源监控模块
    监控CPU使用率、内存使用率、GPU使用率和显存使用情况
    采用单例模式确保全局唯一实例
    """
    # 单例模式实现
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceMonitor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 sampling_interval: float = 1.0,
                 max_samples: int = 1000,
                 enable_gpu_monitoring: bool = True,
                 log_level: str = "INFO",
                 monitor_level: MonitorLevel = MonitorLevel.BASIC,
                 stop_timeout: float = 5.0):
        """
        初始化资源监控器（单例模式）
        
        Args:
            sampling_interval: 采样间隔（秒），必须大于0
            max_samples: 最大采样点数，必须大于0
            enable_gpu_monitoring: 是否启用GPU监控
            log_level: 日志级别
            monitor_level: 监控级别
            stop_timeout: 停止监控时的超时时间（秒）
        """
        # 配置验证
        if sampling_interval <= 0:
            raise ValueError("采样间隔必须大于0")
        if max_samples <= 0:
            raise ValueError("最大采样数必须大于0")
        if stop_timeout <= 0:
            raise ValueError("停止超时必须大于0")
            
        # 防止重复初始化
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                # 更新参数但不重新初始化
                self.sampling_interval = sampling_interval
                self.max_samples = max_samples
                self.log_level = log_level
                self.stop_timeout = stop_timeout
                self.set_monitor_level(monitor_level)
                return
            
            # 基本配置
            self.sampling_interval = sampling_interval
            self.max_samples = max_samples
            self.enable_gpu_monitoring = enable_gpu_monitoring
            self.stop_timeout = stop_timeout
            
            # 监控状态
            self._is_monitoring = False
            self._monitor_thread = None
            self._stop_event = threading.Event()
            
            # 实时监控状态
            self._realtime_enabled = False
            self._realtime_thread = None
            self._realtime_stop_event = threading.Event()
            self._realtime_interval = 2.0
            self._display_callback = None
            
            # 数据存储
            self.metrics_history = deque(maxlen=max_samples)
            self._history_lock = threading.RLock()
            
            # 回调函数
            self._alert_callbacks = []
            
            # 告警阈值
            self.alert_thresholds = {
                'cpu_percent': 90.0,
                'memory_percent': 85.0,
                'gpu_utilization': 95.0,
                'gpu_memory_percent': 90.0
            }
            
            # 进程监控
            self.process = psutil.Process()
            
            # 日志设置
            self.logger = logger
            self.log_level = log_level
            
            # 模块执行时间记录
            self._module_times = {}
            
            # 设置监控级别
            MonitorLevel.current_level = monitor_level
            
            # 初始化指标收集器
            self.collectors = []
            self.gpu_collector = None
            self._initialize_collectors()
            
            # 标记为已初始化
            self._initialized = True
            
            # self.logger.info("资源监控器初始化完成")
    
    def _initialize_collectors(self):
        """初始化指标收集器"""
        self.collectors = []
        
        # 添加CPU和内存收集器（基础监控）
        self.collectors.append(CpuMetricsCollector(self.process))
        self.collectors.append(MemoryMetricsCollector(self.process))
        
        # 添加GPU收集器（如果启用）
        if self.enable_gpu_monitoring:
            self.gpu_collector = TorchGpuMetricsCollector()
            if self.gpu_collector.initialized:
                self.collectors.append(self.gpu_collector)
                self.logger.info("Torch GPU 监控已启用")
            else:
                self.logger.warning("Torch GPU 监控初始化失败，将继续使用CPU和内存监控")
        
        # 添加磁盘收集器（扩展监控）
        if MonitorLevel.current_level >= MonitorLevel.EXTENDED:
            self.collectors.append(DiskMetricsCollector())
    
    def set_monitor_level(self, level: MonitorLevel):
        """设置监控级别"""
        with self._lock:
            MonitorLevel.current_level = level
            
            if level == MonitorLevel.NONE:
                if self._is_monitoring:
                    self.stop_monitoring()
            elif level == MonitorLevel.BASIC:
                self.sampling_interval = max(self.sampling_interval, 5.0)
                # 移除磁盘收集器
                self.collectors = [c for c in self.collectors if not isinstance(c, DiskMetricsCollector)]
            elif level == MonitorLevel.EXTENDED:
                self.sampling_interval = max(self.sampling_interval, 2.0)
                # 确保有磁盘收集器
                if not any(isinstance(c, DiskMetricsCollector) for c in self.collectors):
                    self.collectors.append(DiskMetricsCollector())
            elif level == MonitorLevel.DETAILED:
                self.sampling_interval = max(self.sampling_interval, 1.0)
                # 确保有磁盘收集器
                if not any(isinstance(c, DiskMetricsCollector) for c in self.collectors):
                    self.collectors.append(DiskMetricsCollector())
            elif level == MonitorLevel.DEBUG:
                self.sampling_interval = 0.1
                # 确保有磁盘收集器
                if not any(isinstance(c, DiskMetricsCollector) for c in self.collectors):
                    self.collectors.append(DiskMetricsCollector())
        
        self.logger.info(f"监控级别已设置为: {level.name}")
    
    def start_monitoring(self):
        """开始资源监控"""
        if MonitorLevel.current_level == MonitorLevel.NONE:
            self.logger.warning("监控级别为NONE，无法启动监控")
            return
            
        with self._lock:
            if self._is_monitoring:
                self.logger.warning("资源监控已在运行中")
                return
                
            self._stop_event.clear()
            self._is_monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            
            # self.logger.info(f"资源监控已启动，采样间隔: {self.sampling_interval}秒")
    
    def stop_monitoring(self):
        """停止资源监控"""
        with self._lock:
            if not self._is_monitoring:
                return
                
            self._is_monitoring = False
            self._stop_event.set()
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=self.stop_timeout)
                if self._monitor_thread.is_alive():
                    self.logger.warning(f"监控线程在 {self.stop_timeout} 秒后仍未停止")
            
            # 清理GPU资源
            if hasattr(self, 'gpu_collector') and self.gpu_collector:
                self.gpu_collector.shutdown()
            
            # 停止实时监控
            self.disable_realtime_monitoring()
            
            self.logger.info("资源监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                if metrics:
                    self._store_metrics(metrics)
                    # 资源检查告警
                    # self._check_alerts(metrics)
                    
                # 根据当前监控级别调整采样间隔
                current_interval = self.sampling_interval
                if MonitorLevel.current_level == MonitorLevel.DEBUG:
                    current_interval = 0.1
                elif MonitorLevel.current_level == MonitorLevel.DETAILED:
                    current_interval = max(1.0, current_interval)
                elif MonitorLevel.current_level == MonitorLevel.EXTENDED:
                    current_interval = max(2.0, current_interval)
                elif MonitorLevel.current_level == MonitorLevel.BASIC:
                    current_interval = max(5.0, current_interval)
                
                self._stop_event.wait(current_interval)
                
            except Exception as e:
                self.logger.error(f"监控数据采集失败: {str(e)}")
                time.sleep(max(self.sampling_interval, 2.0))
    
    def _collect_metrics(self) -> Optional[ResourceMetrics]:
        """收集资源指标"""
        timestamp = time.time()
        
        base_metrics = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used_mb': 0.0,
            'threads_count': 0
        }
        
        extended_metrics = {
            'gpu_utilization': None,
            'gpu_memory_used_mb': None,
            'gpu_memory_total_mb': None,
            'gpu_temperature': None,
            'gpu_power_usage': None,
            'disk_read_mb': None,
            'disk_write_mb': None
        }
        
        collectors_copy = self.collectors.copy()
        
        for collector in collectors_copy:
            try:
                collected = collector.collect()
                for key in ['cpu_percent', 'memory_percent', 'memory_used_mb', 'threads_count']:
                    if key in collected:
                        base_metrics[key] = collected[key]
                for key in extended_metrics:
                    if key in collected:
                        extended_metrics[key] = collected[key]
            except Exception as e:
                self.logger.warning(f"收集器 {collector.__class__.__name__} 采集失败: {str(e)}")
        
        metrics = ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=base_metrics['cpu_percent'],
            memory_percent=base_metrics['memory_percent'],
            memory_used_mb=base_metrics['memory_used_mb'],
            threads_count=base_metrics['threads_count'],
            gpu_utilization=extended_metrics['gpu_utilization'],
            gpu_memory_used_mb=extended_metrics['gpu_memory_used_mb'],
            gpu_memory_total_mb=extended_metrics['gpu_memory_total_mb'],
            gpu_temperature=extended_metrics['gpu_temperature'],
            gpu_power_usage=extended_metrics['gpu_power_usage'],
            disk_read_mb=extended_metrics['disk_read_mb'],
            disk_write_mb=extended_metrics['disk_write_mb']
        )
        
        return metrics
    
    def _store_metrics(self, metrics: ResourceMetrics):
        """存储指标数据"""
        with self._history_lock:
            self.metrics_history.append(metrics)
    
    def _check_alerts(self, metrics: ResourceMetrics):
        """检查告警条件"""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"内存使用率过高: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_utilization and metrics.gpu_utilization > self.alert_thresholds['gpu_utilization']:
            alerts.append(f"GPU使用率过高: {metrics.gpu_utilization:.1f}%")
        
        if (metrics.gpu_memory_used_mb and metrics.gpu_memory_total_mb and 
            (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb * 100) > self.alert_thresholds['gpu_memory_percent']):
            alerts.append(f"GPU显存使用率过高: {metrics.gpu_memory_used_mb:.0f}/{metrics.gpu_memory_total_mb:.0f} MB")
        
        if alerts and self._alert_callbacks:
            alert_message = " | ".join(alerts)
            for callback in self._alert_callbacks:
                try:
                    callback(alert_message, metrics)
                except Exception as e:
                    self.logger.error(f"告警回调执行失败: {str(e)}")
    
    def add_alert_callback(self, callback: Callable[[str, ResourceMetrics], None]):
        """添加告警回调函数"""
        self._alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """设置告警阈值"""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            # self.logger.info(f"设置 {metric} 告警阈值为: {threshold}")
        else:
            self.logger.warning(f"未知的监控指标: {metric}")
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """获取当前资源指标"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_metrics_snapshot(self, last_n: int = None) -> List[ResourceMetrics]:
        """获取指标快照"""
        with self._history_lock:
            if last_n and last_n < len(self.metrics_history):
                return list(self.metrics_history)[-last_n:]
            return list(self.metrics_history)
    
    def get_statistics(self, last_n: int = None) -> Dict:
        """获取统计信息"""
        metrics = self.get_metrics_snapshot(last_n)
        
        if not metrics:
            return {}
        
        stats = {
            'sample_count': len(metrics),
            'time_range': {
                'start': datetime.fromtimestamp(metrics[0].timestamp).isoformat(),
                'end': datetime.fromtimestamp(metrics[-1].timestamp).isoformat(),
                'duration_seconds': metrics[-1].timestamp - metrics[0].timestamp
            }
        }
        
        # CPU统计
        cpu_values = [m.cpu_percent for m in metrics]
        stats['cpu'] = {
            'avg': sum(cpu_values) / len(cpu_values),
            'max': max(cpu_values),
            'min': min(cpu_values)
        }
        
        # 内存统计
        memory_values = [m.memory_percent for m in metrics]
        stats['memory'] = {
            'avg': sum(memory_values) / len(memory_values),
            'max': max(memory_values),
            'min': min(memory_values)
        }
        
        # GPU统计（如果可用）
        gpu_util_values = [m.gpu_utilization for m in metrics if m.gpu_utilization is not None]
        if gpu_util_values:
            stats['gpu_utilization'] = {
                'avg': sum(gpu_util_values) / len(gpu_util_values),
                'max': max(gpu_util_values),
                'min': min(gpu_util_values)
            }
        
        gpu_memory_values = [m.gpu_memory_used_mb for m in metrics if m.gpu_memory_used_mb is not None]
        if gpu_memory_values:
            stats['gpu_memory'] = {
                'avg': sum(gpu_memory_values) / len(gpu_memory_values),
                'max': max(gpu_memory_values),
                'min': min(gpu_memory_values)
            }
        
        # 磁盘统计（如果可用）
        if MonitorLevel.current_level >= MonitorLevel.EXTENDED:
            disk_read_values = [m.disk_read_mb for m in metrics if m.disk_read_mb is not None]
            if disk_read_values:
                stats['disk_read'] = {
                    'avg': sum(disk_read_values) / len(disk_read_values),
                    'max': max(disk_read_values),
                    'min': min(disk_read_values)
                }
            
            disk_write_values = [m.disk_write_mb for m in metrics if m.disk_write_mb is not None]
            if disk_write_values:
                stats['disk_write'] = {
                    'avg': sum(disk_write_values) / len(disk_write_values),
                    'max': max(disk_write_values),
                    'min': min(disk_write_values)
                }
        
        return stats
    
    # ==================== 实时监控功能 ====================
    
    def enable_realtime_monitoring(self, update_interval: float = 2.0, 
                                 display_callback: Optional[Callable[[str], None]] = None):
        """启用实时监控显示
        
        Args:
            update_interval: 更新间隔（秒）
            display_callback: 自定义显示回调函数
        """
        if self._realtime_enabled:
            self.logger.warning("实时监控已在运行中")
            return
            
        self._realtime_enabled = True
        self._realtime_interval = update_interval
        self._display_callback = display_callback
        
        # 初始化监控显示区域
        self._init_monitor_display()
        
        # 启动实时显示线程
        self._realtime_stop_event.clear()
        self._realtime_thread = threading.Thread(
            target=self._realtime_monitoring_loop, 
            daemon=True
        )
        self._realtime_thread.start()
        # self.logger.info(f"实时监控已启用，更新间隔: {update_interval}秒")
    
    def disable_realtime_monitoring(self):
        """禁用实时监控显示"""
        if not self._realtime_enabled:
            return
            
        self._realtime_enabled = False
        self._realtime_stop_event.set()
        
        if self._realtime_thread and self._realtime_thread.is_alive():
            self._realtime_thread.join(timeout=2.0)
        
        # 清理监控显示区域
        # self._clear_monitor_display()
        
        self.logger.info("实时监控已禁用")
    
    def _realtime_monitoring_loop(self):
        """实时监控循环"""
        last_display_time = 0
        
        while not self._realtime_stop_event.is_set() and self._realtime_enabled:
            try:
                current_time = time.time()
                if current_time - last_display_time >= self._realtime_interval:
                    self._display_realtime_status()
                    last_display_time = current_time
                
                # 等待下一次更新
                self._realtime_stop_event.wait(0.1)
                
            except Exception as e:
                self.logger.error(f"实时监控显示失败: {str(e)}")
                time.sleep(1.0)
    
    def _init_monitor_display(self):
        """初始化监控显示区域"""
        # 保存当前光标位置
        sys.stdout.write('\033[s')
        
        # 移动光标到屏幕底部
        sys.stdout.write('\033[999B')
        
        # 输出分隔线和初始监控信息
        # sys.stdout.write('\n' + '─' * 80 + '\n')
        sys.stdout.write("实时监控: 初始化中...\n")
        
        # 恢复光标位置
        sys.stdout.write('\033[u')
        sys.stdout.flush()
    
    def _clear_monitor_display(self):
        """清理监控显示区域"""
        # 保存当前光标位置
        sys.stdout.write('\033[s')
        
        # 移动光标到监控区域
        sys.stdout.write('\033[999B\033[2A')
        
        # 清除监控区域
        sys.stdout.write('\033[K\n\033[K\n\033[K')
        
        # 恢复光标位置
        sys.stdout.write('\033[u')
        sys.stdout.flush()
    
    def _display_realtime_status(self):
        """显示实时状态 - 固定在屏幕底部最后一行"""
        metrics = self.get_current_metrics()
        if not metrics:
            return
        
        # 构建简洁的单行状态信息
        timestamp_str = datetime.fromtimestamp(metrics.timestamp).strftime('%H:%M:%S')
        
        # 基础信息
        status_parts = [f"[{timestamp_str}]"]
        status_parts.append(f"CPU:{metrics.cpu_percent:5.1f}%")
        status_parts.append(f"内存:{metrics.memory_percent:5.1f}%")
        
        # 线程数
        if metrics.threads_count is not None:
            status_parts.append(f"线程:{metrics.threads_count:2d}")
        
        # GPU信息
        if metrics.gpu_utilization is not None:
            status_parts.append(f"GPU:{metrics.gpu_utilization:5.1f}%")
        
        if metrics.gpu_memory_used_mb is not None and metrics.gpu_memory_total_mb is not None:
            gpu_memory_percent = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            status_parts.append(f"显存:{gpu_memory_percent:5.1f}%")
        
        # 构建完整状态字符串
        status_text = " | ".join(status_parts)
        
        # 使用回调或默认打印
        if self._display_callback:
            self._display_callback(status_text)
        else:
            # 使用可靠的方法固定在屏幕底部最后一行
            # 保存当前光标位置
            sys.stdout.write('\033[s')
            
            # 移动到屏幕最底部
            sys.stdout.write('\033[999B')
            
            # 移动到行首并清除整行
            sys.stdout.write('\r\033[K')
            
            # 输出监控信息
            sys.stdout.write(status_text)
            
            # 恢复光标位置
            sys.stdout.write('\033[u')
            sys.stdout.flush()
    
    def get_realtime_status_text(self) -> str:
        """获取实时状态文本"""
        metrics = self.get_current_metrics()
        if not metrics:
            return "暂无监控数据"
        
        # 简化的单行格式
        status_parts = []
        status_parts.append(f"CPU:{metrics.cpu_percent:.1f}%")
        status_parts.append(f"内存:{metrics.memory_percent:.1f}%")
        
        if metrics.threads_count is not None:
            status_parts.append(f"线程:{metrics.threads_count}")
        
        if metrics.gpu_utilization is not None:
            status_parts.append(f"GPU:{metrics.gpu_utilization:.1f}%")
        
        if metrics.gpu_memory_used_mb is not None and metrics.gpu_memory_total_mb is not None:
            gpu_memory_percent = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            status_parts.append(f"显存:{gpu_memory_percent:.1f}%")
        
        return " | ".join(status_parts)
    
    def enable_lightweight_realtime_monitoring(self, update_interval: float = 5.0):
        """启用轻量级实时监控（固定在底部）"""
        def lightweight_display(status_text):
            # 使用可靠的方法固定在屏幕底部最后一行
            sys.stdout.write('\033[s')  # 保存光标位置
            
            # 移动到屏幕最底部
            sys.stdout.write('\033[999B')
            
            # 移动到行首并清除整行
            sys.stdout.write('\r\033[K')
            
            # 输出监控信息
            sys.stdout.write(status_text)
            
            # 恢复光标位置
            sys.stdout.write('\033[u')
            sys.stdout.flush()
        
        self.enable_realtime_monitoring(
            update_interval=update_interval,
            display_callback=lightweight_display
        )
    
    def print_current_status(self):
        """打印当前状态"""
        metrics = self.get_current_metrics()
        if not metrics:
            print("暂无监控数据")
            return
        
        print(f"\n=== 资源监控状态 ===")
        print(f"时间: {datetime.fromtimestamp(metrics.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"CPU使用率: {metrics.cpu_percent:.1f}%")
        print(f"内存使用: {metrics.memory_used_mb:.1f} MB ({metrics.memory_percent:.1f}%)")
        
        if metrics.threads_count is not None:
            print(f"线程数: {metrics.threads_count}")
        
        if metrics.gpu_utilization is not None:
            print(f"GPU使用率: {metrics.gpu_utilization:.1f}%")
        
        if metrics.gpu_memory_used_mb is not None and metrics.gpu_memory_total_mb is not None:
            memory_percent = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            print(f"GPU显存: {metrics.gpu_memory_used_mb:.1f}/{metrics.gpu_memory_total_mb:.1f} MB ({memory_percent:.1f}%)")
        
        if metrics.gpu_temperature is not None:
            print(f"GPU温度: {metrics.gpu_temperature}°C")
        
        if metrics.gpu_power_usage is not None:
            print(f"GPU功耗: {metrics.gpu_power_usage:.1f} W")
        
        if MonitorLevel.current_level >= MonitorLevel.EXTENDED:
            if metrics.disk_read_mb is not None:
                print(f"磁盘读取: {metrics.disk_read_mb:.1f} MB/s")
            
            if metrics.disk_write_mb is not None:
                print(f"磁盘写入: {metrics.disk_write_mb:.1f} MB/s")
    
    def export_to_json(self, filepath: str, last_n: int = None):
        """导出监控数据到JSON文件"""
        # 路径安全检查
        try:
            filepath = os.path.abspath(filepath)
            # 确保导出目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except Exception as e:
            self.logger.error(f"导出路径无效: {str(e)}")
            return
        
        metrics = self.get_metrics_snapshot(last_n)
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'sampling_interval': self.sampling_interval,
            'metrics_count': len(metrics),
            'monitor_level': MonitorLevel.current_level.name,
            'metrics': [
                {
                    'timestamp': m.timestamp,
                    'datetime': datetime.fromtimestamp(m.timestamp).isoformat(),
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_used_mb': m.memory_used_mb,
                    'threads_count': m.threads_count,
                    'gpu_utilization': m.gpu_utilization,
                    'gpu_memory_used_mb': m.gpu_memory_used_mb,
                    'gpu_memory_total_mb': m.gpu_memory_total_mb,
                    'gpu_temperature': m.gpu_temperature,
                    'gpu_power_usage': m.gpu_power_usage,
                    'disk_read_mb': m.disk_read_mb,
                    'disk_write_mb': m.disk_write_mb
                }
                for m in metrics
            ]
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"监控数据已导出到: {filepath}")
        except Exception as e:
            self.logger.error(f"数据导出失败: {str(e)}")
    
    def get_module_times(self):
        """获取所有模块的执行时间统计"""
        if not hasattr(self, '_module_times'):
            return {}
        
        stats = {}
        for module_name, times in self._module_times.items():
            durations = [t['elapsed_time'] for t in times]
            stats[module_name] = {
                'calls': len(times),
                'avg_time': sum(durations) / len(durations),
                'max_time': max(durations),
                'min_time': min(durations),
                'total_time': sum(durations)
            }
        
        return stats
    
    def time_block(self, module_name):
        """返回用于计时的上下文管理器"""
        return self.ModuleTimer(self, module_name)
    
    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health_status = {
            'monitoring_active': self._is_monitoring,
            'gpu_available': self.gpu_collector.initialized if hasattr(self, 'gpu_collector') and self.gpu_collector else False,
            'history_size': len(self.metrics_history),
            'collectors_count': len(self.collectors),
            'realtime_enabled': self._realtime_enabled,
            'thread_alive': self._monitor_thread.is_alive() if self._monitor_thread else False
        }
        return health_status
    
    # 上下文管理器支持
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
    
    # 装饰器支持
    @classmethod
    def as_decorator(cls, sampling_interval=2.0, enable_gpu=True, monitor_level=MonitorLevel.BASIC):
        """作为装饰器使用"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                monitor = cls(sampling_interval=sampling_interval, 
                             enable_gpu=enable_gpu, 
                             monitor_level=monitor_level)
                with monitor:
                    return func(*args, **kwargs)
            return wrapper
        return decorator

# ======================== 模块计时 ==========================
class ModuleTimer:
    """模块计时上下文管理器"""
    
    def __init__(self, monitor, module_name):
        self.monitor = monitor
        self.module_name = module_name
        self.start_time = 0
        self.end_time = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        if not hasattr(self.monitor, '_module_times'):
            self.monitor._module_times = {}
        
        if self.module_name not in self.monitor._module_times:
            self.monitor._module_times[self.module_name] = []
        
        self.monitor._module_times[self.module_name].append({
            'start_time': self.start_time,
            'end_time': self.end_time,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # self.monitor.logger.info(f"模块 '{self.module_name}' 执行时间: {elapsed_time:.3f}秒")

ResourceMonitor.ModuleTimer = ModuleTimer

# ==================== 工具函数 ====================
def create_simple_monitor(sampling_interval: float = 2.0, 
                         enable_gpu: bool = True, 
                         monitor_level: MonitorLevel = MonitorLevel.BASIC) -> ResourceMonitor:
    """
    创建简单的资源监控器
    """
    monitor = ResourceMonitor(
        sampling_interval=sampling_interval,
        enable_gpu_monitoring=enable_gpu,
        monitor_level=monitor_level,
        log_level="INFO"
    )
    
    return monitor