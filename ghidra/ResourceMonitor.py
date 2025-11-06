import time
import psutil
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json
import threading
from enum import Enum, IntEnum
from log_utils import global_logger as logger

# 尝试导入可选依赖

# GPU监控依赖
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# PyTorch支持依赖
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
    # 为扩展监控级别添加更多字段
    disk_read_mb: Optional[float] = None
    disk_write_mb: Optional[float] = None
    threads_count: Optional[int] = None

# ==================== 指标收集器接口 ====================
class MetricsCollector:
    """指标收集器基类"""
    def collect(self) -> Dict[str, Any]:
        """收集指标数据"""
        raise NotImplementedError

# ==================== 具体指标收集器实现 ====================
class CpuMetricsCollector(MetricsCollector):
    """CPU指标收集器"""
    def __init__(self, process: psutil.Process):
        self.process = process
        self.cpu_initialized = False
    
    def collect(self) -> Dict[str, Any]:
        try:
            # 对于首次采集，设置短时间间隔以获得准确值
            if not self.cpu_initialized:
                cpu_percent = self.process.cpu_percent(interval=0.05)  # 短暂阻塞以获取准确值
                self.cpu_initialized = True
            else:
                cpu_percent = self.process.cpu_percent(interval=None)  # 非阻塞调用
            
            # 获取线程数用于详细监控
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
            memory_used_mb = memory_info.rss / 1024 / 1024  # 转换为MB
            
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

class GpuMetricsCollector(MetricsCollector):
    """GPU指标收集器"""
    def __init__(self, max_retries: int = 3):
        self.has_nvml = HAS_NVML
        self.device_handles = []
        self.initialized = False
        self.max_retries = max_retries
        self.logger = logger
        
        # 尝试初始化
        if self.has_nvml:
            self.initialize()
    
    def initialize(self) -> bool:
        """初始化GPU监控"""
        if not self.has_nvml:
            self.logger.warning("pynvml未安装，无法进行GPU监控")
            return False
            
        retries = 0
        while retries < self.max_retries:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                if device_count == 0:
                    self.logger.warning("未检测到GPU设备")
                    return False
                    
                self.device_handles = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.device_handles.append(handle)
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    self.logger.info(f"检测到GPU {i}: {gpu_name.decode('utf-8') if isinstance(gpu_name, bytes) else gpu_name}")
                
                self.initialized = True
                self.logger.info(f"GPU监控初始化完成，找到 {device_count} 个GPU设备")
                return True
                
            except Exception as e:
                retries += 1
                self.logger.error(f"GPU监控初始化尝试 {retries}/{self.max_retries} 失败: {str(e)}")
                if retries < self.max_retries:
                    time.sleep(0.5)  # 等待一段时间后重试
        
        self.logger.error(f"GPU监控初始化失败（已尝试 {self.max_retries} 次）")
        return False
    
    def is_handle_valid(self, handle) -> bool:
        """检查GPU句柄是否有效"""
        if not self.has_nvml or not self.initialized:
            return False
        
        try:
            # 尝试获取简单的设备信息来验证句柄
            pynvml.nvmlDeviceGetIndex(handle)
            return True
        except:
            return False
    
    def collect(self) -> Dict[str, Any]:
        if not self.has_nvml or not self.initialized or not self.device_handles:
            return {}
        
        try:
            # 检查第一个GPU句柄是否有效，如果无效则尝试重新初始化
            if not self.is_handle_valid(self.device_handles[0]):
                self.logger.warning("GPU句柄失效，尝试重新初始化")
                if not self.initialize():
                    return {}
            
            # 目前只监控第一个GPU
            handle = self.device_handles[0]
            
            # 带重试的GPU利用率获取
            utilization = None
            retries = 0
            while retries < self.max_retries and utilization is None:
                # 改进的错误处理
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                except pynvml.NVMLError as e:
                    retries += 1
                    error_type = type(e).__name__
                    self.logger.warning(f"GPU利用率获取尝试 {retries}/{self.max_retries} 失败 [{error_type}]: {str(e)}")
                    if retries < self.max_retries:
                        # 根据错误类型调整重试间隔
                        retry_interval = 2.0 if error_type in ['NVMLError_Busy', 'NVMLError_Timeout'] else 1.0
                        time.sleep(retry_interval)
                except Exception as e:
                    retries += 1
                    self.logger.warning(f"GPU利用率获取尝试 {retries}/{self.max_retries} 失败 [UnknownError]: {str(e)}")
                    if retries < self.max_retries:
                        time.sleep(0.5)
            
            # 带重试的显存信息获取
            memory_info = None
            retries = 0
            while retries < self.max_retries and memory_info is None:
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                except Exception as e:
                    retries += 1
                    self.logger.warning(f"显存信息获取尝试 {retries}/{self.max_retries} 失败: {str(e)}")
                    if retries < self.max_retries:
                        time.sleep(0.1)  # 短暂等待后重试
            
            # 构建返回结果
            result = {}
            if utilization:
                result['gpu_utilization'] = utilization.gpu
            
            if memory_info:
                result['gpu_memory_used_mb'] = memory_info.used / 1024 / 1024
                result['gpu_memory_total_mb'] = memory_info.total / 1024 / 1024
            
            # 扩展监控级别下采集更多GPU指标
            if MonitorLevel.current_level >= MonitorLevel.EXTENDED:
                # GPU温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    result['gpu_temperature'] = temperature
                except Exception as te:
                    self.logger.debug(f"GPU温度采集失败: {str(te)}")
                
                # GPU功耗
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                    result['gpu_power_usage'] = power_usage
                except Exception as pe:
                    self.logger.debug(f"GPU功耗采集失败: {str(pe)}")
            
            return result
        except Exception as e:
            self.logger.error(f"GPU指标采集失败: {str(e)}")
            return {}
    
    def shutdown(self):
        """关闭GPU监控"""
        if self.has_nvml and self.initialized:
            try:
                pynvml.nvmlShutdown()
                self.initialized = False
            except Exception as e:
                self.logger.warning(f"GPU监控关闭时发生错误: {str(e)}")

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
            
            # 计算时间差
            time_diff = current_time - self.last_time
            if time_diff <= 0:
                return {}
            
            # 计算读写速率（MB/s）
            read_mb = (current_io_counters.read_bytes - self.last_io_counters.read_bytes) / (1024 * 1024)
            write_mb = (current_io_counters.write_bytes - self.last_io_counters.write_bytes) / (1024 * 1024)
            
            disk_read_mb = read_mb / time_diff
            disk_write_mb = write_mb / time_diff
            
            # 更新上次值
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
                 monitor_level: MonitorLevel = MonitorLevel.BASIC):
        """
        初始化资源监控器（单例模式）
        
        Args:
            sampling_interval: 采样间隔（秒）
            max_samples: 最大采样点数
            enable_gpu_monitoring: 是否启用GPU监控
            log_level: 日志级别
            monitor_level: 监控级别
        """
        # 防止重复初始化
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                # 更新参数但不重新初始化
                self.sampling_interval = sampling_interval
                self.max_samples = max_samples
                self.log_level = log_level
                self.set_monitor_level(monitor_level)
                
                # 如果GPU监控状态改变，重新初始化GPU
                if enable_gpu_monitoring != self.enable_gpu_monitoring:
                    self.enable_gpu_monitoring = enable_gpu_monitoring
                    if hasattr(self, 'gpu_collector'):
                        self.gpu_collector.shutdown()
                        self._initialize_collectors()
                
                return
            
            # 设置监控级别
            MonitorLevel.current_level = monitor_level
            
            # 基本配置
            self.sampling_interval = sampling_interval
            self.max_samples = max_samples
            self.enable_gpu_monitoring = enable_gpu_monitoring
            
            # 监控状态
            self._is_monitoring = False
            self._monitor_thread = None
            self._stop_event = threading.Event()
            
            # 数据存储
            self.metrics_history: List[ResourceMetrics] = []
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
            
            # 初始化指标收集器
            self.collectors = []
            self.gpu_collector = None
            self._initialize_collectors()
            
            # 标记为已初始化
            self._initialized = True
            
            self.logger.info("资源监控器初始化完成")
    
    def _initialize_collectors(self):
        """初始化指标收集器"""
        self.collectors = []
        
        # 添加CPU和内存收集器（基础监控）
        self.collectors.append(CpuMetricsCollector(self.process))
        self.collectors.append(MemoryMetricsCollector(self.process))
        
        # 添加GPU收集器（如果启用）
        if self.enable_gpu_monitoring:
            self.gpu_collector = GpuMetricsCollector()
            if self.gpu_collector.initialized:
                self.collectors.append(self.gpu_collector)
        
        # 添加磁盘收集器（扩展监控）
        if MonitorLevel.current_level >= MonitorLevel.EXTENDED:
            self.collectors.append(DiskMetricsCollector())
    
    def set_monitor_level(self, level: MonitorLevel):
        """设置监控级别"""
        with self._lock:
            MonitorLevel.current_level = level
            
            # 根据级别调整采样间隔
            if level == MonitorLevel.NONE:
                if self._is_monitoring:
                    self.stop_monitoring()
            elif level == MonitorLevel.BASIC:
                self.sampling_interval = max(self.sampling_interval, 5.0)  # 至少5秒
                # 移除磁盘收集器
                self.collectors = [c for c in self.collectors if not isinstance(c, DiskMetricsCollector)]
            elif level == MonitorLevel.EXTENDED:
                self.sampling_interval = max(self.sampling_interval, 2.0)  # 至少2秒
                # 确保有磁盘收集器
                if not any(isinstance(c, DiskMetricsCollector) for c in self.collectors):
                    self.collectors.append(DiskMetricsCollector())
            elif level == MonitorLevel.DETAILED:
                self.sampling_interval = max(self.sampling_interval, 1.0)  # 至少1秒
                # 确保有磁盘收集器
                if not any(isinstance(c, DiskMetricsCollector) for c in self.collectors):
                    self.collectors.append(DiskMetricsCollector())
            elif level == MonitorLevel.DEBUG:
                self.sampling_interval = 0.1  # 调试模式下使用最高频率
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
            
            self.logger.info(f"资源监控已启动，采样间隔: {self.sampling_interval}秒")
    
    def stop_monitoring(self):
        """停止资源监控"""
        with self._lock:
            if not self._is_monitoring:
                return
                
            self._is_monitoring = False
            self._stop_event.set()
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)  # 等待最多5秒
            
            # 清理GPU资源
            if hasattr(self, 'gpu_collector') and self.gpu_collector:
                self.gpu_collector.shutdown()
            
            self.logger.info("资源监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                if metrics:
                    self._store_metrics(metrics)
                    self._check_alerts(metrics)
                    
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
                
                # 等待下一个采样周期
                self._stop_event.wait(current_interval)
                
            except Exception as e:
                self.logger.error(f"监控数据采集失败: {str(e)}")
                # 发生异常时，使用更长的间隔避免频繁出错
                time.sleep(max(self.sampling_interval, 2.0))
    
    def _collect_metrics(self) -> Optional[ResourceMetrics]:
        """收集资源指标"""
        timestamp = time.time()
        
        # 基础指标收集
        base_metrics = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used_mb': 0.0,
            'threads_count': 0
        }
        
        # GPU和其他扩展指标
        extended_metrics = {
            'gpu_utilization': None,
            'gpu_memory_used_mb': None,
            'gpu_memory_total_mb': None,
            'gpu_temperature': None,
            'gpu_power_usage': None,
            'disk_read_mb': None,
            'disk_write_mb': None
        }
        
        # 从所有收集器收集数据
        with self._lock:
            for collector in self.collectors:
                try:
                    collected = collector.collect()
                    # 更新基础指标
                    for key in ['cpu_percent', 'memory_percent', 'memory_used_mb', 'threads_count']:
                        if key in collected:
                            base_metrics[key] = collected[key]
                    # 更新扩展指标
                    for key in extended_metrics:
                        if key in collected:
                            extended_metrics[key] = collected[key]
                except Exception as e:
                    self.logger.warning(f"收集器 {collector.__class__.__name__} 采集失败: {str(e)}")
        
        # 创建指标对象
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
            
            # 限制历史数据大小
            if len(self.metrics_history) > self.max_samples:
                self.metrics_history.pop(0)
    
    def _check_alerts(self, metrics: ResourceMetrics):
        """检查告警条件"""
        alerts = []
        
        # CPU告警
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
        
        # 内存告警
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"内存使用率过高: {metrics.memory_percent:.1f}%")
        
        # GPU告警
        if metrics.gpu_utilization and metrics.gpu_utilization > self.alert_thresholds['gpu_utilization']:
            alerts.append(f"GPU使用率过高: {metrics.gpu_utilization:.1f}%")
        
        if (metrics.gpu_memory_used_mb and metrics.gpu_memory_total_mb and 
            (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb * 100) > self.alert_thresholds['gpu_memory_percent']):
            alerts.append(f"GPU显存使用率过高: {metrics.gpu_memory_used_mb:.0f}/{metrics.gpu_memory_total_mb:.0f} MB")
        
        # 触发告警回调
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
            self.logger.info(f"设置 {metric} 告警阈值为: {threshold}")
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
                return self.metrics_history[-last_n:]
            return self.metrics_history.copy()
    
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
        """获取所有模块的执行时间统计
        
        Returns:
            Dict: 包含每个模块执行时间信息的字典
        """
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
        """返回用于计时的上下文管理器
        
        Args:
            module_name: 模块名称，用于标识被计时的代码块
            
        Returns:
            ModuleTimer: 上下文管理器对象
        """
        return self.ModuleTimer(self, module_name)
    
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
        """作为装饰器使用
        
        Args:
            sampling_interval: 采样间隔
            enable_gpu: 是否启用GPU监控
            monitor_level: 监控级别
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 获取单例实例
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
        # 记录模块执行时间
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
        
        self.monitor.logger.info(f"模块 '{self.module_name}' 执行时间: {elapsed_time:.3f}秒")

# 将ModuleTimer设为ResourceMonitor的内部类
ResourceMonitor.ModuleTimer = ModuleTimer

# ==================== 工具函数 ====================

def create_simple_monitor(sampling_interval: float = 2.0, enable_gpu: bool = True, monitor_level: MonitorLevel = MonitorLevel.BASIC) -> ResourceMonitor:
    """
    创建简单的资源监控器
    
    Args:
        sampling_interval: 采样间隔
        enable_gpu: 是否启用GPU监控
        monitor_level: 监控级别
    
    Returns:
        ResourceMonitor实例
    """
    # 由于ResourceMonitor是单例模式，这里直接返回实例
    monitor = ResourceMonitor(
        sampling_interval=sampling_interval,
        enable_gpu_monitoring=enable_gpu,
        monitor_level=monitor_level,
        log_level="INFO"
    )
    
    # 添加简单的告警回调
    # def alert_handler(message, metrics):
    #     print(f"🚨 资源告警: {message}")
    
    # monitor.add_alert_callback(alert_handler)
    
    return monitor