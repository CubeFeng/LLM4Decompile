import os
import json

from exceptions import ResourceError
from log_utils import global_logger as logger


def get_device(force_gpu=False):
    """获取可用的设备"""
    import torch
    if force_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"强制使用GPU: {gpu_name} ({total_memory:.2f} GB)")
            return device
        else:
            raise ResourceError("强制使用GPU，但未检测到可用的CUDA设备")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        return device


class DecompilerConfig:
    """反编译器配置管理类"""

    def __init__(self, config_file=None):
        # 基础路径配置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # 模型配置
        self.model_path = os.path.join(self.script_dir, "../models/llm4decompile-1.3b-v2")

        # vLLM配置
        self.gpus = 1  # 使用的GPU数量
        self.vllm_max_model_len = 8192  # vLLM最大模型长度
        self.vllm_gpu_memory_utilization = 0.9  # GPU内存利用率

        # 工具路径配置
        self.ghidra_path = os.path.join(self.script_dir, "ghidra_11.0.3_PUBLIC/support/analyzeHeadless")
        # self.ghidra_path = os.path.join(self.script_dir, "ghidra_11.1.2_PUBLIC/support/analyzeHeadless")
        self.postscript = os.path.join(self.script_dir, "decompile.py")
        self.project_path = "."
        self.project_name = "tmp_ghidra_proj"

        # 二进制文件配置
        self.binary_filename = "cwe-020"
        # self.binary_filename = "bin_init"
        self.binary_dir = os.path.join(self.script_dir, "../cwe")
        self.binary_path = os.path.join(self.binary_dir, self.binary_filename)
        self.file_name = self.binary_filename

        # GPU配置
        self.batch_size = 1
        self.batch_accumulation = 1
        self.max_input_length = 4048
        self.max_new_tokens = 4048
        self.use_multi_gpu = False
        self.force_gpu = True

        # 性能优化配置
        self.enable_compile = False  # vLLM不需要这个配置
        self.enable_optimized_data_loading = True
        self.use_prefetch = True
        self.num_prefetch_workers = 2
        self.enable_cuda_graph = False  # vLLM不需要这个配置

        # 监控配置
        self.monitor_interval = 0.5
        self.timeout_duration = 100
        self.monitor_performance = False
        self.enable_realtime_monitoring = False  # 启用实时监控
        self.realtime_update_interval = 1.0  # 实时监控更新间隔

        # 如果提供了配置文件，则从文件加载配置
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def load_from_file(self, config_file):
        """从JSON文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            logger.info(f"从文件 {config_file} 加载配置成功")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {str(e)}，使用默认配置")

    def save_to_file(self, config_file):
        """保存配置到JSON文件"""
        try:
            config_dict = {key: getattr(self, key) for key in dir(self)
                           if not key.startswith('_') and not callable(getattr(self, key))}

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"配置已保存到 {config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")

    @property
    def device(self):
        """延迟获取设备"""
        return get_device(self.force_gpu)