import torch
import gc
from log_utils import global_logger as logger
from exceptions import ModelLoadingError


class ModelManager:
    """模型管理器（上下文管理器）"""

    def __init__(self, config):
        self.config = config
        self.llm = None
        self.tokenizer = None
        self.device = None
        self.logger = logger
        self.sampling_params = None

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()

    def load_model(self):
        """加载模型和tokenizer"""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            self.logger.info(f"正在从 {self.config.model_path} 加载模型...")

            # 检查硬件支持
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                support_bfloat16 = capability[0] >= 8
            else:
                support_bfloat16 = False

            # 选择合适的数据类型
            if support_bfloat16:
                dtype = torch.bfloat16
                self.logger.info("dtype 使用 bfloat16 数据类型")
            else:
                dtype = torch.float16
                self.logger.info("dtype 使用 float16 数据类型")

            # 获取设备
            self.device = self.config.device

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.logger.info("Tokenizer加载完成")

            # vLLM参数配置
            vllm_kwargs = {
                "model": self.config.model_path,
                "dtype": dtype,
                "gpu_memory_utilization": getattr(self.config, 'vllm_gpu_memory_utilization', 0.9),
                "max_model_len": getattr(self.config, 'vllm_max_model_len',
                                         self.config.max_input_length + self.config.max_new_tokens),
                "trust_remote_code": True,
                "max_num_seqs": self.config.batch_size,
            }

            # 如果配置了多GPU
            if hasattr(self.config, 'gpus') and self.config.gpus > 1:
                vllm_kwargs["tensor_parallel_size"] = self.config.gpus
                self.logger.info(f"使用多GPU模式，设备数量: {self.config.gpus}")

            # 加载vLLM模型
            self.llm = LLM(**vllm_kwargs)
            self.logger.info(f"vLLM模型加载成功")

            # 配置采样参数
            self.sampling_params = SamplingParams(
                temperature=0.0, # 要求逻辑精准，不能创意。
                max_tokens=self.config.max_new_tokens
            )

            # self.sampling_params = SamplingParams(
            #     temperature=0.0,          # 低温度：减少随机性，提高确定性和一致性
            #     # top_p=0.9,                # nucleus sampling：保留高概率 token，兼顾多样性与稳定性
            #     # top_k=50,                 # 可选：限制每步只从 top 50 个 token 中采样（进一步稳定输出）
            #     max_tokens=self.config.max_new_tokens,           # 足够容纳一段结构清晰的伪代码（含注释和缩进）
            #     stop_token_ids=None,      # 默认即可；若模型有特殊 EOS 可指定
            #     # repetition_penalty=1.1,   # 轻微惩罚重复，避免啰嗦（vLLM 0.7.3+ 支持）
            #     skip_special_tokens=True  # 解码时跳过特殊 token（如 <s>, </s>）
            # )

        except Exception as e:
            raise ModelLoadingError(f"模型加载失败: {str(e)}")

    def unload_model(self):
        """卸载模型释放内存"""
        try:
            if self.llm:
                del self.llm
            if self.tokenizer:
                del self.tokenizer

            # 清理PyTorch分布式进程组
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
                    self.logger.info("PyTorch分布式进程组已销毁")
            except (ImportError, AttributeError):
                # 如果没有初始化分布式进程组或模块不存在，忽略错误
                pass

            torch.cuda.empty_cache()
            gc.collect()

            self.logger.info("模型已卸载，内存已释放")
        except Exception as e:
            self.logger.warning(f"模型卸载过程中出现警告: {str(e)}")