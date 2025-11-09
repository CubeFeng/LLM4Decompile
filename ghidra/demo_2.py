import sys
import tempfile
import torch
import gc
import os
import subprocess
import time
import traceback
import psutil
import threading
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from log_utils import global_logger as logger
from ResourceMonitor import MonitorLevel, ResourceMonitor, create_simple_monitor

# ==================== 配置管理类 ====================
class DecompilerConfig:
    """反编译器配置管理类"""
    
    def __init__(self, config_file=None):
        # 基础路径配置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # 模型配置
        self.model_path = os.path.join(self.script_dir, "../models/llm4decompile-1.3b-v2")

        # vLLM配置
        self.gpus = 1  # 使用的GPU数量
        self.vllm_max_model_len = 4096  # vLLM最大模型长度
        self.vllm_gpu_memory_utilization = 0.9  # GPU内存利用率
        
        # 工具路径配置
        self.ghidra_path = os.path.join(self.script_dir, "ghidra_11.0.3_PUBLIC/support/analyzeHeadless")
        self.postscript = os.path.join(self.script_dir, "decompile.py")
        self.project_path = "."
        self.project_name = "tmp_ghidra_proj"
        
        # 二进制文件配置
        self.binary_filename = "cwe-020"
        self.binary_dir = os.path.join(self.script_dir, "../cwe")

        # self.binary_filename = "libandroid_jni.so"
        # self.binary_dir = os.path.join(self.script_dir, "../samples")

        self.binary_path = os.path.join(self.binary_dir, self.binary_filename)
        self.file_name = self.binary_filename

        # GPU配置
        self.batch_size = 4   # vLLM可以处理更大批次，例如4。在实际测试中，vLLM通常能支持比传统框架大3-5倍的批处理大小，选择4倍是一个平衡性能和稳定性的经验值
        self.batch_accumulation = 1
        self.max_input_length = 2048
        self.max_new_tokens = 2048
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
        self.monitor_performance = True
        self.enable_realtime_monitoring = False  # 启用实时监控
        self.realtime_update_interval = 1.0     # 实时监控更新间隔
        
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

# ==================== 异常处理类 ====================
class DecompilerError(Exception):
    """反编译流程异常基类"""
    pass

class GhidraExecutionError(DecompilerError):
    """Ghidra执行异常"""
    pass

class ModelLoadingError(DecompilerError):
    """模型加载异常"""
    pass

class InferenceError(DecompilerError):
    """推理过程异常"""
    pass

class ResourceError(DecompilerError):
    """资源异常"""
    pass

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
            self.device = get_device(self.config.force_gpu)
            
            # 加载tokenizer（仍然使用Hugging Face的tokenizer）
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.logger.info("Tokenizer加载完成")
            
            # vLLM参数配置
            vllm_kwargs = {
                "model": self.config.model_path,
                "dtype": dtype, # 改为 float16 以兼容计算能力 7.5 的 GPU
                "gpu_memory_utilization": getattr(self.config, 'vllm_gpu_memory_utilization', 0.9),
                "max_model_len": getattr(self.config, 'vllm_max_model_len', self.config.max_input_length + self.config.max_new_tokens),
                "trust_remote_code": True,  # 添加此参数以确保正确处理模型配置
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
                temperature=0.0,
                max_tokens=self.config.max_new_tokens
            )
            
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

# ==================== 健康检查系统 ====================
def health_check():
    """系统健康检查"""
    checks = {}
    
    try:
        # GPU健康检查
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                # 测试小规模内存分配
                test_tensor = torch.randn(100, 100).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                checks['gpu_memory'] = 'healthy'
                checks['gpu_count'] = torch.cuda.device_count()
                checks['gpu_name'] = torch.cuda.get_device_name(0)
            except Exception as e:
                checks['gpu_memory'] = f'unhealthy: {str(e)}'
        else:
            checks['gpu_memory'] = 'no_cuda_device'
        
        # 磁盘空间检查
        disk_usage = psutil.disk_usage('.')
        checks['disk_space'] = f'{disk_usage.free / 1024**3:.1f}GB free'
        checks['disk_total'] = f'{disk_usage.total / 1024**3:.1f}GB total'
        
        # 内存检查
        memory = psutil.virtual_memory()
        checks['system_memory'] = f'{memory.available / 1024**3:.1f}GB available'
        checks['memory_percent'] = f'{memory.percent}% used'
        
        # CPU检查
        checks['cpu_cores'] = psutil.cpu_count(logical=False)
        checks['cpu_threads'] = psutil.cpu_count(logical=True)
        checks['cpu_usage'] = f'{psutil.cpu_percent(interval=0.1)}%'
        
        checks['overall_status'] = 'healthy' if 'unhealthy' not in str(checks) else 'degraded'
        
    except Exception as e:
        checks['overall_status'] = 'check_failed'
        checks['error'] = str(e)
    
    return checks

# ==================== 模块化流水线 ====================
class DecompilerPipeline:
    """反编译器模块化流水线"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        self.modules = {}
        self.results = {}
        self.performance_monitor = create_simple_monitor(
            sampling_interval=1.0, 
            enable_gpu=True,
            monitor_level=MonitorLevel.BASIC
        )
    
    def register_module(self, name, module):
        """注册处理模块"""
        self.modules[name] = module
        self.logger.debug(f"注册模块: {name}")
    
    def execute_pipeline(self):
        """执行处理流水线"""
        modules_order = ['ghidra', 'preprocess', 'model_inference', 'postprocess']
        
        self.logger.info("开始执行反编译流水线")
        
        for module_name in modules_order:
            if module_name in self.modules:
                try:
                    self.logger.info(f"🎯 开始执行模块: {module_name}")
                    
                    # 记录模块开始时间
                    module_start_time = time.time()
                    
                    # 性能监控
                    with self.performance_monitor.time_block(module_name):
                        result = self.modules[module_name].process(self.results)
                    
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

# ==================== 具体模块实现 ====================
class GhidraModule:
    """Ghidra反编译模块"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
    
    def process(self, previous_results):
        """执行Ghidra反编译"""
        self.logger.info("开始Ghidra反编译...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pid = os.getpid()
            output_path = os.path.join(temp_dir, f"{pid}_binary.c")
            
            # 构建Ghidra命令
            command = [
                self.config.ghidra_path,
                temp_dir,
                self.config.project_name,
                "-import", self.config.binary_path,
                "-postScript", self.config.postscript, output_path,
                "-deleteProject",
            ]
            
            try:
                # 执行Ghidra反编译
                result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=self.config.timeout_duration)
                self.logger.info("Ghidra反编译完成")
                
                # 读取反编译结果
                with open(output_path, 'r', encoding='utf-8') as f:
                    c_decompile = f.read()
                
                # 保存完整反编译代码
                ghidra_output_file = self.config.file_name + '_ghidra_decompiled_code.txt'
                with open(ghidra_output_file, 'w', encoding='utf-8') as f:
                    f.write(c_decompile)
                
                self.logger.info(f"Ghidra反编译代码已保存到: {ghidra_output_file}")
                
                return {
                    'raw_code': c_decompile,
                    'output_file': ghidra_output_file
                }
                
            except subprocess.CalledProcessError as e:
                raise GhidraExecutionError(f"Ghidra执行失败: {str(e)}\n错误输出: {e.stderr}")
            except subprocess.TimeoutExpired:
                raise GhidraExecutionError("Ghidra执行超时")
            except Exception as e:
                raise GhidraExecutionError(f"Ghidra处理失败: {str(e)}")

class PreprocessModule:
    """预处理模块"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        self.min_function_length = getattr(config, 'min_function_length', 10)
    
    def process(self, previous_results):
        """预处理反编译代码"""
        self.logger.info("开始预处理反编译代码...")
        
        if 'ghidra' not in previous_results:
            raise DecompilerError("缺少Ghidra反编译结果")
        
        try:
            c_decompile = previous_results['ghidra']['raw_code']
            
            # 分割函数
            all_lines = c_decompile.split('\n')
            functions = []
            current_func = []
            in_multiline_comment = False
            
            for line_idx, line in enumerate(all_lines):
                try:
                    # 处理多行注释
                    if in_multiline_comment:
                        if '*/' in line:
                            line = line.split('*/', 1)[1].strip()
                            in_multiline_comment = False
                            if not line:
                                continue
                    else:
                        if '//' in line and '// Function:' not in line:
                            code_part = line.split('//', 1)[0].strip()
                            if not code_part:
                                continue
                            line = code_part
                        
                        if '/*' in line:
                            if '*/' in line:
                                code_before = line.split('/*', 1)[0].strip()
                                code_after = line.split('*/', 1)[1].strip()
                                line = code_before + ' ' + code_after
                            else:
                                code_part = line.split('/*', 1)[0].strip()
                                if code_part:
                                    line = code_part
                                else:
                                    in_multiline_comment = True
                                    continue
                except Exception as e:
                    self.logger.warning(f"处理第{line_idx+1}行时出错: {str(e)}")
                
                # 函数分割逻辑
                if '// Function:' in line:
                    if len(current_func) > 0:
                        functions.append('\n'.join(current_func))
                    current_func = []
                else:
                    if line.strip():
                        current_func.append(line)
            
            if current_func:
                functions.append('\n'.join(current_func))
            
            # 使用改进的函数过滤逻辑
            filtered_functions = self._filter_functions(functions)
            
            self.logger.info(f"代码分割完成: 总共 {len(functions)} 个函数，过滤后 {len(filtered_functions)} 个函数")
            
            return {
                'all_functions': functions,
                'filtered_functions': filtered_functions
            }
        except Exception as e:
            self.logger.error(f"预处理过程中发生错误: {str(e)}")
            raise DecompilerError(f"预处理失败: {str(e)}")
    
    def _filter_functions(self, functions):
        """改进的函数过滤逻辑"""
        filtered = []
        for func in functions:
            lines = func.strip().split('\n')
            # 1. 基本长度过滤
            if len(func.strip()) < self.min_function_length:
                continue
            
            # 2. 过滤只有声明没有实现的函数
            if len(lines) <= 3 and ('{' not in func or '}' not in func):
                continue
            
            filtered.append(func)
        return filtered

class ModelInferenceModule:
    """模型推理模块"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
    
    def process(self, previous_results):
        """执行模型推理"""
        self.logger.info("开始模型推理优化...")
        
        if 'preprocess' not in previous_results:
            raise DecompilerError("缺少预处理结果")
        
        filtered_functions = previous_results['preprocess']['filtered_functions']
        
        # 使用模型管理器
        with ModelManager(self.config) as model_mgr:
            return self._process_with_model(model_mgr, filtered_functions)
    
    def _process_with_model(self, model_mgr, filtered_functions):
        """使用模型处理函数"""
        optimized_functions = []
        processed_count = 0
        failed_count = 0
        total_filtered = len(filtered_functions)
        
        self.logger.info(f"开始处理 {total_filtered} 个函数")
        
        # 动态计算批处理大小
        dynamic_batch_size = self._calculate_optimal_batch_size(model_mgr, filtered_functions)
        self.logger.info(f"使用动态批处理大小: {dynamic_batch_size}")
        
        # 分批处理函数
        for i in range(0, total_filtered, dynamic_batch_size):
            batch_end = min(i + dynamic_batch_size, total_filtered)
            batch_indices = list(range(i, batch_end))
            
            try:
                batch_results = self._process_batch(model_mgr, filtered_functions, batch_indices, i, total_filtered)
                processed_count += len(batch_results)
                optimized_functions.extend(batch_results)
                
            except Exception as e:
                self.logger.error(f"批处理 {i//dynamic_batch_size+1} 失败: {str(e)}")
                failed_count += len(batch_indices)
                # 出错时添加原始函数
                for idx in batch_indices:
                    optimized_functions.append(f"// Function {idx+1} (处理失败)\n" + filtered_functions[idx])
        
        # 合并优化后的代码
        all_optimized_code = '\n\n'.join(optimized_functions)
        
        # 保存结果
        refined_output_file = self.config.file_name + '_binary_refined.c'
        refined_output_file_txt = self.config.file_name + '_model_refined_code.txt'
        
        with open(refined_output_file, 'w', encoding='utf-8') as f:
            f.write(all_optimized_code)
        with open(refined_output_file_txt, 'w', encoding='utf-8') as f:
            f.write(all_optimized_code)
        
        self.logger.info(f"模型推理完成: 成功 {processed_count}, 失败 {failed_count}")
        
        return {
            'optimized_code': all_optimized_code,
            'refined_c_file': refined_output_file,
            'refined_txt_file': refined_output_file_txt,
            'processed_count': processed_count,
            'failed_count': failed_count
        }
    
    def _calculate_optimal_batch_size(self, model_mgr, sample_functions):
        """动态计算最优批处理大小"""
        if not torch.cuda.is_available():
            return 1
        
        # 对于vLLM，我们可以使用更大的批处理大小，因为它有更好的内存管理
        current_batch_size = min(self.config.batch_size, 32)  # vLLM可以处理更大批次
        
        if len(sample_functions) <= current_batch_size:
            return min(current_batch_size, len(sample_functions))
        
        # vLLM有自动内存管理，所以我们不需要像Transformers那样逐步减小批处理大小
        # 但为了安全起见，我们可以进行一次简单的测试
        try:
            test_prompts = []
            for i in range(min(4, len(sample_functions))):  # 只测试4个样本
                prompt = self._preprocess_prompt(sample_functions[i])
                test_prompts.append(prompt)
            
            # 简单测试vLLM是否能工作
            model_mgr.llm.generate(test_prompts, model_mgr.sampling_params)
            self.logger.info(f"vLLM批处理测试通过，使用批处理大小: {current_batch_size}")
            return current_batch_size
        except Exception as e:
            self.logger.warning(f"vLLM批处理测试失败: {str(e)}，使用较小的批处理大小")
            return max(1, current_batch_size // 2)
            
            return 1
    
    def _preprocess_prompt(self, func):
        """预处理单个函数的提示文本"""
        before = "# This is the assembly code:\n"
        after = "\n# What is the source code?\n"
        prompt = before + func.strip() + after
        
        if len(prompt) > self.config.max_input_length * 4:
            prompt = prompt[:self.config.max_input_length * 4]
        
        return prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError, InferenceError))
    )
    
    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RuntimeError, InferenceError))
    )
    def _process_batch(self, model_mgr, filtered_functions, batch_indices, start_idx, total_count):
        """处理单个批次（带重试机制）"""
        batch_prompts = []
        for idx in batch_indices:
            prompt = self._preprocess_prompt(filtered_functions[idx])
            batch_prompts.append(prompt)
        
        # 使用vLLM进行推理
        try:
            # vLLM自动处理批处理和填充
            outputs = model_mgr.llm.generate(batch_prompts, model_mgr.sampling_params)
            
            # 处理结果
            batch_results = []
            for j, output in enumerate(outputs):
                # 获取生成的文本
                optimized_code = output.outputs[0].text.strip()
                batch_results.append(optimized_code)
            
            current_progress = min(start_idx + len(batch_indices), total_count)
            self.logger.info(f"  批处理进度: {current_progress}/{total_count}")
            
            return batch_results
        except Exception as e:
            self.logger.error(f"vLLM推理失败: {str(e)}")
            raise InferenceError(f"vLLM推理失败: {str(e)}")


class PostprocessModule:
    """后处理模块"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
    
    def process(self, previous_results):
        """执行后处理"""
        # 汇总所有结果
        final_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'output_files': []
        }
        
        # 收集所有输出文件
        if 'ghidra' in previous_results:
            final_result['output_files'].append(previous_results['ghidra']['output_file'])
        
        if 'model_inference' in previous_results:
            final_result['output_files'].append(previous_results['model_inference']['refined_c_file'])
            final_result['output_files'].append(previous_results['model_inference']['refined_txt_file'])
            final_result['processed_count'] = previous_results['model_inference']['processed_count']
            final_result['failed_count'] = previous_results['model_inference']['failed_count']
        
        return final_result

# ==================== 工具函数 ====================
def get_device(force_gpu=False):
    """获取可用的设备"""
    if force_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"强制使用GPU: {gpu_name} ({total_memory:.2f} GB)")
            return device
        else:
            raise ResourceError("强制使用GPU，但未检测到可用的CUDA设备")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        return device

# ==================== 主程序 ====================
def main():
    """主程序入口"""
    # 初始化配置和日志
    config = DecompilerConfig()
    
    logger.info("======= 开始二进制反编译流程 =======")
    
    try:
        # 创建资源监控器
        resource_monitor = create_simple_monitor(
            sampling_interval=config.monitor_interval,
            enable_gpu=True,
            monitor_level=MonitorLevel.BASIC
        )
        
        # 启动监控
        resource_monitor.start_monitoring()
        
        # 启用实时监控显示（根据配置）
        if config.enable_realtime_monitoring:
            resource_monitor.enable_realtime_monitoring(
                update_interval=config.realtime_update_interval
            )
            logger.info("✅ 实时监控已启用")
        
        # 设置告警阈值
        resource_monitor.set_alert_threshold('cpu_percent', 85.0)
        resource_monitor.set_alert_threshold('memory_percent', 80.0)
        resource_monitor.set_alert_threshold('gpu_utilization', 95.0)
        resource_monitor.set_alert_threshold('gpu_memory_percent', 90.0)
        
        # 添加告警回调
        def alert_handler(message, metrics):
            logger.warning(f"🚨 资源告警: {message}")
            # 可以在告警时执行特定操作，如保存快照、发送通知等
        
        resource_monitor.add_alert_callback(alert_handler)
        
        # 创建流水线
        pipeline = DecompilerPipeline(config)
        
        # 注册模块
        pipeline.register_module('ghidra', GhidraModule(config))
        pipeline.register_module('preprocess', PreprocessModule(config))
        pipeline.register_module('model_inference', ModelInferenceModule(config))
        pipeline.register_module('postprocess', PostprocessModule(config))
        
        # 执行流水线
        results = pipeline.execute_pipeline()
        
        # 获取最终统计信息
        stats = resource_monitor.get_statistics()
        
        # 停止实时监控显示
        if config.enable_realtime_monitoring:
            resource_monitor.disable_realtime_monitoring()
        
        # 停止监控
        resource_monitor.stop_monitoring()
        
        # 打印最终统计报告
        # logger.info("="*60)
        # logger.info("📊 反编译流程资源使用统计")
        # logger.info("="*60)
        
        # if stats:
        #     logger.info(f"监控时长: {stats['time_range']['duration_seconds']:.1f} 秒")
        #     logger.info(f"采样数量: {stats['sample_count']} 次")
            
        #     logger.info(f"📈 CPU使用率: {stats['cpu']['avg']:.1f}% (峰值: {stats['cpu']['max']:.1f}%)")
        #     logger.info(f"📈 内存使用率: {stats['memory']['avg']:.1f}% (峰值: {stats['memory']['max']:.1f}%)")
            
        #     if 'gpu_utilization' in stats:
        #         logger.info(f"🎮 GPU使用率: {stats['gpu_utilization']['avg']:.1f}% (峰值: {stats['gpu_utilization']['max']:.1f}%)")
            
        #     if 'gpu_memory' in stats:
        #         logger.info(f"🎯 平均显存使用: {stats['gpu_memory']['avg']:.1f} MB")
        
        logger.info("======= 反编译流程成功完成 =======")
        
        return {
            'success': True,
            'results': results,
            'resource_statistics': stats
        }
            
    except Exception as e:
        logger.error(f"反编译流程失败: {str(e)}")
        logger.debug("错误详情:", exc_info=True)
        
        # 确保监控被停止
        try:
            if 'resource_monitor' in locals():
                if config.enable_realtime_monitoring:
                    resource_monitor.disable_realtime_monitoring()
                resource_monitor.stop_monitoring()
        except:
            pass
        
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

if __name__ == "__main__":
    # 设置PyTorch优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 运行主程序
    result = main()
    
    if result['success']:
        print("✅ 反编译流程成功完成")
    else:
        print(f"❌ 反编译流程失败: {result['error']}")
        exit(1)