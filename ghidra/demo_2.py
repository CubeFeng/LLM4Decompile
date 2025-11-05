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
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==================== 配置管理类 ====================
class DecompilerConfig:
    """反编译器配置管理类"""
    
    def __init__(self, config_file=None):
        # 基础路径配置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # 模型配置
        self.model_path = os.path.join(self.script_dir, "../models/llm4decompile-1.3b-v2")
        
        # 工具路径配置
        self.ghidra_path = os.path.join(self.script_dir, "ghidra_11.0.3_PUBLIC/support/analyzeHeadless")
        self.postscript = os.path.join(self.script_dir, "decompile.py")
        self.project_path = "."
        self.project_name = "tmp_ghidra_proj"
        
        # 二进制文件配置
        self.binary_path = os.path.join(self.script_dir, "../samples/cwe-020")
        # self.binary_path = os.path.join(self.script_dir, "../samples/libandroid_jni.so")
        self.file_name = "cwe-020"
        
        # GPU配置
        self.batch_size = 1
        self.batch_accumulation = 1
        self.max_input_length = 2048
        self.max_new_tokens = 2048
        self.use_multi_gpu = False
        self.force_gpu = True
        
        # 性能优化配置
        self.enable_compile = True
        self.enable_optimized_data_loading = True
        self.use_prefetch = True
        self.num_prefetch_workers = 2
        self.enable_cuda_graph = False
        
        # 监控配置
        self.monitor_interval = 0.5
        self.timeout_duration = 100
        
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
                    
            logging.info(f"从文件 {config_file} 加载配置成功")
        except Exception as e:
            logging.warning(f"加载配置文件失败: {str(e)}，使用默认配置")
    
    def save_to_file(self, config_file):
        """保存配置到JSON文件"""
        try:
            config_dict = {key: getattr(self, key) for key in dir(self) 
                          if not key.startswith('_') and not callable(getattr(self, key))}
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
            logging.info(f"配置已保存到 {config_file}")
        except Exception as e:
            logging.error(f"保存配置文件失败: {str(e)}")
    
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

# ==================== 资源管理器 ====================
class GPUResourceManager:
    """GPU资源管理器（上下文管理器）"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('decompiler')
    
    def __enter__(self):
        self.cleanup()
        self.logger.info("GPU资源管理器已启动")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        self.logger.info("GPU资源管理器已清理")
    
    def cleanup(self):
        """清理GPU和系统内存"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            self.logger.warning(f"资源清理过程中出现警告: {str(e)}")

class ModelManager:
    """模型管理器（上下文管理器）"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.logger = logging.getLogger('decompiler')
    
    def __enter__(self):
        self.load_model()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
    
    def load_model(self):
        """加载模型和tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.logger.info(f"正在从 {self.config.model_path} 加载模型...")
            
            # 获取设备
            self.device = get_device(self.config.force_gpu)
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.logger.info("Tokenizer加载完成")
            
            # 加载模型
            model_kwargs = {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "use_cache": True,
                "local_files_only": True  # 本地加载模型，而不是从 HUgging Face HUb 下载
            }
            
            # 设备映射配置
            if self.config.use_multi_gpu and torch.cuda.device_count() > 1 and not self.config.force_gpu:
                model_kwargs["device_map"] = "balanced"
                self.logger.info(f"使用多GPU模式，设备数量: {torch.cuda.device_count()}")
            elif self.config.force_gpu:
                model_kwargs["device_map"] = "cuda:0"
            else:
                model_kwargs["device_map"] = "auto"
            
            try:
                import accelerate
                self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
                self.logger.info("模型成功加载（使用accelerate）")
            except ImportError:
                self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
                self.model = self.model.to(self.device)
                self.logger.info("模型成功加载（使用标准方式）")
            
            # 编译模型优化性能
            if self.config.enable_compile and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(
                        self.model,
                        mode="max-autotune",
                        dynamic=True,
                        fullgraph=True
                    )
                    self.logger.info("模型编译优化完成")
                except Exception as e:
                    self.logger.warning(f"模型编译失败: {str(e)}")
            
            self.logger.info(f"模型已加载到设备: {self.device}")
            
        except Exception as e:
            raise ModelLoadingError(f"模型加载失败: {str(e)}")
    
    def unload_model(self):
        """卸载模型释放内存"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
                
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
        self.logger = logging.getLogger('decompiler')
        self.modules = {}
        self.results = {}
        self.performance_monitor = PerformanceMonitor()
    
    def register_module(self, name, module):
        """注册处理模块"""
        self.modules[name] = module
        self.logger.debug(f"注册模块: {name}")
    
    def execute_pipeline(self):
        """执行处理流水线"""
        modules_order = ['health_check', 'ghidra', 'preprocess', 'model_inference', 'postprocess']
        
        self.logger.info("开始执行反编译流水线")
        
        for module_name in self.modules.keys():
            if module_name in self.modules:
                try:
                    self.logger.info(f"执行模块: {module_name}")
                    
                    # 性能监控
                    start_time = time.time()
                    result = self.modules[module_name].process(self.results)
                    elapsed_time = time.time() - start_time
                    
                    self.results[module_name] = result
                    self.performance_monitor.record_module_time(module_name, elapsed_time)
                    
                    self.logger.info(f"模块 {module_name} 执行完成，耗时: {elapsed_time:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"模块 {module_name} 执行失败: {str(e)}")
                    raise DecompilerError(f"模块 {module_name} 执行失败") from e
            else:
                self.logger.warning(f"未找到模块: {module_name}，跳过")
                    
        self.logger.info("反编译流水线执行完成")
        return self.results

# ==================== 具体模块实现 ====================
class HealthCheckModule:
    """健康检查模块"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('decompiler')
    
    def process(self, previous_results):
        """执行健康检查"""
        self.logger.info("执行系统健康检查...")
        health_status = health_check()
        
        # 记录健康状态
        for check, status in health_status.items():
            self.logger.info(f"  健康检查 - {check}: {status}")
        
        # 检查关键资源
        if health_status.get('overall_status') != 'healthy':
            self.logger.warning("系统健康状态异常，但将继续执行")
        
        return health_status

class GhidraModule:
    """Ghidra反编译模块"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('decompiler')
    
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
        self.logger = logging.getLogger('decompiler')
        # 从配置中获取最小函数长度，如果没有则使用默认值10
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
                            # 多行注释结束，获取注释结束后的内容
                            line = line.split('*/', 1)[1].strip()
                            in_multiline_comment = False
                            if not line:  # 如果注释后没有内容，跳过此行
                                continue
                    else:
                        # 检查是否有单行注释，但是要保留函数标记
                        if '//' in line and '// Function:' not in line:
                            # 保留注释前的代码（如果有）
                            code_part = line.split('//', 1)[0].strip()
                            if not code_part:  # 如果只有注释没有代码，跳过此行
                                continue
                            line = code_part
                        
                        # 检查是否开始多行注释
                        if '/*' in line:
                            if '*/' in line:  # 单行内完成的多行注释
                                code_before = line.split('/*', 1)[0].strip()
                                code_after = line.split('*/', 1)[1].strip()
                                line = code_before + ' ' + code_after
                            else:  # 多行注释开始
                                code_part = line.split('/*', 1)[0].strip()
                                if code_part:  # 保留注释前的代码
                                    line = code_part

                                else:  # 如果只有注释开始标记，跳过此行
                                    in_multiline_comment = True
                                    continue
                except Exception as e:
                    self.logger.warning(f"处理第{line_idx+1}行时出错: {str(e)}")
                    # 出错时保留原始行，继续处理
                
                # 函数分割逻辑
                if '// Function:' in line:  # 保留函数标记
                    # 移除函数注释 Function
                    if len(current_func) > 0:
                        functions.append('\n'.join(current_func))
                    current_func = []

                # if '// Function:' in line and current_func:  # 保留函数标记
                #     functions.append('\n'.join(current_func))
                #     current_func = [line]
                else:
                    if line.strip():  # 只添加非空行
                        current_func.append(line)
            
            if current_func:  # 确保最后一个函数也被添加
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
        self.logger = logging.getLogger('decompiler')
        self.performance_monitor = PerformanceMonitor()
    
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
        
        # 性能监控线程
        monitor_thread = threading.Thread(
            target=self.performance_monitor.continuous_monitoring,
            daemon=True
        )
        monitor_thread.start()
        
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
            return 1  # CPU模式使用较小的批处理大小
        
        current_batch_size = self.config.batch_size
        
        # 如果只有少量函数，不需要调整
        if len(sample_functions) <= current_batch_size:
            return min(current_batch_size, len(sample_functions))
        
        # 测试当前批处理大小是否合适
        while current_batch_size >= 1:
            try:
                # 准备测试数据
                test_prompts = []
                for i in range(min(current_batch_size, len(sample_functions))):
                    prompt = self._preprocess_prompt(sample_functions[i])
                    test_prompts.append(prompt)
                
                # 测试推理
                inputs = model_mgr.tokenizer(
                    test_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_input_length,
                    padding=True
                ).to(model_mgr.device)
                
                with torch.no_grad():
                    model_mgr.model.generate(
                        **inputs,
                        max_new_tokens=10,  # 只生成少量token进行测试
                        use_cache=True,
                        do_sample=False
                    )
                
                torch.cuda.empty_cache()
                self.logger.info(f"批处理大小 {current_batch_size} 测试通过")
                return current_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    current_batch_size //= 2
                    torch.cuda.empty_cache()
                    self.logger.warning(f"批处理大小过大，调整为: {current_batch_size}")
                    continue
                raise
            except Exception as e:
                self.logger.warning(f"批处理大小测试失败: {str(e)}，使用默认大小")
                return self.config.batch_size
        
        return 1  # 最低保证
    
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
    def _process_batch(self, model_mgr, filtered_functions, batch_indices, start_idx, total_count):
        """处理单个批次（带重试机制）"""
        batch_prompts = []
        for idx in batch_indices:
            prompt = self._preprocess_prompt(filtered_functions[idx])
            batch_prompts.append(prompt)
        
        # Tokenization
        inputs = model_mgr.tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
            padding=True
        ).to(model_mgr.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model_mgr.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                pad_token_id=model_mgr.tokenizer.eos_token_id
            )
        
        # 解码结果
        batch_results = []
        for j, idx in enumerate(batch_indices):
            input_len = inputs["input_ids"][j].shape[0]
            if model_mgr.tokenizer.pad_token_id is not None:
                input_len = (inputs["input_ids"][j] != model_mgr.tokenizer.pad_token_id).sum().item()
            
            gen_ids = outputs[j, input_len:]
            optimized_code = model_mgr.tokenizer.decode(gen_ids, skip_special_tokens=True)
            batch_results.append(optimized_code)
            # batch_results.append(f"// Function {idx+1}\n" + optimized_code)  # 20251105 是否包含 Function 标记
        
        current_progress = min(start_idx + len(batch_indices), total_count)
        self.logger.info(f"  批处理进度: {current_progress}/{total_count}")
        
        return batch_results

class PostprocessModule:
    """后处理模块"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('decompiler')
    
    def process(self, previous_results):
        """执行后处理"""
        self.logger.info("执行后处理...")
        
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
        
        self.logger.info("后处理完成")
        return final_result

# ==================== 性能监控系统 ====================
class PerformanceMonitor:
    """性能监控系统"""
    
    def __init__(self):
        self.metrics = {
            "module_times": {},
            "gpu_util_samples": [],
            "cpu_util_samples": [],
            "memory_samples": [],
            "start_time": time.time()
        }
        self.process = psutil.Process(os.getpid())
        self.logger = logging.getLogger('decompiler')
        self._stop_monitoring = False
    
    def record_module_time(self, module_name, elapsed_time):
        """记录模块执行时间"""
        self.metrics["module_times"][module_name] = elapsed_time
    
    def continuous_monitoring(self):
        """持续性能监控"""
        while not self._stop_monitoring:
            try:
                # CPU和内存监控
                cpu_percent = self.process.cpu_percent(interval=None)
                memory_percent = self.process.memory_percent()
                
                self.metrics["cpu_util_samples"].append(cpu_percent)
                self.metrics["memory_samples"].append(memory_percent)
                
                # GPU监控
                if torch.cuda.is_available():
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                        self.metrics["gpu_util_samples"].append(gpu_util)
                        pynvml.nvmlShutdown()
                    except ImportError:
                        pass
                
                time.sleep(0.5)  # 监控间隔
                
            except Exception as e:
                self.logger.debug(f"性能监控采样失败: {str(e)}")
                break
    
    def stop_monitoring(self):
        """停止监控"""
        self._stop_monitoring = True
    
    def generate_report(self):
        """生成性能报告"""
        total_time = time.time() - self.metrics["start_time"]
        
        report = {
            "total_execution_time": total_time,
            "module_breakdown": self.metrics["module_times"],
            "average_cpu_usage": sum(self.metrics["cpu_util_samples"]) / len(self.metrics["cpu_util_samples"]) if self.metrics["cpu_util_samples"] else 0,
            "average_memory_usage": sum(self.metrics["memory_samples"]) / len(self.metrics["memory_samples"]) if self.metrics["memory_samples"] else 0,
            "average_gpu_usage": sum(self.metrics["gpu_util_samples"]) / len(self.metrics["gpu_util_samples"]) if self.metrics["gpu_util_samples"] else 0,
        }
        
        self.logger.info("=== 性能报告 ===")
        self.logger.info(f"总执行时间: {report['total_execution_time']:.2f}s")
        for module, time_taken in report['module_breakdown'].items():
            self.logger.info(f"  {module}: {time_taken:.2f}s")
        self.logger.info(f"平均CPU使用率: {report['average_cpu_usage']:.1f}%")
        self.logger.info(f"平均内存使用率: {report['average_memory_usage']:.1f}%")
        self.logger.info(f"平均GPU使用率: {report['average_gpu_usage']:.1f}%")
        
        return report

# ==================== 工具函数 ====================
def get_device(force_gpu=False):
    """获取可用的设备"""
    if force_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"强制使用GPU: {gpu_name} ({total_memory:.2f} GB)")
            return device
        else:
            raise ResourceError("强制使用GPU，但未检测到可用的CUDA设备")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {device}")
        return device

def setup_logging(log_level=logging.INFO):
    """设置日志系统"""
    logger = logging.getLogger('decompiler')
    logger.setLevel(log_level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 文件处理器
    log_file = f'decompiler_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    return logger

# ==================== 主程序 ====================
def main():
    """主程序入口"""
    # 初始化配置和日志
    config = DecompilerConfig()
    logger = setup_logging()
    
    logger.info("===== 开始二进制反编译流程 =====")
    
    try:
        # 健康检查
        health_status = health_check()
        logger.info("系统健康状态检查完成")
        
        # 使用资源管理器
        with GPUResourceManager(config):
            # 创建流水线
            pipeline = DecompilerPipeline(config)
            
            # 注册模块
            # pipeline.register_module('health_check', HealthCheckModule(config))
            pipeline.register_module('ghidra', GhidraModule(config))
            pipeline.register_module('preprocess', PreprocessModule(config))
            pipeline.register_module('model_inference', ModelInferenceModule(config))
            pipeline.register_module('postprocess', PostprocessModule(config))
            
            # 执行流水线
            results = pipeline.execute_pipeline()
            
            # 生成性能报告
            # performance_report = pipeline.performance_monitor.generate_report()
            
            logger.info("===== 反编译流程成功完成 =====")
            
            return {
                'success': True,
                'results': results,
                # 'performance': performance_report,
                'health_status': health_status
            }
            
    except Exception as e:
        logger.error(f"反编译流程失败: {str(e)}")
        logger.debug("错误详情:", exc_info=True)
        
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
        print(f"📊 总执行时间: {result['performance']['total_execution_time']:.2f}s")
    else:
        print(f"❌ 反编译流程失败: {result['error']}")
        exit(1)