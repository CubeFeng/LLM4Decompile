import os
import torch
from datetime import datetime
from ghidra.log_utils import global_logger as logger
from ghidra.exceptions import DecompilerError, InferenceError
from ghidra.modle_manager import ModelManager
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


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
                self.logger.error(f"批处理 {i // dynamic_batch_size + 1} 失败: {str(e)}")
                failed_count += len(batch_indices)
                # 出错时添加原始函数
                for idx in batch_indices:
                    optimized_functions.append(f"// Function {idx + 1} (处理失败)\n" + filtered_functions[idx])

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

        # 对于vLLM，我们可以使用更大的批处理大小
        current_batch_size = min(self.config.batch_size, 32)  # vLLM可以处理更大批次

        if len(sample_functions) <= current_batch_size:
            return min(current_batch_size, len(sample_functions))

        # vLLM有自动内存管理，进行一次简单的测试
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
