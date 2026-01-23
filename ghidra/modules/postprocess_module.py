from datetime import datetime
from log_utils import global_logger as logger


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