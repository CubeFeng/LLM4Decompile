import os
from typing import Dict, Union, Any
from log_utils import global_logger as logger
from exceptions import GhidraExecutionError


class ReadGhidraModule:
    """
    Ghidra反编译结果读取模块
    用于从本地文件中读取ghidra的反编译结果
    """

    def __init__(self, config: Any):
        """
        初始化Ghidra反编译结果读取器

        Args:
            config: 配置对象，应包含必要的配置信息
        """
        self.config = config
        self.logger = logger

    def process(self, previous_results: Union[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        读取指定的Ghidra反编译结果文件

        Args:
            previous_results: 可以是包含文件路径的字典，或者直接是文件路径字符串

        Returns:
            dict: 包含反编译代码和输出文件路径的字典

        Raises:
            GhidraExecutionError: 当读取文件失败时抛出
        """
        self.logger.info("开始读取Ghidra反编译结果...")

        # 获取文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # ghidra_output_file = os.path.join(script_dir, "../../sample_pseudo.txt")
        # ghidra_output_file = os.path.join(script_dir, "../../ghidra_fragment_code.txt")
        ghidra_output_file = os.path.join(script_dir, "../../cwe-020_ghidra_decompiled_code.txt")

        # if isinstance(previous_results, str):
        #     ghidra_output_file = previous_results
        # elif isinstance(previous_results, dict):
        #     ghidra_output_file = previous_results.get('decompile_file') or previous_results.get('output_file')
        # elif hasattr(self.config, 'ghidra_output_file'):
        #     ghidra_output_file = self.config.ghidra_output_file
        # elif isinstance(self.config, dict) and 'ghidra_output_file' in self.config:
        #     ghidra_output_file = self.config['ghidra_output_file']

        # 验证文件路径
        if not ghidra_output_file:
            raise GhidraExecutionError("无法获取Ghidra反编译结果文件路径")

        try:
            # 验证文件
            if not os.path.isfile(ghidra_output_file):
                raise FileNotFoundError(f"Ghidra输出文件不存在或不是文件: {ghidra_output_file}")

            # 尝试多种编码读取文件
            c_decompile = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(ghidra_output_file, 'r', encoding=encoding) as f:
                        c_decompile = f.read()
                        break
                except UnicodeDecodeError:
                    continue

            if c_decompile is None:
                raise UnicodeDecodeError("编码错误", b"", 0, 1, "所有编码尝试均失败")

            self.logger.info(f"成功读取Ghidra反编译结果文件: {ghidra_output_file}, 文件大小: {len(c_decompile)} 字符")
            
            return {
                'raw_code': c_decompile,
                'output_file': ghidra_output_file
            }
        except (FileNotFoundError, IsADirectoryError) as e:
            self.logger.error(f"文件错误: {str(e)}")
            raise GhidraExecutionError(f"Ghidra输出文件访问失败: {str(e)}")
        except UnicodeDecodeError:
            self.logger.error("文件编码错误")
            raise GhidraExecutionError("无法解码Ghidra反编译结果文件")
        except Exception as e:
            self.logger.error(f"读取Ghidra反编译结果时出错: {str(e)}")
            raise GhidraExecutionError(f"读取Ghidra反编译结果失败: {str(e)}")