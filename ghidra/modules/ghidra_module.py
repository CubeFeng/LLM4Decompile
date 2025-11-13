import os
import tempfile
import subprocess
from ghidra.log_utils import global_logger as logger
from ghidra.exceptions import GhidraExecutionError


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
                result = subprocess.run(command, text=True, capture_output=True, check=True,
                                        timeout=self.config.timeout_duration)
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
