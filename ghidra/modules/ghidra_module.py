import os
import tempfile
import subprocess
from typing import Dict, Any

from log_utils import global_logger as logger
from exceptions import GhidraExecutionError
from modules.base_task import BaseTask


class GhidraTask(BaseTask):

    # def __init__(self, task_id: str, config):
    def __init__(self,  config):
        super().__init__("task_id", self.__class__.__name__, config)
        self.config = config
        self.logger = logger

    def execute(self, context: Dict[str, Any]) -> Any:
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
                "-max-cpu", str(int(os.cpu_count() or 8) // 2),  # 修复为整数并处理None情况
            ]

            try:
                # 执行Ghidra反编译
                result = subprocess.run(command, text=True, capture_output=True, check=True,
                                        # timeout=self.config.timeout_duration,  # 不设置超时时间，程序一直执行（主要针对大文件）
                                        )
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
