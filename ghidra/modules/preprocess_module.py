from log_utils import global_logger as logger
from exceptions import DecompilerError


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
                    self.logger.warning(f"处理第{line_idx + 1}行时出错: {str(e)}")

                # 函数分割逻辑
                if '// Function:' in line:
                    if len(current_func) > 0:
                        functions.append('\n'.join(current_func))
                    current_func = [line]
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