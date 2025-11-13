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
