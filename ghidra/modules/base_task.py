from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTask(ABC):
    """
    任务流中的任务基类。
    所有具体任务必须继承此类并实现 execute 方法。
    """

    def __init__(self, task_id: str, name: str, config):
        self.task_id = task_id
        self.name = name
        self.config = config or {}
        self.result: Any = None
        self.status: str = "pending"  # 可选状态：pending, running, success, failed

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        执行任务的核心逻辑。
        :param context: 任务上下文（如输入数据、全局变量、前序任务结果等）
        :return: 任务执行结果
        """
        pass

    def on_success(self) -> None:
        """任务成功后的回调（可选）"""
        self.status = "success"

    def on_failure(self, error: Exception) -> None:
        """任务失败后的回调（可选）"""
        self.status = "failed"
        # 可在此记录日志、发送告警等

    def process(self, context: Dict[str, Any]) -> Any:
        """
        任务的统一入口，封装执行流程（含异常处理）。
        """
        self.status = "running"
        try:
            self.result = self.execute(context)
            self.on_success()
            return self.result
        except Exception as e:
            self.on_failure(e)
            raise  # 可选择是否重新抛出异常
