import torch

from config import DecompilerConfig
from decompiler_pipeline import DecompilerPipeline
from modules.ghidra_module import GhidraTask
from modules.read_ghidra_module import ReadGhidraModule
from modules.model_interence_module import ModelInferenceModule
from modules.postprocess_module import PostprocessModule
from modules.preprocess_module import PreprocessModule
from log_utils import global_logger as logger
from resource_monitor import MonitorLevel, create_simple_monitor



# ==================== 主程序 ====================
def main():
    """主程序入口"""
    # 初始化配置和日志
    global resource_monitor
    config = DecompilerConfig()

    logger.info("======= 开始二进制反编译流程 =======")

    try:
        # 创建资源监控器
        resource_monitor = create_simple_monitor(
            sampling_interval=config.monitor_interval,
            enable_gpu=True,
            monitor_level=MonitorLevel.EXTENDED
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
        pipeline.register_module('ghidra', GhidraTask(config))
        # pipeline.register_module('ghidra', ReadGhidraModule(config))
        pipeline.register_module('preprocess', PreprocessModule(config))
        pipeline.register_module('model_inference', ModelInferenceModule(config))
        pipeline.register_module('postprocess', PostprocessModule(config))

        # 执行流水线
        results = pipeline.execute_pipeline()

        # 获取最终统计信息
        # stats = resource_monitor.get_statistics()

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
            # 'resource_statistics': stats
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
