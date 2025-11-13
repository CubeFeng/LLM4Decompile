import torch
import psutil

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
        checks['disk_space'] = f'{disk_usage.free / 1024 ** 3:.1f}GB free'
        checks['disk_total'] = f'{disk_usage.total / 1024 ** 3:.1f}GB total'

        # 内存检查
        memory = psutil.virtual_memory()
        checks['system_memory'] = f'{memory.available / 1024 ** 3:.1f}GB available'
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