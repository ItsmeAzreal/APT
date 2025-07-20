import torch
import pynvml

def print_device_info():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Name:", torch.cuda.get_device_name(i))
            print(f"GPU {i} Memory Total (MB):", torch.cuda.get_device_properties(i).total_memory // 1024**2)
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU {i} Temp (C): {temp}, Utilization: {util.gpu}%")
