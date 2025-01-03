import os
import psutil

def mem_check():
    pid = os.getpid()
    python_process = psutil.Process(pid)
    # percent = psutil.virtual_memory().percent
    memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
    # print(f'RAM usage:{memoryUse} = {percent}%')
    return memoryUse