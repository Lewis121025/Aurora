import sys
import os
import numpy as np
import pytest

# 确保能导入同目录下的“算法任务.py”
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from 算法任务 import MemoryEngineLab
except ImportError:
    pytest.fail("!!! 无法找到 算法任务.py 文件 !!!")

... (测试代码内容同原 test.py)
