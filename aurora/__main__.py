"""包入口模块。

支持通过 `python -m aurora` 运行 CLI：

```bash
python -m aurora turn "Hello Aurora"
python -m aurora doze
python -m aurora sleep
```
"""
from aurora.surface.cli import main


if __name__ == "__main__":
    main()
