"""包入口模块。

支持通过 `python -m aurora` 运行 CLI：

```bash
python -m aurora turn "Hello Aurora"
python -m aurora snapshot --relation-id default
python -m aurora status
```
"""
from aurora.surface.cli import main


if __name__ == "__main__":
    main()
