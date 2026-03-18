"""Package entry module.

Supports running CLI via `python -m aurora`:

```bash
python -m aurora turn "Hello Aurora"
python -m aurora snapshot --relation-id default
python -m aurora status
```
"""
from aurora.surface.cli import main


if __name__ == "__main__":
    main()
