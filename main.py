from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from libcst_extractor.cli import main


if __name__ == "__main__":
    main()
