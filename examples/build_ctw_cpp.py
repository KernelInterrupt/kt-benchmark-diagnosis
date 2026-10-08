import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pykt.utils.ctw_cpp import build_ctw_cpp_shared


def main():
    path = build_ctw_cpp_shared(force=True)
    print(path)


if __name__ == "__main__":
    main()
