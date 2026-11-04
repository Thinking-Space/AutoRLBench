"""Trusted entry point for the pinned Python 3.9 test interpreter."""
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**2, 4 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    main()
