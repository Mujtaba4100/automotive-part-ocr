"""
Top-level launcher script.

Allows the application to be started from the project root with::

    python main.py

It simply delegates to the real entry-point inside the ``app`` package.
"""

from app.main import main

if __name__ == "__main__":
    main()
