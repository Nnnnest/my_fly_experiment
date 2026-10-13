"""python -m analysis.run_all [--force]"""
import sys

from analysis import validate, tables, figures

if not validate.validate() and "--force" not in sys.argv:
    sys.exit("validation failed; fix the data or rerun with --force")
tables.main()
figures.main()
