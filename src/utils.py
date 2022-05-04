import pandas as pd
import numpy as np


def component_func(row):
    if row.startswith('tptp'):
        return 'tptp'
    elif row.startswith('update'):
        return "update"
    elif row.startswith('git'):
        return "git"
    elif row.startswith('ci'):
        return "ci"
    elif row.startswith('user'):
        return "user"
    elif row.startswith('sql'):
        return "sql"
    elif row.startswith('ruby'):
        return "ruby"
    elif row.startswith('gef'):
        return "gef"
    elif row.startswith('cdt'):
        return "cdt"
    elif row.startswith('wst'):
        return "wst"
    elif row.startswith('ecf'):
        return "ecf"
    elif row.startswith('jst'):
        return "jst"
    elif row.startswith('tptp'):
        return "tptp"
    elif row.startswith('cdo'):
        return "cdo"
    elif 'package' in row:
        return "package"
    elif 'connector' in row:
        return "connector"
    elif 'report' in row:
        return "report"
    else:
        return row




