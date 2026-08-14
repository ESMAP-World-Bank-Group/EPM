import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = r"C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Azerbaijan\SSC"

def show_all_sheets(fname, label):
    path = f"{BASE}\\{fname}"
    xl = pd.ExcelFile(path, engine='xlrd')
    print(f"\n{'='*70}")
    print(f"FILE: {fname}  ({label})")
    print(f"Sheets: {xl.sheet_names}")
    for sname in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sname, header=None, engine='xlrd')
        print(f"\n  --- Sheet: '{sname}' ---")
        print(df.to_string())
    return xl

# 1) 005_3en.xls — Installed capacity (MW) by type
show_all_sheets("005_3en.xls", "Installed electricity capacity (MW)")

# 2) 005_4en.xls — Electricity generation (GWh)
show_all_sheets("005_4en.xls", "Electricity generation (GWh)")

# 3) 003_1.16en.xls — Hydro energy balance 2007-2024
show_all_sheets("003_1.16en.xls", "Hydro energy balance 2007-2024")

# 4) 003_1.17en.xls — Solar energy balance 2007-2024
show_all_sheets("003_1.17en.xls", "Solar energy balance 2007-2024")

# 5) 003_1.18en.xls — Wind energy balance 2007-2024
show_all_sheets("003_1.18en.xls", "Wind energy balance 2007-2024")

# 6) 005_8en.xls — Electricity consumption by region 2015-2024
show_all_sheets("005_8en.xls", "Electricity consumption by region 2015-2024")

# 7) 002_a-b_en.xls — Summary energy balance 2007-2024
show_all_sheets("002_a-b_en.xls", "Summary energy balance 2007-2024")
