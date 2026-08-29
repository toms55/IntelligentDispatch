import nemosis as n
from datetime import datetime
from pathlib import Path
import pandas as pd

data_cache = str(Path(__file__).resolve().parent.parent / "data_cache")
Path(data_cache).mkdir(exist_ok=True)  # nemosis needs the dir to exist

def fetch_dispatch_data(start_time: datetime, end_time: datetime, selected_columns: list[str]) -> pd.DataFrame:
    """
    fetch data from NEMOSIS dispatch region sum table.

    typical arg for selected_columns: 
        ['SETTLEMENTDATE', 'REGIONID', 'TOTALDEMAND', 'AVAILABLEGENERATION', 'DEMANDFORECAST', 'SEMISCHEDULE_CLEAREDMW', 'NETINTERCHANGE', 'UIGF', 'TOTALINTERMITTENTGENERATION']
    """
    dispatch_data = n.dynamic_data_compiler(
        start_time.strftime("%Y/%m/%d %H:%M:%S"),
        end_time.strftime("%Y/%m/%d %H:%M:%S"),
        "DISPATCHREGIONSUM",
        data_cache,
        select_columns=selected_columns,
    )
    return dispatch_data 

def fetch_price_data(start_time: datetime, end_time: datetime, selected_columns: list[str]) -> pd.DataFrame:
    """
    fetch data from NEMOSIS dispatch price table.

    typical arg for selected_columns:
        ['SETTLEMENTDATE', 'REGIONID', 'RRP']
    """

    price_data = n.dynamic_data_compiler(
        start_time.strftime("%Y/%m/%d %H:%M:%S"),
        end_time.strftime("%Y/%m/%d %H:%M:%S"),
        "DISPATCHPRICE",
        data_cache,
        select_columns=selected_columns,
    )
    return price_data

def fetch_interconnector_data(start_time: datetime, end_time: datetime, selected_columns: list[str]) -> pd.DataFrame:
    """
    fetch data from NEMOSIS dispatch interconnector table.

    typical arg for selected_columns: 
        ["SETTLEMENTDATE", "INTERCONNECTORID", "INTERVENTION", "METEREDMWFLOW", "MWFLOW", "MWLOSSES"]
    """
    interconnector_data = n.dynamic_data_compiler(
        start_time.strftime("%Y/%m/%d %H:%M:%S"),
        end_time.strftime("%Y/%m/%d %H:%M:%S"),
        "DISPATCHINTERCONNECTORRES",
        data_cache,
        select_columns=selected_columns,
    )
    return interconnector_data
