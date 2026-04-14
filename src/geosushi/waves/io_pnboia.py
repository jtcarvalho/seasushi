import pandas as pd

PNBOIA_FILL_VALUE = -9999.0


def load_pnboia(file_path, variables=None, start_date=None, end_date=None,
                drop_flags=True, mask_fill=True):
    """
    Load PNBOIA buoy data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the PNBOIA CSV file. The first column is expected to be
        a parseable datetime index.
    variables : list of str, optional
        Column names to return (e.g. ['Hs', 'Tp', 'Dm']).
        If None, all columns are returned (subject to drop_flags).
    start_date : str or datetime-like, optional
        Inclusive start of the date range filter.
    end_date : str or datetime-like, optional
        Inclusive end of the date range filter.
    drop_flags : bool, optional
        If True (default), drop quality-flag columns (those starting with 'Flag_').
    mask_fill : bool, optional
        If True (default), replace the PNBOIA fill value (-9999) with NaN in
        all non-flag columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by time with the requested variables and date range.
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index.name = 'time'

    if mask_fill:
        data_cols = [c for c in df.columns if not c.startswith('Flag_')]
        df[data_cols] = df[data_cols].where(df[data_cols] != PNBOIA_FILL_VALUE)

    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]

    if drop_flags:
        flag_cols = [c for c in df.columns if c.startswith('Flag_')]
        df = df.drop(columns=flag_cols)

    if variables is not None:
        df = df[variables]

    return df
