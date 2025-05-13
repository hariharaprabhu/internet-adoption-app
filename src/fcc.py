import pandas as pd
def clean_fcc_data(fcc_df):
    fcc_df = fcc_df[(fcc_df['biz_res'] == 'R') & (fcc_df['technology'] == 'Any Technology')].copy()
    fcc_df = fcc_df.rename(columns={
        'geography_desc_full': 'NAME',
        'speed_10_1': 'fcc_10mbps_coverage',
        'speed_25_3': 'fcc_25mbps_coverage',
        'speed_100_20': 'fcc_100mbps_coverage',
        'speed_250_25': 'fcc_250mbps_coverage',
        'speed_1000_100': 'fcc_gigabit_coverage'
    })
    fcc_df['NAME'] = (
        fcc_df['NAME']
        .str.replace(', TX', ', Texas', regex=False)
        .str.replace(r'\b(city|cdp|town)\b', '', case=False, regex=True)
        .str.replace(r'\s+,', ',', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.lower()
    )
    return fcc_df.drop_duplicates(subset=['NAME'])