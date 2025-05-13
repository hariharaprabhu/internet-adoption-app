import json
import pandas as pd
import requests
def get_acs_data(year, api_key, variables):
    endpoint = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": ",".join(["NAME"] + variables),
        "for": "zip code tabulation area:*",
        "key": api_key
    }
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        for col in variables:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        raise Exception(f"Census API error: {response.status_code}")

def prepare_acs_features(df, year):
    df["internet_adoption_rate"] = df["B28002_007E"] / df["B28002_001E"]
    df["senior_population"] = df[[
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E"
    ]].sum(axis=1)
    df["under18_population"] = df[[
        "B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E",
        "B01001_027E", "B01001_028E", "B01001_029E", "B01001_030E"
    ]].sum(axis=1)
    df["pct_seniors"] = df["senior_population"] / df["B01001_001E"]
    df["pct_under18"] = df["under18_population"] / df["B01001_001E"]
    df["poverty_rate"] = df["B17001_002E"] / df["B17001_001E"]
    df["pct_white"] = df["B02001_002E"] / df["B01001_001E"]
    df["pct_black"] = df["B02001_003E"] / df["B01001_001E"]
    df["pct_hispanic"] = df["B03003_003E"] / df["B01001_001E"]
    df["pct_bachelor_plus"] = (
        df["B15003_022E"] + df["B15003_023E"] + df["B15003_025E"]
    ) / df["B01001_001E"]
    df["acs_year"] = year
    df["avg_household_size"] = df["B25010_001E"]
    df.rename(columns={"zip code tabulation area": "zip"}, inplace=True)
    selected_columns = [
        "zip", "internet_adoption_rate", "B19013_001E", "B01001_001E",
        "pct_seniors", "pct_under18", "poverty_rate",
        "pct_white", "pct_black", "pct_hispanic", "pct_bachelor_plus", "acs_year", "avg_household_size"
    ]
    return df[selected_columns].rename(columns={"B19013_001E": "median_income", "B01001_001E": "total_population"})
