import requests
import pandas as pd
def get_hud_zip_place_mapping(state_abbr, hud_api_key):
    url = f"https://www.huduser.gov/hudapi/public/usps?type=3&query={state_abbr}"
    headers = {"Authorization": f"Bearer {hud_api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json()["data"]["results"])
    else:
        raise Exception(f"HUD API error: {response.status_code}")