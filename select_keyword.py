import pandas as pd
import random

def select_keyword(csv_adr):
    df = pd.read_csv(csv_adr)
    index = random.randint(0, len(df))
    return {'CN':df['Translate'][index], 'EN':df['Prompt'][index]}
