import pandas as pd
import random
import os


def select_keyword(name):
    abs_path = os.path.dirname(os.path.realpath(__file__)) + "/csv/"
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)
    csv_name = abs_path + f'{name}.csv'
    df = pd.read_csv(csv_name)
    index = random.randint(0, len(df))
    return {'CN': df['Translate'][index], 'EN': df['Prompt'][index]}


if __name__ == "__main__":
    print(select_keyword('color'))