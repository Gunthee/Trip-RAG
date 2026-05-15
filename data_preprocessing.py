import re
import pandas as pd


def clean_text(text: str) -> str:
    # Remove hashtags
    text = re.sub(r'#.*?#', ' ', text)

    # Remove unwanted sections
    text = re.sub(r'-', ' ', text)

    return text.strip()

df = pd.read_csv('tours_merged_cleaned2.csv')

df['description'] = df['description'].apply(clean_text)

#print(df['description'][100])