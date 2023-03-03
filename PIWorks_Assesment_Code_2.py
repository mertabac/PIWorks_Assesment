import pandas as pd

df = pd.read_csv('country_vaccination_stats.csv')

df['daily_vaccinations'] = df.groupby('country')['daily_vaccinations'].apply(lambda
                                                             x: x.fillna(x.min()))
df['daily_vaccinations'].fillna(value=0, inplace=True)

#Task 2
df_new = df.groupby('country')['daily_vaccinations'].median().sort_values(ascending=False)

print(df_new.head(3))