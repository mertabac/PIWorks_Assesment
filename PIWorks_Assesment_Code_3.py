import pandas as pd

df = pd.read_csv('country_vaccination_stats.csv')

df['daily_vaccinations'] = df.groupby('country')['daily_vaccinations'].apply(lambda
                                                             x: x.fillna(x.min()))

#Task 3
date = '1/6/2021'
sum = df.loc[df['date'] == date, 'daily_vaccinations'].sum()