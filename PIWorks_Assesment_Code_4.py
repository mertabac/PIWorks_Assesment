import pandas as pd

#Creating example dataframe
df = pd.DataFrame({'url': ['<url>https://xcd32112.smart_meter.com</url>', '<url>https://abcd1234.smart_meter.com</url>', '<url>https://efgh5678.smart_meter.com</url>']})

# Task
df['output'] = df['url'].str.extract(r'<url>https://(.*?)</url>')

print(df)
