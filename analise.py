import pandas as pd

df = pd.read_csv('/home/alexandre/Documentos/Alexandre/doutorado/AM/audit_data/trial.csv')

print('# Quantidade de instâncias por classe:')
print(df.groupby('Risk')['Risk'].count())

print('\n# Colunas vazias:')
print(df.isnull().sum())

print('\n# Quantidade de valores diferentes para a coluna LOCATION_ID:')
print(df.LOCATION_ID.unique().size)

