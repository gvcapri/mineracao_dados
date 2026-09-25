import pandas as pd

df= pd.read_csv(r"C:\Users\guilh\OneDrive\Desktop\pessoal\codigos\python\train.csv")

caminho=r"C:\Users\guilh\OneDrive\Desktop\pessoal\codigos\python\train.csv"

dados=pd.read_csv(caminho)
dados.describe()
media_terreno= round(dados['LotArea'].mean())

ano_atual= pd.Timestamp.now().year

nova_idade_casa= round(ano_atual - dados['YearBuilt'].max())

print(f"A média do tamanho do terreno é: {media_terreno}")
print(f"A média da idade das casas é: {nova_idade_casa}")