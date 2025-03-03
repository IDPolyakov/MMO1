import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv(r"D:\JC++\Python\Titanic\train.csv")

pd.set_option("display.max_columns", None)
print(df.head())

print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode()[0])

print(df.isnull().sum())

num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df.head())

df = pd.get_dummies(df, columns=["HomePlanet", "Name"], drop_first = True)

processed_file = "processed_titanic.csv"
df.to_csv(processed_file, index = False)
print(f"Обработанные данные сохранены в {processed_file}")
