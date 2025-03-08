import pandas as pd
import sklearn.preprocessing
#Обработка train.csv
def solve(path):
    df = pd.read_csv(path)
    pd.set_option("display.max_columns", None)
    print(df.head())
    print(df.isnull().sum())
    # print(len(df.columns))

    #Заполнение пропущенных значений
    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    print(df.isnull().sum())

    #print(df["Cabin"].nunique())

    #Нормализация числовых данных
    scaler = sklearn.preprocessing.MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(df.head())

    #Преобразование категориальных данных
    df = pd.get_dummies(df, columns = ["HomePlanet"], drop_first = True)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand = True)
    df['Deck'].fillna('Unknown')
    df['Side'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['Deck', 'Side'], drop_first = True)
    df.drop(columns = ['Cabin', 'Num'], inplace = True)
    print(df.head())
    print(len(df.columns))

    processed_file = "processed_" + path.split('\\')[len(path.split('\\')) - 1]
    df.to_csv(processed_file, index = False)
    print(f"Обработанные данные сохранены в {processed_file}")