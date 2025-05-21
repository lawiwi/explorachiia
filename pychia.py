import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------Cargar y procesar datos--------------------------------------------------------------------------------------------#
# Cargar CSV
df = pd.read_csv("data/fontanar.csv")

# Convertir la fecha ISO a datetime
df["publishedAtDate"] = pd.to_datetime(df["publishedAtDate"])

# Extraer fecha y día de la semana
df["fecha"] = df["publishedAtDate"].dt.date
df["dia_semana"] = df["publishedAtDate"].dt.day_name()
df["dia_semana_num"] = df["publishedAtDate"].dt.weekday

df["visitas"] = 1

# Agrupar visitas por día
df_agrupado = df.groupby(["fecha", "dia_semana", "dia_semana_num"]).agg({"visitas": "sum"}).reset_index()

print(df_agrupado.head())  # Verificar estructura final
print(df.head())  # Verificar estructura final

# ---------Entrenar modelo--------------------------------------------------------------------------------------------#

# Variables de entrada (X) y variable objetivo (y)
X = df_agrupado[["dia_semana_num"]]  # Solo día de la semana como número
y = df_agrupado["visitas"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# ---------Evaluar modelo--------------------------------------------------------------------------------------------#
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Error absoluto medio (MAE):", mae)

# ---------Ejemplo de predicción--------------------------------------------------------------------------------------------#
dias_semana = {
    0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
    4: "Viernes", 5: "Sábado", 6: "Domingo"
}

for i in range(7):
    prediccion = modelo.predict(pd.DataFrame({"dia_semana_num": [i]}))[0]
    print(f"Predicción de visitas para {dias_semana[i]}: {prediccion:.2f}")