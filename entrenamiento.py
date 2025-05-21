import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Esto previene errores con Tkinter en Flask
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import json

meses_alta = [6, 7, 12, 1]  # Junio, Julio, Diciembre, Enero
# Obtener todos los archivos CSV en la carpeta data/
data_folder = "data"
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

def es_temporada_alta(mes):
    return 1 if mes in meses_alta else 0

def evaluar_y_guardar_modelo(modelo, X_test, y_test, empresa):
    y_pred = modelo.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Guardar en JSON
    resultados = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }

    os.makedirs('metricas_modelos', exist_ok=True)
    with open(f'metricas_modelos/{empresa}.json', 'w') as f:
        json.dump(resultados, f)

def realizar_validacion_cruzada(modelo, X, y, empresa):
    scores = cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    
    print(f"Validación cruzada (RMSE) para {empresa}: {rmse_scores}")
    print(f"RMSE promedio para {empresa}: {round(np.mean(rmse_scores), 2)}")

    # Guardar los resultados de la validación cruzada
    resultados_cv = {
        "RMSE_folds": [round(s, 2) for s in rmse_scores],
        "RMSE_promedio": round(np.mean(rmse_scores), 2)
    }

    os.makedirs('metricas_modelos', exist_ok=True)
    with open(f'metricas_modelos/validacion_cruzada_{empresa}.json', 'w') as f:
        json.dump(resultados_cv, f)

# Recorrer cada archivo CSV
for file in csv_files:
    nombre_empresa = file.replace("data", "").replace(".csv", "")  # Extraer nombre
    print(f"\n--- Procesando: {nombre_empresa} ---")

    # Cargar datos
    df = pd.read_csv(os.path.join(data_folder, file))
    df["publishedAtDate"] = pd.to_datetime(df["publishedAtDate"])
    df["fecha"] = df["publishedAtDate"].dt.date
    df["dia_semana"] = df["publishedAtDate"].dt.day_name()
    df["dia_semana_num"] = df["publishedAtDate"].dt.weekday
    df["mes"] = df["publishedAtDate"].dt.month
    df["temporada_alta"] = df["mes"].apply(es_temporada_alta)
    df["visitas"] = np.where(df["publishAt"] == "registro click", 1, 5)
    df["visitas"] *= 50

    # Agrupar por día
    df_agrupado = df.groupby(["fecha", "dia_semana", "dia_semana_num", "mes", "temporada_alta"]).agg({"visitas": "sum"}).reset_index()

    # Entrenamiento
    X = df_agrupado[["dia_semana_num", "mes", "temporada_alta"]]
    y = df_agrupado["visitas"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    realizar_validacion_cruzada(modelo, X, y, nombre_empresa)

    # Evaluar
    y_pred = modelo.predict(X_test)

    # Guardar modelo
    ruta_modelo = os.path.join("modelos", f"modelo_{nombre_empresa}.pkl")
    joblib.dump(modelo, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")

    evaluar_y_guardar_modelo(modelo, X_test, y_test, nombre_empresa)
    print(f"Métricas guardadas para {nombre_empresa}")

def generar_grafico_prediccion_semanal(empresa):
    modelo_path = os.path.join('modelos', f'modelo_{empresa}.pkl')
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo para {empresa}")

    modelo = joblib.load(modelo_path)

    # Crear el DataFrame con características esperadas por el modelo
    dias = pd.DataFrame({
        'dia_semana_num': range(7),
        'mes': [6]*7,  # Puedes ajustar el mes si quieres mostrar un mes actual o promedio
        'temporada_alta': [1 if 6 in [6, 7, 12, 1] else 0]*7
    })

    predicciones = modelo.predict(dias)

    nombres_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    carpeta_salida = os.path.join('static', 'plots')
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_imagen = os.path.join(carpeta_salida, f'grafico_{empresa}.png')

    plt.figure(figsize=(10, 5))
    plt.bar(nombres_dias, predicciones, color='#4c98af')
    plt.title(f'Predicción de visitas por día - {empresa}')
    plt.ylabel('Visitas estimadas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

    return ruta_imagen

def evaluar_y_guardar_modelo(modelo, X_test, y_test, empresa):
    y_pred = modelo.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Guardar en JSON
    resultados = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }

    os.makedirs('metricas_modelos', exist_ok=True)
    with open(f'metricas_modelos/{empresa}.json', 'w') as f:
        json.dump(resultados, f)
