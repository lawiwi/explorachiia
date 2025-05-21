from flask import Flask, render_template, request, redirect, url_for,jsonify
from entrenamiento import generar_grafico_prediccion_semanal
from datetime import datetime
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import joblib
import json

app = Flask(__name__, template_folder='Templates')

# Home 

@app.route("/")
def chiaentre():
    return render_template('Chia/iniciachia.html')

@app.route('/prediccion/<empresa>', methods=['GET'])
def prediccion_empresa(empresa):
    ruta_imagen = generar_grafico_prediccion_semanal(empresa)
    if not ruta_imagen:
        return f"No se encontró el modelo para {empresa}", 404

    imagen_url = '/' + ruta_imagen.replace("\\", "/")
    return render_template('Prediccion/prediccion_empresa.html', empresa=empresa, imagen_url=imagen_url)

@app.route('/Chia/Acercade/EvaluacionModelo')
def evaluacion():
    return render_template('Machine/EvaluacionModelo.html')

@app.route('/evaluar_modelo', methods=['POST'])
def evaluar_modelo():
    empresa = request.form['empresa']
    return redirect(url_for('mostrar_metricas', empresa=empresa))

@app.route('/validacion_modelo', methods=['POST'])
def validacion_modelo():
    empresa = request.form['empresa']
    return redirect(url_for('validacion_cruzada', empresa=empresa))

@app.route('/metricas/<empresa>')
def mostrar_metricas(empresa):
    ruta_json = f'metricas_modelos/{empresa}.json'
    if os.path.exists(ruta_json):
        with open(ruta_json, 'r') as f:
            metricas = json.load(f)
    else:
        metricas = {"error": "No se encontraron métricas para esta empresa."}

    return render_template('Machine/metricasempresa.html', empresa=empresa, metricas=metricas)


@app.route('/validacion_cruzada/<empresa>')
def validacion_cruzada(empresa):
    ruta_json = f'metricas_modelos/validacion_cruzada_{empresa}.json'

    if not os.path.exists(ruta_json):
        return f"No se encontraron datos de validación cruzada para {empresa}", 404

    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    return render_template('Machine/validacioncruzada.html',
                           empresa=empresa,
                           rmse_folds=datos['RMSE_folds'],
                           rmse_promedio=datos['RMSE_promedio'])

@app.route("/registrar_click", methods=["POST"])
def registrar_click():
    data = request.get_json()
    empresa = data.get("empresa")
    

    if not empresa:
        return jsonify({"error": "No se proporcionó el nombre de la empresa"}), 400

    # Obtener fecha actual
    fecha_actual = datetime.now().date()

    # Ruta del archivo CSV para la empresa
    ruta_csv = f"data/{empresa}.csv"

    # Verificar si el archivo existe, si no, crearlo con encabezados
    archivo_nuevo = not os.path.exists(ruta_csv)
    with open(ruta_csv, mode="a", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo, quoting=csv.QUOTE_NONNUMERIC)
        if archivo_nuevo:
            writer.writerow(["publishedAtDate", "comentario"])  # encabezados mínimos
        writer.writerow([fecha_actual.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), "registro click"])

    return jsonify({"mensaje": f"Click registrado para {empresa} en {fecha_actual}"})


@app.route("/Chia/Acercade/EntendimientoDelNegocio")
def entendimiento():
    return render_template('Machine/EntendimientoDelNegocio.html')

@app.route("/Chia/Acercade/IngenieriaDatos")
def IngenieriaDatos():
    return render_template('Machine/IngenieriaDatos.html')

@app.route("/Chia/Acercade/IngenieriaModelo")
def IngenieriaModelo():
    return render_template('Machine/IngenieriaModelo.html')



@app.route("/Chia/Acercade")
def Acercade():
    return render_template('Machine/Acercade.html')

@app.route("/Chia/Empresas")
def Empresas():
    return render_template('Prediccion/einicio.html')