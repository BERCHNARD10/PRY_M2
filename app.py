from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo
modelo_rf = joblib.load('model2.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def realizar_prediccion():
    try:
        # Obtener los datos del POST request
        global_radiation = float(request.form['global_radiation'])
        cloud_cover = float(request.form['cloud_cover'])
        mean_temp = float(request.form['mean_temp'])
        max_temp = float(request.form['max_temp'])
        pressure = float(request.form['pressure'])
        min_temp = float(request.form['min_temp'])
    
        
        # Crear un DataFrame con los datos recibidos
        nuevo_registro_df = pd.DataFrame([[global_radiation, cloud_cover, mean_temp, max_temp, pressure, min_temp]], columns=['global_radiation', 'cloud_cover', 'mean_temp', 'max_temp', 'pressure', 'min_temp'])
        app.logger.debug(f'DataFrame creado: {nuevo_registro_df}')
        # Realizar la predicción
        prediccion = modelo_rf.predict(nuevo_registro_df)
        
        app.logger.debug(f'Predicción: {prediccion[0]}') #prediccion
        prediccion_value = prediccion[0] if isinstance(prediccion[0], (float, int)) else prediccion[0].tolist()

        return jsonify({'Predicción': prediccion_value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.debug = True
    app.run()
