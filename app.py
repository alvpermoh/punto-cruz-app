from flask import Flask, render_template, request, redirect, send_file, url_for

import os
from convertir_dmc2 import convert_image_to_dmc

app = Flask(__name__, template_folder='../templates')

#@app.route('/')
#def home():
#    return "Hola desde Gunicorn"

UPLOAD_FOLDER = 'static/resultados'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        archivo = request.files['imagen']
        paleta = request.form.get('paleta')
        
        if archivo:
            nombre_archivo = archivo.filename
            ruta_imagen = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)
            archivo.save(ruta_imagen)

            nombre_salida = nombre_archivo.rsplit('.', 1)[0] + '.pdf'
            ruta_salida = os.path.join(app.config['UPLOAD_FOLDER'], nombre_salida)

            convert_image_to_dmc(ruta_imagen, ruta_salida, paleta, 165)

            nombre_resultado = "resultado.png"
            return render_template(
                "resultado.html",
                pdf_url=ruta_salida,
                imagen_url=url_for('static', filename=f'bloques/{nombre_resultado}')
            )

    print("Buscando index.html en:", app.jinja_loader.searchpath)

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
