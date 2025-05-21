from flask import Flask, render_template, request, redirect, send_file, url_for
import base64
import os
from io import BytesIO
from PIL import Image
#from convertir_dmc2 import convert_image_to_dmc
from prueba_web_optimizada import convert_image_to_dmc

def convertir_numpy_a_base64(img_np):
    img_pil = Image.fromarray(img_np)
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return img_base64


app = Flask(__name__)

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
        numero = request.form.get('numero')
        
        if archivo and numero:
            numero = int(numero)
            nombre_archivo = archivo.filename
            ruta_imagen = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)
            archivo.save(ruta_imagen)

            nombre_salida = nombre_archivo.rsplit('.', 1)[0] + '.pdf'
            ruta_salida = os.path.join(app.config['UPLOAD_FOLDER'], nombre_salida)

            img_np=convert_image_to_dmc(ruta_imagen, ruta_salida, paleta,numero)

            
        img_base64 = convertir_numpy_a_base64(img_np)

        return render_template(
            "resultado.html",
            pdf_url=ruta_salida,
            imagen_base64=img_base64
        )

    

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
