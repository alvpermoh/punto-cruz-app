import cv2
import numpy as np
import sys
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from PIL import Image
import os
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO


# Símbolos "guays" para cada clave
simbolos = {
    '0': '■',  # cuadrado
    '1': '●',  # círculo negro
    '2': '▲',  # triángulo arriba
    '3': '▼',  # triángulo abajo
    '4': '◆',  # rombo
    '5': '★',  # estrella negra
    '6': '☀',  # sol
    '7': '☁',  # nube
    '8': '✿',  # flor
    '9': '☂',  # paraguas
    'A': '♞',  # caballo ajedrez
    'B': '♣',  # trébol
    'C': '☕',  # taza
    'D': '⚡',  # rayo
    'E': '♠',  # pica
    'F': '♤',  # pica blanca
    'G': '♧',  # trébol blanco
    'H': '♥',  # corazón
    'I': '♢',  # diamante blanco
    'J': '☽',  # luna
    'K': '♖',  # torre ajedrez
    'L': '♜',  # torre ajedrez negra
    'M': '♛',  # reina ajedrez
    'N': '♚'   # rey ajedrez
}

colores = {
    '0': '#000000',
    '1': '#FF6347',
    '2': '#1E90FF',
    '3': '#32CD32',
    '4': '#FFD700',
    '5': '#8A2BE2',
    '6': '#FFA500',
    '7': '#87CEEB',
    '8': '#FF69B4',
    '9': '#00CED1',
    'A': '#2E8B57',
    'B': '#6A5ACD',
    'C': '#FF4500',
    'D': '#FFD700',
    'E': '#808080',
    'F': '#A9A9A9',
    'G': '#00FA9A',
    'H': '#FF1493',
    'I': '#B0E0E6',
    'J': '#DAA520',
    'K': '#696969',
    'L': '#8B4513',
    'M': '#7B68EE',
    'N': '#4682B4'
}

DMC_RGB = {
    # Sección 1
    106: (106, 80, 70), 90: (85, 32, 14), 92: (122, 113, 98), 125: (167, 210, 120),
    67: (43, 57, 41), 93: (255, 141, 1), 121: (255, 255, 1), 52: (255, 215, 204),
    48: (255, 226, 226), 107: (255, 255, 255), 115: (255, 255, 255), 53: (255, 244, 220),

    # Sección 2
    701: (63, 143, 41), 3851: (73, 179, 161), 807: (100, 171, 186), 798: (70,106, 142),
    3731: (218,103,131), 326: (179,59,75), 349: (210,16,53), 947: (255, 123,77),
    741: (255, 163, 43), 973: (255, 227, 0), 712: (255, 251, 239), 310: (0, 0, 0),

    # Sección 3
    "B5200": (255, 255, 255), 23: (237, 226, 237), 605: (255,192,205), 967: (255,222,213),
    3770: (255,238,227), 3823: (255,253,227), 14: (208,251,178), 747: (229,252,253),
    964: (169,226,216), 800: (192,204,22), 26: (215,202,230), 1: (252, 252, 252),

    # Sección 4
    225: (255, 223, 213), 3354: (228,166,172), 3733: (232,135,155), 3687: (201,107,112),
    304: (183,31,51), 350: (224,72,72), 20: (247,175,147), 3825: (253,189,150),
    19: (247,201,95), 727: (255,241,175), 772: (228,236,212), 164: (200,216,184),
    988: (115,139,91), 471: (174,191,121), 520: (102,109,79), 501: (57, 111, 82),
    3840: (176,192,218), 3839: (123,142,171), 327: (99,54,102), 33: (156,89,158),
    915: (130, 0, 67)
}


DMC_PALETTES = {
    "matizados": {
        "codes": [106, 90, 92, 125, 67, 93, 121, 52, 48, 107, 115, 53],
        "colors": [DMC_RGB.get(c, (0,0,0)) for c in [106, 90, 92, 125, 67, 93, 121, 52, 48, 107, 115, 53]]
    },
    "pasion": {
        "codes": [701, 3851, 807, 798, 3731, 326, 349, 947, 741, 973, 712, 310],
        "colors": [DMC_RGB.get(c, (0,0,0)) for c in [701, 3851, 807, 798, 3731, 326, 349, 947, 741, 973, 712, 310]]
    },
    "pastel": {
        "codes": ["B5200", 23, 605, 967, 3770, 3823, 14, 747, 964, 800, 26, 1],
        "colors": [DMC_RGB.get(c, (0,0,0)) for c in [23, 605, 967, 3770, 3823, 14, 747, 964, 800, 26, 1]]
    },
    "amazon1": {
        "codes": ["B5200",23, 225, 3354, 3733, 3687, 304, 350, 20, 3825, 19, 727, 3823, 772, 164, 988, 471, 520, 501, 3840, 3839, 327, 33, 915],
        "colors": [DMC_RGB.get(c, (0,0,0)) for c in [23, 225, 3354, 3733, 3687, 304, 350, 20, 3825, 19, 727, 3823, 772, 164, 988, 471, 520, 501, 3840, 3839, 327, 33, 915]]
    }
    
}

SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def matriz_a_imagen(matriz, filename=None, offset_x=0, offset_y=0):
    filas, columnas = len(matriz), len(matriz[0])

    # Ajustar el tamaño de la figura según el tamaño de la matriz
    escala = 0.4  # Ajusta este valor para cambiar el tamaño de las celdas
    fig_width = columnas * escala
    fig_height = filas * escala
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax.axis('off')

    def es_color_oscuro(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        luminancia = 0.299 * r + 0.587 * g + 0.114 * b
        return luminancia < 140

    # Ajustar el tamaño del texto según el tamaño de las celdas
    fontsize = max(10, int(22 * escala))

    for y in range(filas):
        for x in range(columnas):
            pos_x_real = x + offset_x
            pos_y_real = y + offset_y
            valor = matriz[y][x]
            color = colores.get(valor, '#FFFFFF')
            simbolo = simbolos.get(valor, '?')

            rect = plt.Rectangle((x, filas - y - 1), 1, 1, facecolor=color)
            ax.add_patch(rect)

            color_texto = 'white' if es_color_oscuro(color) else 'black'

            ax.text(
                    x + 0.5, filas - y - 0.5, simbolo,
                    ha='center', va='center',
                    fontsize=fontsize, color=color_texto, fontweight='bold'
                )
            if y == 0 and (pos_x_real % 10 == 0):
                ax.text(
                    x + 0.5, filas - y +0.2 , str(pos_x_real),
                    ha='center', va='center',
                    fontsize=fontsize - 2, color='blue'
                )

            if x == 0 and (pos_y_real % 10 == 0):
                ax.text(
                    x - 0.5, filas - y - 0.5, str(pos_y_real),
                    ha='center', va='center',
                    fontsize=fontsize - 2, color='blue'
                )

    margen = 0.5
    ax.set_xlim(-margen, columnas + margen)
    ax.set_ylim(-margen, filas + margen)

    ax.set_aspect('equal')

    if filename:
        filepath = os.path.join("static", "bloques", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    
    fig.canvas.draw()  # Renderiza la figura en el backend
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_array


    #return filename

def leyenda_a_imagen(legend, conteo_simbolos, filename=None):
    n = len(legend)
    fig_height = n * 0.5 + 0.5
    fig_width = 6  # Ampliado para la columna extra
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    box_size = 0.3
    espacio_texto_cuadro = 0.1
    espacio_columna_num = 3.5  # posición X donde empieza la columna de número de puntos

    for i, (dmc_code, symbol) in enumerate(legend.items()):
        y = n - i - 1  # Para que empiece arriba

        texto = f"{simbolos[symbol]} DMC {dmc_code}"
        color_hex = colores.get(symbol, "#000000")

        # Dibujar texto
        ax.text(0, y, texto, va='center', ha='left', fontsize=12, color='black')

        # Dibujar rectángulo del color a la derecha del texto
        ax.add_patch(plt.Rectangle(
            (2 + espacio_texto_cuadro, y - box_size/2),
            box_size, box_size,
            facecolor=color_hex,
            edgecolor='none'
        ))

        # Dibujar número de puntos en la columna nueva
        cantidad = conteo_simbolos.get(symbol, 0)
        ax.text(espacio_columna_num, y, str(cantidad), va='center', ha='left', fontsize=12, color='black')

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()  # Para que el 0 esté arriba

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename

    fig.canvas.draw()  # Renderiza la figura en el backend
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_array

    
    
def closest_color(pixel, colors):
    colors = np.array(colors)
    distances = np.sqrt(np.sum((colors - pixel) ** 2, axis=1))
    index = np.argmin(distances)
    return colors[index]

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def create_legend(legend, output_path):
    legend_text = "Leyenda:\n"
    for code, symbol in legend.items():
        legend_text += f"{symbol}: DMC {code}\n"
    with open(output_path.replace(".pdf", "_legend.txt"), "w") as f:
        f.write(legend_text)

def resize_image(image_path, max_side_length):
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return None

    height, width = img.shape[:2]
    if height > width:
        new_height = max_side_length
        new_width = int((max_side_length / height) * width)
    else:
        new_width = max_side_length
        new_height = int((max_side_length / width) * height)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img

def divide_matrix(matrix, block_size=30):
    matrix_np = np.array(matrix)
    h, w = matrix_np.shape[:2]
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = matrix_np[y:y + block_size, x:x + block_size]
            blocks.append((block, x, y))
    return blocks

# Apply a function to each block and store the results in a vector
def apply_function_to_blocks(matrix, func, block_size=30):
    blocks = divide_matrix(matrix, block_size)
    results = []
    print(len(blocks))
    for i, (block, offset_x, offset_y) in enumerate(blocks):
        filename = None
        print(i,offset_x,offset_y)
        resultado= matriz_a_imagen(block, filename, offset_x=offset_x, offset_y=offset_y)

        results.append(resultado) 

    return results

# Función de conversión actualizada para incluir el redimensionado
def convert_image_to_dmc(image_path, output_path, palette_name, max_side_length):

    resized_img = resize_image(image_path, max_side_length)
    if resized_img is None:
        return

    if palette_name not in DMC_PALETTES:
        print(f"Paleta '{palette_name}' no encontrada.")
        return

    dmc_codes = DMC_PALETTES[palette_name]["codes"]
    dmc_colors = DMC_PALETTES[palette_name]["colors"]

    img = resized_img
    if img is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    legend = {}
    output_img = np.zeros_like(img_rgb)
    output_img2 = [["" for _ in range(w)] for _ in range(h)]
    conteo_simbolos = {}
    # Convertimos el bucle para vectorizar la búsqueda de colores
    for i in range(h):
        for j in range(w):
            closest = closest_color(img_rgb[i, j], dmc_colors)
            output_img[i, j] = closest
            index = np.where(np.all(dmc_colors == closest, axis=1))[0][0]
            symbol = SYMBOLS[index % len(SYMBOLS)]
            output_img2[i][j] = symbol
            if dmc_codes[index] not in legend:
                legend[dmc_codes[index]] = symbol
                color_hex = rgb_to_hex(dmc_colors[index])
                print(color_hex)
                colores[symbol] = color_hex

            # Contar apariciones del símbolo
            if symbol not in conteo_simbolos:
                conteo_simbolos[symbol] = 0
            conteo_simbolos[symbol] += 1




    print("Hecho 1")
    # Guardar imagen convertida
    converted_image_path = output_path.replace(".pdf", "_converted.jpg")
    #symbol_image_path = "simbolos.png"

    cv2.imwrite(converted_image_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(len(output_img2), len(output_img2[0]))
    symbol_images=apply_function_to_blocks(output_img2, matriz_a_imagen, block_size=50)
    print("Hecho 2")
    #matriz_a_imagen(output_img2, "resultado.png")
    img_final=matriz_a_imagen(output_img2, None)

    # Crear leyenda
    create_legend(legend, output_path)
    leyenda_imagen=leyenda_a_imagen(legend=legend,conteo_simbolos=conteo_simbolos, filename="leyenda.png")

    cell_size = 10  # o el tamaño que quieras
    grid = [list(legend.values())]  # Ejemplo simple, una fila con los símbolos usados

    # Generar PDF
    generate_pdf(output_path, grid, legend, cell_size, converted_image_path,symbol_images,leyenda_imagen)
    print("Hecho 3")
    return img_final


def generate_pdf(output_path, grid, legend, cell_size, image_path, imagenes,leyenda_imagen):
    c = canvas.Canvas(output_path, pagesize=A4)
    page_width, page_height = A4
    margin = 20

    # Primera página: Imagen convertida
    img = Image.open(image_path)
    width, height = img.size
    max_width = page_width - 2 * margin
    max_height = page_height - 2 * margin
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            width = max_width
            height = width / aspect_ratio
        else:
            height = max_height
            width = height * aspect_ratio

    x = (page_width - width) / 2
    y = (page_height - height) / 2
    c.drawImage(image_path, x, y, width, height)

    c.showPage()

    for img_array in imagenes:
        # Convertir array NumPy a imagen PIL
        img_pil = Image.fromarray(img_array.astype('uint8'))
        
        # Guardar en un buffer BytesIO en formato PNG
        img_buffer = BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)  # Volver al inicio del buffer
        
        # Crear ImageReader a partir del buffer
        img_reader = ImageReader(img_buffer)
        
        width, height = img_pil.size
        
        max_width = page_width - 2 * margin
        max_height = page_height - 2 * margin
        aspect_ratio = width / height
        
        # Ajustar tamaño manteniendo la relación de aspecto
        if width > max_width or height > max_height:
            if width / max_width > height / max_height:
                width = max_width
                height = width / aspect_ratio
            else:
                height = max_height
                width = height * aspect_ratio
        
        # Centrar la imagen
        x = (page_width - width) / 2
        y = (page_height - height) / 2
        
        # Dibujar la imagen en el canvas con ImageReader
        c.drawImage(img_reader, x, y, width, height)
        
        c.showPage()

    
    # Tercera página: Leyenda vertical
    img_pil = Image.fromarray(img_array.astype('uint8'))
        
    # Guardar en un buffer BytesIO en formato PNG
    img_buffer = BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)  # Volver al inicio del buffer
        
    # Crear ImageReader a partir del buffer
    img_reader = ImageReader(img_buffer)
    width, height = img_pil.size
    max_width = page_width - 2 * margin
    max_height = page_height - 2 * margin
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            width = max_width
            height = width / aspect_ratio
        else:
            height = max_height
            width = height * aspect_ratio

    x = (page_width - width) / 2
    y = (page_height - height) / 2 
    c.drawImage(img_reader, x, y, width, height)
    c.save()



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python script.py ruta_imagen_entrada ruta_pdf_salida paleta")
    else:
        _, input_path, output_path, palette,N = sys.argv
        convert_image_to_dmc(input_path, output_path, palette.lower(),N)


