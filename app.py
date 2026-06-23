import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Evita bloqueos de CORS con tu puerto de Angular (4200)
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. ORDEN ALFABÉTICO EXACTO DEL DATASET ENTRENADO (12 Clases)
CLASS_NAMES = [
    "0 minutos",
    "105 minutos",
    "120 minutos",
    "135 minutos",
    "15 minutos",
    "150 minutos",
    "165 minutos",
    "30 minutos",
    "45 minutos",
    "60 minutos",
    "75 minutos",
    "90 minutos"
]

# Forzamos CPU para que no busque CUDA de Nvidia de manera nativa en Windows
device = torch.device("cpu")
torch.set_num_threads(1) # Límite de hilos para evitar que Render se quede sin memoria RAM

def load_model():
    # Inicializa la estructura limpia de EfficientNet-B0
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
    
    # REGLA APLICADA: Nombre estandarizado solicitado por Arturo
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_meatscan_final.pth")
    
    # Carga de pesos matemáticos ignorando alertas de entornos antiguos
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval() # Congela las capas en modo evaluación
    return model

try:
    modelo_pytorch = load_model()
    print("\n=======================================================")
    print("=== MODELO EFFICIENTNET (.PTH) INTEGRADO CON ÉXITO ===")
    print("=======================================================\n")
except Exception as e:
    print(f"\n--- ERROR AL INICIALIZAR EL ARCHIVO .PTH: {e} ---\n")
    modelo_pytorch = None

# 2. TRANSFORMACIONES DE MATRIZ EXIGIDAS POR EFFICIENTNET
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    # Corrige la orientación automática de la cámara si la foto viene volteada
    image = ImageOps.exif_transpose(image).convert("RGB")
    return TRANSFORM(image).unsqueeze(0).to(device)

@app.route('/', methods=['GET'])
def index():
    return {"status": "online", "engine": "PyTorch EfficientNet-B0"}, 200

@app.route('/predict', methods=['POST'])
def predict():
    if modelo_pytorch is None:
        return jsonify({"tiempo": "ERROR", "confianza": 0, "error": "Modelo .pth ausente"}), 500

    if 'file' not in request.files:
        return jsonify({"tiempo": "ERROR", "confianza": 0, "error": "Falta el parámetro de imagen"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"tiempo": "ERROR", "confianza": 0, "error": "Archivo no seleccionado"}), 400

    try:
        # Flujo de lectura asíncrona de la imagen recibida de Angular
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        
        # Preprocesamiento idéntico al ambiente de entrenamiento
        inputs = preprocess_image(image)

        # Inferencia matemática pura desactivando gradientes
        with torch.no_grad():
            outputs = modelo_pytorch(inputs)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_index = torch.max(probabilities, dim=0)

        # Captura el label mapeado
        predicted_class = CLASS_NAMES[predicted_index.item()]
        
        # Mapeo limpio para tu frontend: Convierte "15 minutos" a "15 MIN" para que tu Angular lo entienda
        tiempo_formateado = predicted_class.replace(" minutos", " min").upper()
        confianza_porcentaje = confidence.item() * 100

        # Mantenemos tus logs intactos en consola para auditoría visual
        print("\n--- NUEVA PREDICCIÓN PYTORCH (EfficientNet) ---")
        print(f"Resultado final mapeado: {tiempo_formateado}")
        print(f"Confianza del disparo: {confianza_porcentaje:.2f}%")
        print("-----------------------------------------------\n")

        # Retorno de variables nativas inalteradas para no romper la interfaz
        return jsonify({
            "tiempo": tiempo_formateado,
            "confianza": confianza_porcentaje
        }), 200

    except Exception as e:
        print(f"Fallo en hilo de predicción: {e}")
        return jsonify({"tiempo": "ERROR", "confianza": 0, "error": str(e)}), 500

if __name__ == '__main__':
    # Lanzamos el backend limpio sin recargas forzadas de Debug
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)