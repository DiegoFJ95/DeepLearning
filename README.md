#  Deep Learning Chatbot

Modelo de deep learning basado en LSTM para generación de texto conversacional en inglés. Entrenado con datos de Wikipedia, diálogos de South Park y conversaciones humano-chatbot.

[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/diegofj95/chatbot-deeplearning)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

## Descripción del Problema

El objetivo de este proyecto es desarrollar un chatbot conversacional capaz de generar respuestas coherentes en inglés utilizando técnicas de Deep Learning. El modelo aprende patrones conversacionales de múltiples fuentes de datos y puede mantener diálogos interactivos con usuarios.

### Desafíos Principales

- **Coherencia contextual**: Mantener el hilo de la conversación a través de múltiples intercambios
- **Diversidad de respuestas**: Evitar respuestas genéricas y repetitivas
- **Limitaciones de recursos**: Optimizar el modelo para funcionar dentro de las restricciones de memoria de Kaggle (16GB GPU)
- **Balance entre vocabulario y eficiencia**: Usar tokenización a nivel de subpalabra (50K tokens) mientras se mantiene un entrenamiento factible

## Arquitectura del Modelo

El modelo implementa una arquitectura de red neuronal recurrente con las siguientes capas:

```
Input (Sequence Length: 100 tokens)
    ↓
Embedding Layer (vocab_size: 50,257 → embedding_dim: 256)
    ↓
LSTM Layer (700 units, return_sequences=True, return_state=True)
    ↓
Dense Layer (50,257 units - vocabulary size)
    ↓
Output (Next token prediction)
```

### Especificaciones Técnicas

- **Total de parámetros**: 50,775,549 (~194 MB)
- **Tokenizador**: Tiktoken (GPT-2 tokenizer - r50k_base)
- **Función de pérdida**: Sparse Categorical Crossentropy
- **Optimizador**: Adam
- **Batch size**: 16
- **Sequence length**: 100 tokens

## Datasets

El modelo fue entrenado con una combinación de 5 datasets (585,415 caracteres totales):

1. **3K Conversations Dataset** - Conversaciones formales e informales
2. **Chatbot Training Dataset** - Patrones de pregunta-respuesta
3. **Human Conversation Data** - Diálogos entre humanos
4. **South Park Dialogues** (150K caracteres) - Conversación coloquial
5. **Wikipedia Sentences** (150K caracteres) - Estructura gramatical formal

##  Cómo Usar

### Prerrequisitos

```bash
pip install tensorflow tiktoken numpy pandas
```

### Opción 1: Ejecutar en Kaggle (Recomendado)

1. Abre el notebook en Kaggle: [chatbot-deeplearning](https://www.kaggle.com/code/diegofj95/chatbot-deeplearning)
2. Habilita GPU (Settings → Accelerator → GPU T4 x2)
3. Para llevar a cabo el entrenamiento:
    1. Ejecuta todas las celdas en orden a excepción de la celda 24 para cargar los pesos: model.load_weights 
4. Para usar el modelo pre-entrenado:
    1. Ejecutar todas las celdas a excepción de la celda 23 de entrenamiento: history = model.fit()
    2. Ejecutar la celda 24 para cargar los pesos: model.load_weights
5. Utilizar el chat interactivo de la penúltima celda.

### Opción 2: Ejecutar Localmente

1. **Clona el repositorio**
```bash
git clone https://github.com/DiegoFJ95/DeepLearning.git
cd DeepLearning
```

2. **Descarga los pesos del modelo**
   
   Descarga `ckpt_14.weights.h5` desde [Google Drive](https://drive.google.com/drive/folders/1s9gQTbxfnzUNeNSvIgYoQJ9N0U8TvsdL?usp=sharing) y colócalo en el directorio del proyecto.

3. **Abre el notebook**
```bash
jupyter notebook chatbot-deeplearning.ipynb
```

4. **Ejecuta las celdas en orden**:
   - Celda 1: Instalación de dependencias
   - Celdas 2-8: Carga de datos y construcción del dataset
   - Celdas 9-15: Definición y construcción del modelo
   - Celda 24: Carga de pesos preentrenados
   ```python
   model.load_weights('ckpt_14.weights.h5')
   ```
   - Celdas 26-27: Configuración de generación de texto
   - **Penúltima celda**: Chat interactivo

### Interacción con el Chatbot

Una vez ejecutada la última celda, verás:

```
Welcome to the ChatBot, it is trained on data from Wikipedia, South Park, 
and both human and robot conversations!
To exit just type "exit"

You: 
```

Escribe tu mensaje y presiona Enter. El modelo generará una respuesta. Para salir, escribe `exit`.

**Ejemplo de uso**:
```
You: Hello, how are you?
Bot: I'm doing well. How about yourself?

You: What do you like to do?
Bot: I like to play video games and watch movies.

You: exit
```

## Resultados

### Métricas de Entrenamiento

- **Loss final**: ~0.094 (epoch 14)
- **Tiempo por epoch**: ~12 minutos (GPU T4)
- **Convergencia**: Modelo en estado de underfitting (requiere más entrenamiento)

### Ejemplos de Generación

**Prompt**: "Hello, how are you?"

**Output**:
```
Hello, how are you? Really good, really good. it's a great price. 
i think i'll buy both of them. i bought you a pair of pants. 
thank you. i hope they fit.
```

### Limitaciones Observadas

- Pérdida ocasional de coherencia temática en conversaciones largas
- Tendencia a respuestas genéricas ("what do you mean?", "yes")
- Necesita más epochs de entrenamiento para mejor generalización

## Configuración Avanzada

### Ajustar Temperatura de Generación

En la clase `OneStep`, modifica el parámetro `temperature`:

```python
one_step_model = OneStep(model, encode_text, decode_ids, temperature=0.8)
```

- `temperature < 1.0`: Respuestas más conservadoras y predecibles
- `temperature = 1.0`: Muestreo directo de la distribución
- `temperature > 1.0`: Respuestas más creativas y arriesgadas

### Cambiar Longitud de Generación

En el bucle de generación:

```python
for _ in range(200):  # Cambiar 200 al número deseado de tokens
    next_id, states = one_step_model.generate_one_step(input_ids, states)
    # ...
```


## 📁 Estructura del Repositorio

```
DeepLearning/
│
├── chatbot-deeplearning.ipynb    # Notebook principal con todo el código
├── reporte.pdf                    # Documentación técnica completa
├── README.md                      # Este archivo
└── (ckpt_14.weights.h5)          # Pesos del modelo (descargar aparte)
```

### Comparación de Encodings

| Método | Vocab Size | Embedding Dim | Timesteps | Tiempo/Epoch |
|--------|------------|---------------|-----------|--------------|
| Carácter | 114 | 512 | 2048 | 14 seg |
| Tiktoken | 50,257 | 256 | 100 | 12 min |

##  Autor

**Diego Isaac Fuentes Juvera** - A01705506

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
