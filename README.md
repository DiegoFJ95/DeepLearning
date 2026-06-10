# Deep Learning Chatbot — Q&A con LSTM

Modelo de deep learning basado en LSTM para generación de respuestas conversacionales en inglés, entrenado con pares pregunta-respuesta mediante un enfoque de **Causal Language Modeling con tokens separadores y pérdida enmascarada**.

[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/diegofj95/chatbot-deeplearning)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

---

## Descripción del Problema

El objetivo de este proyecto es desarrollar un chatbot conversacional capaz de generar respuestas coherentes en inglés utilizando técnicas de Deep Learning. El modelo aprende la relación estructurada **pregunta → respuesta** de múltiples fuentes de datos conversacionales y puede mantener diálogos interactivos con usuarios.

### Evolución del Proyecto

El proyecto pasó por dos enfoques distintos de entrenamiento:

**Versión 1 — Language Model puro:** El texto de todas las fuentes se concatenaba en una sola cadena continua y el modelo aprendía a predecir el siguiente token dado los anteriores, sin distinción entre pregunta y respuesta. Esto resultó en un modelo que generaba texto plausible pero no aprendía la direccionalidad conversacional.

**Versión 2 (actual) — Entrenamiento Q&A con pérdida enmascarada:** Los datos se estructuran como pares `(pregunta, respuesta)`. Se usan tokens separadores especiales (`Q_TOKEN`, `A_TOKEN`) para delimitar cada parte, y la función de pérdida se enmascara para que el modelo solo aprenda de los tokens de respuesta, ignorando la parte de la pregunta durante el backpropagation.

### Desafíos Principales

- **Estructura conversacional:** Transformar fuentes de texto heterogéneas en pares pregunta-respuesta consistentes
- **Pérdida enmascarada:** Implementar un training loop personalizado para ignorar el loss en los tokens de pregunta y padding
- **Coherencia contextual:** Mantener el hilo de la conversación a través de múltiples intercambios
- **Limitaciones de recursos:** Optimizar el modelo para funcionar dentro de las restricciones de memoria de Kaggle (16 GB GPU)
- **Balance entre vocabulario y eficiencia:** Usar tokenización a nivel de subpalabra (50K tokens) mientras se mantiene un entrenamiento factible

---

## Arquitectura del Modelo

El modelo implementa una arquitectura de red neuronal recurrente con las siguientes capas:

```
Input (Sequence Length: 150 tokens)
    ↓
Embedding Layer (vocab_size: 50,260 → embedding_dim: 256)
    ↓
LSTM Layer (700 units, return_sequences=True, return_state=True)
    ↓
Dense Layer (50,260 units — vocabulary size)
    ↓
Output (Next token prediction — solo sobre tokens de respuesta)
```

### Especificaciones Técnicas

| Parámetro | Valor |
|---|---|
| Total de parámetros | ~50.8 M (~194 MB) |
| Tokenizador | Tiktoken GPT-2 (`r50k_base`) |
| Vocabulario base | 50,257 tokens |
| Tokens especiales añadidos | 3 (`Q_TOKEN`, `A_TOKEN`, `PAD_TOKEN`) |
| Vocab size final | 50,260 |
| Función de pérdida | Sparse Categorical Crossentropy enmascarada |
| Optimizador | Adam |
| Batch size | 16 |
| Sequence length | 150 tokens |
| Máx. tokens pregunta | 60 |
| Máx. tokens respuesta | 80 |

### Tokens Especiales

El modelo extiende el vocabulario de tiktoken con tres tokens adicionales:

```python
Q_TOKEN   = 50257  # Marca el inicio de una pregunta
A_TOKEN   = 50258  # Marca el inicio (y fin) de una respuesta
PAD_TOKEN = 50259  # Rellena secuencias más cortas que SEQUENCE_LENGTH
```

El formato de cada secuencia de entrenamiento es:

```
[Q_TOKEN] <tokens de la pregunta> [A_TOKEN] <tokens de la respuesta> [A_TOKEN]
```

El `A_TOKEN` al final actúa como token de parada durante la inferencia.

---

## Datasets

El modelo fue entrenado con una combinación de **6 datasets**, obteniendo un total de **~92,000 pares pregunta-respuesta**:

| Dataset | Fuente | Pares aprox. | Formato |
|---|---|---|---|
| 3K Conversations | Kaggle | 3,724 | CSV con columnas `question` / `answer` |
| Chatbot Training Dataset | Kaggle | ~7,000 | TXT separado por tabulación |
| Human Conversation Data | Kaggle | ~15,000 | TXT con prefijos `Human 1:` / `Human 2:` |
| South Park Dialogues | Kaggle | ~45,000 | CSV con columnas `season`, `episode`, `character`, `line` |
| Cornell Movie Dialogs Corpus | Kaggle | ~20,000 | TSV con líneas consecutivas por película |

### Estrategia de Extracción de Pares

Cada fuente requirió una estrategia de parseo diferente:

**South Park y Cornell Movie Dialogs:** Se emparejaron líneas consecutivas de personajes distintos dentro del mismo episodio o película. Esto garantiza que los pares sean intercambios conversacionales reales y no diálogos de escenas distintas.

**Chatbot dataset:** El archivo ya venía con el formato `pregunta\trespuesta` en cada línea, por lo que solo requirió separar por tabulación.

**Human chat:** Cada línea se usó tanto como pregunta (respecto a la anterior) y como respuesta (respecto a la siguiente), generando el máximo de pares posibles de la conversación continua.

**Conversation CSV:** Las columnas `question` y `answer` se mapearon directamente.

---

## Procesamiento de Datos

### Tokenización

Se utiliza **Tiktoken** con el tokenizador `r50k_base` de GPT-2, que implementa Byte Pair Encoding (BPE). Este método identifica las secuencias de caracteres más frecuentes y las trata como unidades indivisibles, representando palabras comunes con un solo token y palabras menos frecuentes con subpalabras.

### Construcción de Secuencias

```python
# Formato de cada secuencia de entrenamiento
q_ids  = [Q_TOKEN] + enc.encode(question)[:60]
a_ids  = [A_TOKEN] + enc.encode(answer)[:80] + [A_TOKEN]
combined = q_ids + a_ids  # máx. 150 tokens
```

Cada par se convierte en dos secuencias desplazadas un token (paradigma autoregresivo estándar):

- **Input `x`:** `combined[:-1]` — todos los tokens menos el último
- **Target `y`:** `combined[1:]` — todos los tokens menos el primero

### Máscara de Pérdida

La diferencia central respecto a un language model estándar es la máscara que acompaña a cada secuencia:

```
Máscara:  [0, 0, 0, ..., 1, 1, 1, ..., 0, 0]
           ←── pregunta ──→ ←─ respuesta ─→ ← pad →
```

El loss solo se calcula donde la máscara vale `1`, obligando al modelo a aprender exclusivamente a generar respuestas, no a predecir los tokens de la pregunta.

---

## Función de Pérdida Enmascarada

```python
def masked_loss(targets, logits, masks):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    per_token_loss = loss_fn(targets, logits)   # (batch, seq_len)
    masked = per_token_loss * masks             # cero en preguntas y padding
    return tf.reduce_sum(masked) / (tf.reduce_sum(masks) + 1e-8)
```

Al usar `reduction='none'` el loss se calcula individualmente por token, permitiendo aplicar la máscara antes de promediar. El denominador usa el conteo real de tokens activos (no el tamaño total del batch), lo cual da un loss correctamente normalizado independientemente del largo de las preguntas.

---

## Entrenamiento

El entrenamiento usa un loop personalizado en lugar de `model.fit()`, necesario para pasar la máscara como tercer argumento a cada paso:

```python
@tf.function
def train_step(inputs, targets, masks):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = masked_loss(targets, logits, masks)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

### Configuración de Entrenamiento

| Parámetro | Valor |
|---|---|
| Epochs planificadas | 20 |
| Epochs completadas | 18 |
| Tiempo por epoch | 15–16 minutos |
| GPU | Kaggle P100 |
| Optimizer | Adam (lr=0.001) |
| Loss epoch 18 | ~1.66 |

### Checkpoints

Al final de cada epoch se guardan tanto los pesos del modelo como el estado del optimizador, permitiendo reanudar el entrenamiento exactamente donde se detuvo:

```python
model.save_weights(f"./training_checkpoints/ckpt_{epoch+1}.weights.h5")
np.save(f"./training_checkpoints/optimizer_{epoch+1}.npy",
        [v.numpy() for v in optimizer.variables], allow_pickle=True)
```

---

## Generación de Respuestas

### Clase `OneStep`

```python
class OneStep(tf.keras.Model):
    def generate_one_step(self, input_ids, states=None):
        predicted_logits, states = self.model(input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]   # última posición
        predicted_logits = predicted_logits / self.temperature
        predicted_id = tf.random.categorical(predicted_logits, num_samples=1)
        return tf.squeeze(predicted_id, axis=-1), states
```

La generación es token a token. El estado del LSTM se pasa entre pasos para que el modelo mantenga el contexto sin reprocesar la secuencia completa desde el inicio.

### Inferencia Q&A

```python
def chat_response(model_step, question, max_new_tokens=100):
    prompt_ids = [Q_TOKEN] + enc.encode(question) + [A_TOKEN]
    input_ids = tf.constant([prompt_ids], dtype=tf.int32)
    states = None
    generated = []
    for _ in range(max_new_tokens):
        next_id, states = model_step.generate_one_step(input_ids, states)
        next_id = int(next_id.numpy()[0])
        if next_id in (A_TOKEN, PAD_TOKEN):
            break
        generated.append(next_id)
        input_ids = tf.constant([[next_id]], dtype=tf.int32)
    return enc.decode(generated)
```

El prompt se construye con el prefijo `[Q_TOKEN]` para indicar al modelo que viene una pregunta, y se cierra con `[A_TOKEN]` para que empiece a generar la respuesta. La generación se detiene al encontrar otro `A_TOKEN` (token de parada aprendido) o `PAD_TOKEN`.

---

## Cómo Usar

### Prerrequisitos

```bash
pip install tensorflow tiktoken numpy pandas tqdm
```

### Opción 1: Ejecutar en Kaggle (Recomendado)

1. Abre el notebook en Kaggle: [chatbot-deeplearning](https://www.kaggle.com/code/diegofj95/chatbot-deeplearning)
2. Habilita GPU (Settings → Accelerator → GPU P100)
3. **Para entrenar desde cero:** ejecuta todas las celdas excepto la de carga de pesos (`model.load_weights`)
4. **Para usar el modelo preentrenado:** ejecuta todas las celdas excepto el loop de entrenamiento, luego ejecuta la celda de carga de pesos
5. Usa el chat interactivo en la última celda

### Opción 2: Ejecutar Localmente

1. **Clona el repositorio**

```bash
git clone https://github.com/DiegoFJ95/DeepLearning.git
cd DeepLearning
```

2. **Descarga los pesos del modelo**

Descarga `ckpt_18.weights.h5` desde [Google Drive](https://drive.google.com/drive/folders/16d4coXSQw4PuMTC5NBHknPRGSUGYdc4v?usp=sharing) y colócalo en el directorio del proyecto.

3. **Abre el notebook y ejecuta en orden:**
   - Celdas 1–2: Instalación e importación de librerías
   - Celdas 3–4: Verificación de GPU
   - Celdas 7–18: Carga y construcción del dataset de pares Q&A
   - Celdas 15–17: Tokenización y construcción del dataset
   - Celdas 21–22: Definición e instanciación del modelo
   - Celda 30: Construcción del modelo y carga de pesos preentrenados
   - Celdas 32–33: Definición de `OneStep` y `chat_response`
   - Celda 34: Chat interactivo

### Interacción con el Chatbot

```
Welcome to the ChatBot!
Type "exit" to exit.

You: How are you doing today?
Bot: [response]

---

## Ajustes y Configuración Avanzada

### Temperatura de Generación

```python
one_step_model = OneStep(model, encode_text, decode_ids, temperature=0.8)
```

| Temperatura | Efecto |
|---|---|
| `< 1.0` | Respuestas más conservadoras y predecibles |
| `= 1.0` | Muestreo directo de la distribución (default) |
| `> 1.0` | Respuestas más creativas, mayor riesgo de incoherencia |

### Longitud Máxima de Respuesta

```python
chat_response(one_step_model, question, max_new_tokens=150)
```

### Reanudar Entrenamiento

```python
RESUME_FROM = 18

dummy = tf.zeros((1, SEQUENCE_LENGTH - 1), dtype=tf.int32)
model(dummy)
model.load_weights(f"./training_checkpoints/ckpt_{RESUME_FROM}.weights.h5")

for inputs, targets, masks in dataset.take(1):
    train_step(inputs, targets, masks)

optimizer_weights = np.load(
    f"./training_checkpoints/optimizer_{RESUME_FROM}.npy", allow_pickle=True)
for var, val in zip(optimizer.variables, optimizer_weights):
    var.assign(val)

for epoch in range(RESUME_FROM, EPOCHS):
    ...
```

---

## Resultados

### Métricas de Entrenamiento

| Epoch | Loss |
|---|---|
| 1 | ~4.72 |
| 9 | ~2.31 |
| 18 | ~1.66 |

La pérdida disminuyó consistentemente a lo largo de los 18 epochs completados, sin señales de divergencia. El modelo se encuentra en estado de **underfitting moderado**: la curva de aprendizaje aún muestra capacidad de mejora con más epochs o más datos.

### Ejemplo de output:

```
Welcome to the ChatBot!
Type "exit" to exit.

You:  hello

Bot: any a bit start obviously
________________________________________________________________________________

You:  um, whats your name?

Bot: Computers are like they are.
________________________________________________________________________________

You:  Oh how nice, do you like pizza?

Bot: Sure!
________________________________________________________________________________

You:  What is your favorite number?

Bot: It's a sci-fi book about different kinds of years. I'll see if they uh had a two Indian.
________________________________________________________________________________

You:  How old are you?

Bot: the point is to save his life, Butters. You're staying the video.
________________________________________________________________________________

You:  What happened to Cartman?

Bot: Everything is so bad, crap.
________________________________________________________________________________

You:  indeed

Bot: Hi
________________________________________________________________________________

You:  good bbye bot

Bot: Let's go out and find the train, baby.
```

### Comparación de Arquitecturas Exploradas

| Método | Vocab Size | Embedding | Timesteps | Tiempo/Epoch | Coherencia |
|---|---|---|---|---|---|
| Nivel de carácter | 114 | 512 | 2048 | ~14 seg | Baja — inventa palabras |
| Tiktoken (LM puro) | 50,257 | 256 | 100 | ~12 min | Media — texto fluido, sin estructura Q&A |
| Tiktoken + Q&A mask (actual) | 50,260 | 256 | 150 | ~15 min | Media-alta — respuestas dirigidas |

### Limitaciones Observadas

- **Coherencia temática:** El modelo tiende a derivar hacia temas genéricos después de pocos intercambios
- **Respuestas cortas:** Frecuentemente genera respuestas breves ("what do you mean?", "yes") antes de alcanzar el token de parada
- **Memoria de largo plazo:** Mantiene contexto inmediato pero olvida información de turnos anteriores
- **Nombres propios:** El tokenizador no está ajustado al dataset específico; algunos nombres propios no se representan bien
- **Entrenamiento incompleto:** 18 de 20 epochs planificados; el modelo aún tiene margen de mejora

---

## Estructura del Repositorio

```
DeepLearning/
│
├── chatbot-deeplearning.ipynb    # Notebook principal con todo el código
├── reporte.pdf                   # Documentación técnica completa
├── README_Chatbot.md             # Este archivo
└── (ckpt_18.weights.h5)         # Pesos del modelo (descargar aparte desde Google Drive)
```

---

## Trabajo Futuro

**A corto plazo:**
- Completar los 20 epochs planificados y evaluar si continuar más allá
- Implementar early stopping basado en una métrica de validación conversacional
- Ajustar el learning rate con cosine decay

**A mediano plazo:**
- Aumentar el dataset a más de 200,000 pares de alta calidad
- Experimentar con vocabularios más pequeños y ajustados al dominio (10K–20K tokens)
- Agregar una segunda capa LSTM para mayor capacidad de modelado

**A largo plazo:**
- Migrar a arquitectura Transformer (más eficiente para contextos largos)
- Implementar mecanismos de atención
- Fine-tuning desde un modelo preentrenado pequeño

---

## Autor

**Diego Isaac Fuentes Juvera** — A01705506

## Licencia

Este proyecto está bajo la Licencia MIT
