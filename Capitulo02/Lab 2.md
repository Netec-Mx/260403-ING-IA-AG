# Crear un Agente de QA con LangChain usando Base de Conocimiento Local y API Pública

## Metadatos

| Propiedad | Valor |
|-----------|-------|
| **Duración** | 75 minutos |
| **Complejidad** | Intermedio |
| **Nivel Bloom** | Aplicar |
| **Módulo** | Módulo 2 – Agentes y Herramientas en LangChain |

## Descripción General

En este laboratorio construirás un agente de Question Answering (QA) completo utilizando LangChain como orquestador central. A través de cuatro fases progresivas, explorarás la diferencia práctica entre cadenas y agentes, indexarás documentos locales con LlamaIndex para crear una base de conocimiento RAG, definirás múltiples herramientas especializadas y ensamblarás un agente ReAct capaz de seleccionar automáticamente la herramienta correcta para responder preguntas variadas.

Este laboratorio refleja un caso de uso real en empresas que necesitan asistentes inteligentes capaces de consultar documentación interna, realizar cálculos y buscar información externa de forma autónoma, sin requerir flujos rígidos predefinidos.

## Objetivos de Aprendizaje

Al completar este laboratorio, serás capaz de:

- [ ] Distinguir mediante código ejecutable la diferencia entre una cadena LangChain (LCEL) y un agente con herramientas, analizando el flujo de decisión de cada uno
- [ ] Construir un índice vectorial RAG con LlamaIndex sobre documentos locales (TXT) y exponerlo como herramienta consultable por el agente
- [ ] Integrar al menos tres herramientas en el agente: búsqueda RAG en base de conocimiento local, calculadora matemática y consulta a Wikipedia
- [ ] Implementar un agente tipo ReAct con LangChain que seleccione automáticamente la herramienta correcta según la naturaleza de cada pregunta
- [ ] Evaluar el comportamiento del agente con preguntas variadas, interpretando el razonamiento paso a paso (chain-of-thought) en los logs de ejecución

## Prerrequisitos

### Conocimientos Requeridos

- Python 3.10+ (funciones, decoradores, manejo de excepciones, entornos virtuales)
- Comprensión básica de LLMs y cómo se invocan mediante API (prompt → completion)
- Conocimiento elemental de embeddings vectoriales y búsqueda por similitud semántica
- Experiencia consumiendo APIs REST con `requests` y manejo de respuestas JSON
- Haber completado el Módulo 1 del curso o tener conocimiento equivalente sobre fundamentos de LangChain

### Accesos Requeridos

- API key de OpenAI activa con saldo disponible (estimado: $0.50–$1.00 USD para este laboratorio)
- Conexión a Internet estable (mínimo 10 Mbps) para descargar paquetes y consultar APIs
- Cuenta de OpenAI con límite de gasto configurado (recomendado: $5 USD máximo)

> ⚠️ **Nota de Costos:** Este laboratorio usa GPT-3.5-turbo o GPT-4o-mini para minimizar costos. Configura un límite de gasto en [platform.openai.com/account/limits](https://platform.openai.com/account/limits) antes de comenzar.

## Entorno de Laboratorio

### Requisitos de Hardware

| Componente | Especificación |
|------------|----------------|
| Procesador | CPU 64-bit, mínimo 4 núcleos |
| Memoria RAM | Mínimo 8 GB (recomendado 16 GB) |
| Almacenamiento | Mínimo 2 GB libres para dependencias e índices |
| Conexión a Internet | Mínimo 10 Mbps |

### Requisitos de Software

| Software | Versión | Propósito |
|----------|---------|-----------|
| Python | 3.10 o 3.11 | Lenguaje principal del laboratorio |
| pip | 23.x o superior | Gestión de dependencias |
| LangChain | 0.2.x o superior | Framework de agentes y cadenas |
| LangChain Community | 0.2.x o superior | Herramientas adicionales e integraciones |
| LangChain OpenAI | 0.1.x o superior | Conector oficial con OpenAI |
| LlamaIndex | 0.10.x o superior | Indexación RAG y búsqueda semántica |
| FAISS (faiss-cpu) | 1.7.x | Motor de búsqueda vectorial local |
| OpenAI Python SDK | 1.x o superior | Cliente para modelos GPT |
| python-dotenv | 1.0.x | Gestión de variables de entorno |
| wikipedia | 1.4.x | Wrapper Python para API de Wikipedia |
| Jupyter Notebook | 7.x | Entorno interactivo de desarrollo |

### Configuración Inicial

Desde tu máquina virtual ingresa a Visual Studio Code. Abre la terminal integrada con la combinación de teclas (Ctrl+Ñ) o selecciona en el menú superior **Terminal** y luego haz clic en **New Terminal**.

En la terminal ejecuta los siguientes comandos:

```powershell
cd ~/Documents
```

```powershell
# 1. Crear directorio del laboratorio
mkdir lab_02_00_01
cd lab_02_00_01

# 2. Instala la versión de Python compatible con llama-index
winget install --id Python.Python.3.11 -e

# 3. Crear entorno virtual dedicado
py -3.11 -m venv venv_lab02

# 4. Activar el entorno virtual
.\venv_lab02\Scripts\Activate.ps1

# 5. Actualizar pip
python -m pip install --upgrade pip

# 6. Instalar todas las dependencias
pip install langchain langchain-classic langchain-community langchain-openai langchain-core llama-index llama-index-embeddings-openai llama-index-vector-stores-faiss faiss-cpu openai python-dotenv wikipedia langchainhub jupyter

# 7. Verificar instalación
python -c "import langchain, llama_index, faiss, wikipedia; print('✅ Dependencias OK')"

# 8. Crear estructura de carpetas del laboratorio
New-Item -ItemType Directory -Path documentos, notebooks -Force

# 9. Crear archivo .gitignore
New-Item -Path .env -ItemType File
New-Item -Path .gitignore -ItemType File
```
Selecciona **File** en el menú superior, luego **Open Folder...**, busca **Documents**, luego selecciona la carpeta **lab_02_00_01** y haz clic en el botón **Select Folder**. Abre el archivo `.gitignore` para proteger tus credenciales. Este paso es **obligatorio** antes de escribir cualquier API key. Escribe el siguiente contenido:

```text
.env
venv_lab02/
__pycache__/
*.pyc
*.pyo
indice_faiss/
.ipynb_checkpoints/
```

3. Abre el archivo `.env` y agrega tu API key de OpenAI. **Consulta a tu instructor por la API Key y el modelo**

```
AZURE_OPENAI_API_KEY=xxxxxxxx
AZURE_OPENAI_ENDPOINT=https://tu-recurso.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## Instrucciones Paso a Paso

### Paso 1: Crear los Documentos de la Base de Conocimiento

1. Crea los documentos de conocimiento ejecutando el siguiente script Python desde la terminal (con el entorno virtual activado):

```powershell
@'
import os

documentos = {
    "ia_fundamentos.txt": """
Inteligencia Artificial: Fundamentos

La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por parte de sistemas informáticos. Estos procesos incluyen el aprendizaje (la adquisición de información y reglas para usar la información), el razonamiento (usar las reglas para llegar a conclusiones aproximadas o definitivas) y la autocorrección.

Historia de la IA:
El término "inteligencia artificial" fue acuñado por John McCarthy en 1956 durante la Conferencia de Dartmouth. Sin embargo, los fundamentos matemáticos fueron establecidos por Alan Turing en 1950 con su famoso artículo "Computing Machinery and Intelligence", donde propuso el Test de Turing.

En la década de 1980, las redes neuronales artificiales ganaron popularidad. En 2012, AlexNet demostró el poder del deep learning al ganar el concurso ImageNet con una ventaja significativa. Desde entonces, el campo ha experimentado un crecimiento exponencial.

Tipos de IA:
1. IA Estrecha (ANI): Diseñada para realizar una tarea específica. Ejemplos: reconocimiento facial, recomendaciones de Netflix, asistentes de voz.
2. IA General (AGI): Hipotética IA con capacidad cognitiva equivalente a la humana en cualquier dominio intelectual.
3. Superinteligencia (ASI): IA hipotética que supera la inteligencia humana en todos los aspectos.

Aplicaciones actuales:
- Procesamiento de lenguaje natural (NLP)
- Visión por computadora
- Sistemas de recomendación
- Diagnóstico médico asistido
- Vehículos autónomos
- Generación de contenido creativo
""",

    "machine_learning.txt": """
Machine Learning: Conceptos Esenciales

El aprendizaje automático (Machine Learning o ML) es una rama de la IA que permite a los sistemas aprender y mejorar automáticamente a partir de la experiencia sin ser programados explícitamente. Se enfoca en desarrollar programas que puedan acceder a datos y usarlos para aprender por sí mismos.

Tipos de Aprendizaje:

1. Aprendizaje Supervisado:
El algoritmo aprende de datos etiquetados. Para cada entrada, existe una salida esperada conocida.
Ejemplos: clasificación de correos spam, predicción de precios de casas, diagnóstico de enfermedades.
Algoritmos comunes: Regresión Lineal, Árboles de Decisión, Random Forest, SVM, Redes Neuronales.

2. Aprendizaje No Supervisado:
El algoritmo encuentra patrones en datos sin etiquetas.
Ejemplos: segmentación de clientes, detección de anomalías, compresión de datos.
Algoritmos comunes: K-Means, DBSCAN, PCA, Autoencoders.

3. Aprendizaje por Refuerzo:
Un agente aprende tomando acciones en un entorno para maximizar una recompensa acumulada.
Ejemplos: AlphaGo, control de robots, optimización de estrategias de trading.

Conceptos Clave:
- Overfitting: El modelo memoriza los datos de entrenamiento pero no generaliza bien.
- Underfitting: El modelo es demasiado simple para capturar los patrones en los datos.
- Validación cruzada: Técnica para evaluar la capacidad de generalización del modelo.
- Hiperparámetros: Configuraciones del modelo que se establecen antes del entrenamiento.

Métricas de Evaluación:
- Para clasificación: Precisión, Recall, F1-Score, AUC-ROC
- Para regresión: MAE, MSE, RMSE, R²
""",

    "deep_learning.txt": """
Deep Learning: Redes Neuronales Profundas

El aprendizaje profundo (Deep Learning) es un subconjunto del machine learning que utiliza redes neuronales artificiales con múltiples capas (de ahí "profundo") para aprender representaciones de datos con múltiples niveles de abstracción.

Arquitecturas Principales:

1. Redes Neuronales Convolucionales (CNN):
Especializadas en procesamiento de imágenes y video.
Componentes: capas convolucionales, pooling, capas completamente conectadas.
Aplicaciones: reconocimiento de objetos, detección facial, diagnóstico médico por imagen.
Modelos famosos: AlexNet (2012), VGG, ResNet, EfficientNet.

2. Redes Neuronales Recurrentes (RNN) y LSTM:
Diseñadas para datos secuenciales como texto y series temporales.
Las LSTM (Long Short-Term Memory) resuelven el problema del gradiente desvaneciente.
Aplicaciones: traducción automática, generación de texto, predicción de series temporales.

3. Transformers:
Arquitectura revolucionaria introducida en 2017 en el paper "Attention is All You Need".
Mecanismo de atención: permite al modelo enfocarse en partes relevantes de la entrada.
Base de modelos modernos: BERT, GPT, T5, LLaMA.
Aplicaciones: NLP, generación de imágenes, audio, código.

4. Redes Generativas Adversariales (GAN):
Dos redes compiten: un generador crea datos sintéticos y un discriminador los evalúa.
Aplicaciones: generación de imágenes realistas, transferencia de estilo, aumento de datos.

Frameworks populares:
- TensorFlow / Keras (Google)
- PyTorch (Meta)
- JAX (Google Research)

Hardware para Deep Learning:
Las GPU (Unidades de Procesamiento Gráfico) son esenciales para entrenar redes profundas.
NVIDIA domina el mercado con sus GPU CUDA-compatibles y la plataforma cuDNN.
TPU (Tensor Processing Units) de Google ofrecen aceleración especializada para TensorFlow.
""",

    "nlp_transformers.txt": """
Procesamiento de Lenguaje Natural y Modelos de Lenguaje

El Procesamiento de Lenguaje Natural (NLP) es la rama de la IA que permite a las computadoras entender, interpretar y generar lenguaje humano. Es uno de los campos más activos de la IA moderna.

Evolución del NLP:
- Años 50-80: Sistemas basados en reglas y gramáticas formales.
- Años 90-2000: Modelos estadísticos (n-gramas, HMM, CRF).
- 2013: Word2Vec introduce embeddings de palabras densos.
- 2017: Transformers revolucionan el campo con el mecanismo de atención.
- 2018: BERT (Google) establece nuevos récords en benchmarks de comprensión.
- 2020: GPT-3 (OpenAI) demuestra capacidades emergentes con 175 billones de parámetros.
- 2022: ChatGPT democratiza el acceso a LLMs conversacionales.
- 2023-2024: GPT-4, Claude, Gemini y LLaMA2/3 elevan el estándar.

Modelos de Lenguaje Grande (LLM):
Los LLM son modelos de lenguaje entrenados en enormes corpus de texto con miles de millones de parámetros. Exhiben capacidades emergentes que no estaban presentes en modelos más pequeños.

Capacidades de los LLM modernos:
- Comprensión y generación de texto en múltiples idiomas
- Razonamiento matemático y lógico
- Escritura de código en múltiples lenguajes de programación
- Resumen y síntesis de documentos largos
- Traducción y análisis de sentimientos
- Respuesta a preguntas basadas en contexto

Técnicas de Optimización:
- Fine-tuning: Ajuste de parámetros del modelo para una tarea específica.
- RLHF (Reinforcement Learning from Human Feedback): Alineación con preferencias humanas.
- Prompt Engineering: Diseño de instrucciones efectivas para guiar al modelo.
- RAG (Retrieval-Augmented Generation): Combinación de recuperación de información con generación.

Desafíos del NLP:
- Ambigüedad lingüística y comprensión del contexto
- Sesgos en datos de entrenamiento
- Alucinaciones: generación de información incorrecta con aparente confianza
- Consumo computacional y huella de carbono
""",

    "etica_ia.txt": """
Ética en Inteligencia Artificial

La ética en IA se refiere al conjunto de principios, valores y normas que guían el desarrollo, despliegue y uso responsable de sistemas de inteligencia artificial. Es un campo interdisciplinario que involucra filosofía, derecho, sociología y tecnología.

Principios Fundamentales de la IA Ética:

1. Transparencia:
Los sistemas de IA deben ser explicables y comprensibles. Los usuarios deben poder entender cómo y por qué el sistema tomó una decisión particular. Esto es especialmente crítico en aplicaciones de alto impacto como medicina, justicia o crédito financiero.

2. Equidad y No Discriminación:
Los sistemas de IA no deben perpetuar o amplificar sesgos existentes en los datos de entrenamiento. Deben ser evaluados para garantizar que no discriminen por raza, género, edad, origen étnico u otras características protegidas.

3. Privacidad y Protección de Datos:
Los sistemas de IA que procesan datos personales deben cumplir con regulaciones como el GDPR (Europa) o la CCPA (California). El principio de minimización de datos establece que solo se deben recopilar los datos estrictamente necesarios.

4. Responsabilidad (Accountability):
Debe existir claridad sobre quién es responsable cuando un sistema de IA causa daño: ¿el desarrollador, el operador o el usuario? Los marcos regulatorios como la AI Act de la UE establecen responsabilidades según el nivel de riesgo.

5. Seguridad y Robustez:
Los sistemas de IA deben ser resistentes a ataques adversariales, fallos inesperados y comportamientos fuera de distribución. La seguridad debe considerarse desde el diseño (security by design).

6. Beneficio Social:
El desarrollo de IA debe orientarse hacia el bienestar humano y social, no exclusivamente hacia el beneficio económico.

Riesgos y Desafíos Éticos:
- Deepfakes y desinformación generada por IA
- Vigilancia masiva y erosión de la privacidad
- Automatización y desplazamiento laboral
- Concentración de poder en pocas corporaciones tecnológicas
- Uso militar y autónomo de sistemas letales
- Dependencia excesiva en sistemas no verificables

Marcos Regulatorios:
- AI Act (Unión Europea, 2024): Primera regulación integral de IA por nivel de riesgo.
- Blueprint for an AI Bill of Rights (EE.UU., 2022): Principios no vinculantes.
- Recomendación sobre Ética de la IA (UNESCO, 2021): Marco global de 193 países.
"""
}

os.makedirs("documentos", exist_ok=True)
for nombre, contenido in documentos.items():
    with open(f"documentos/{nombre}", "w", encoding="utf-8") as f:
        f.write(contenido.strip())
    print(f"Creado: documentos/{nombre}")

print(f"\nTotal: {len(documentos)} documentos creados en la carpeta 'documentos/'")
'@ | python
```

2. Verifica que los documentos fueron creados correctamente:

```powershell
Get-ChildItem -Force documentos
Get-ChildItem -Filter *.txt -Path documentos | Measure-Object
```

**Salida Esperada:**

```
Creado: documentos/ia_fundamentos.txt
Creado: documentos/machine_learning.txt
Creado: documentos/deep_learning.txt
Creado: documentos/nlp_transformers.txt
Creado: documentos/etica_ia.txt

Total: 5 documentos creados en la carpeta 'documentos/'
```

```
  documentos/deep_learning.txt
  documentos/etica_ia.txt
  documentos/ia_fundamentos.txt
  documentos/machine_learning.txt
  documentos/nlp_transformers.txt
```

**Verificación:**

- Confirma que existen exactamente 5 archivos `.txt` en la carpeta `documentos/`
- Verifica que cada archivo tiene contenido (tamaño > 0 bytes)

---

### Paso 2: Exploración Comparativa – Cadena vs Agente

1. Inicia Jupyter Notebook en la carpeta del laboratorio:

```powershell
jupyter notebook
```

2. Crea un nuevo notebook llamado `lab_02_cadenas_vs_agentes.ipynb` y ejecuta las siguientes celdas en orden:

**Celda 1 – Configuración de entorno:**

```python
# Celda 1: Configuración inicial
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Verificar que las variables de Azure OpenAI estén disponibles
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

required = {
    "AZURE_OPENAI_API_KEY": api_key,
    "AZURE_OPENAI_ENDPOINT": endpoint,
    "AZURE_OPENAI_DEPLOYMENT_NAME": deployment,
    "AZURE_OPENAI_API_VERSION": api_version,
}
missing = [name for name, value in required.items() if not value]
if missing:
    raise ValueError(f"ERROR: Variables faltantes en .env: {', '.join(missing)}")

print("Variables de Azure OpenAI configuradas correctamente.")
print(f"Endpoint: {endpoint}")
print(f"Deployment: {deployment}")
print(f"API Version: {api_version}")
print("Entorno configurado correctamente.")
```

**Celda 2 – Implementación con Cadena LCEL:**

```python
# Celda 2: Enfoque con Cadena LCEL (flujo fijo)
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("=" * 60)
print("ENFOQUE 1: CADENA LCEL (Flujo Fijo y Predecible)")
print("=" * 60)

# Definir componentes de la cadena
modelo = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
)
parser = StrOutputParser()

# Cadena para responder preguntas generales
prompt_qa = ChatPromptTemplate.from_template(
    """Eres un asistente de IA. Responde la siguiente pregunta de forma concisa.
    
Pregunta: {pregunta}

Respuesta:"""
)

# Construir la cadena con el operador pipe (|)
cadena_qa = prompt_qa | modelo | parser

# Probar con diferentes preguntas
preguntas_prueba = [
    "¿Qué es el machine learning?",
    "¿Cuánto es 1234 multiplicado por 567?",
    "¿Quién inventó el término inteligencia artificial?"
]

print("\nEjecutando cadena con 3 preguntas diferentes:\n")
for i, pregunta in enumerate(preguntas_prueba, 1):
    print(f"Pregunta {i}: {pregunta}")
    respuesta = cadena_qa.invoke({"pregunta": pregunta})
    print(f"Respuesta: {respuesta[:200]}...")
    print(f"{'─' * 50}")

print("\n⚠️  OBSERVACIÓN: La cadena siempre sigue el MISMO flujo:")
print("   Pregunta → Prompt → LLM → Parser → Respuesta")
print("   No puede buscar información actualizada ni hacer cálculos verificables.")
```

**Celda 3 – Implementación con Agente ReAct:**

```python
# Celda 3: Enfoque con Agente ReAct (flujo dinámico usando Azure OpenAI con .env)
import os
import math
from dotenv import load_dotenv  # Librería para cargar el archivo .env
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# Cargar las variables desde el archivo .env
load_dotenv()

# Importación clásica compatible con tu entorno virtual
try:
    from langchain.agents import initialize_agent, AgentType
except ImportError:
    from langchain_classic.agents import initialize_agent, AgentType

print("=" * 60)
print("ENFOQUE 2: AGENTE ReAct (Flujo Dinámico y Adaptable - AZURE)")
print("=" * 60)

@tool
def calculadora(expresion: str) -> str:
    """Evalúa expresiones matemáticas. Útil para cálculos numéricos exactos.
    Ejemplos de entrada: '1234 * 567', '25 ** 2', 'math.sqrt(144)'"""
    try:
        resultado = eval(expresion, {"__builtins__": {}}, {"math": math})
        return f"Resultado: {resultado}"
    except Exception as e:
        return f"Error al calcular: {str(e)}"

@tool
def conocimiento_ia(consulta: str) -> str:
    """Responde preguntas sobre inteligencia artificial, machine learning y temas relacionados.
    Usa esta herramienta para preguntas sobre conceptos de IA."""
    base = {
        "machine learning": "El machine learning es una rama de la IA que permite a los sistemas aprender automáticamente de datos sin ser programados explícitamente.",
        "inteligencia artificial": "La IA fue definida por John McCarthy en 1956. Simula procesos de inteligencia humana mediante sistemas informáticos.",
        "deep learning": "El deep learning usa redes neuronales con múltiples capas para aprender representaciones complejas de datos."
    }
    consulta_lower = consulta.lower()
    for clave, respuesta in base.items():
        if clave in consulta_lower:
            return respuesta
    return f"Información sobre '{consulta}': Tema relacionado con inteligencia artificial."

# Configuramos el modelo extrayendo los valores directamente de las variables de entorno cargadas
modelo_agente = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

herramientas_demo = [calculadora, conocimiento_ia]

ejecutor = initialize_agent(
    tools=herramientas_demo,
    llm=modelo_agente,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

print("\n" + "=" * 60)
print("Ejecutando agente con las MISMAS preguntas en Azure OpenAI:")
print("=" * 60)

preguntas_prueba = [
    "¿Cuánto es 1234 multiplicado por 567?",
    "¿Qué es el machine learning?",
]

for pregunta in preguntas_prueba:
    print(f"\n{'═' * 60}")
    print(f"PREGUNTA: {pregunta}")
    print('═' * 60)
    resultado = ejecutor.invoke({"input": pregunta})
    print(f"\nRESPUESTA FINAL: {resultado['output']}")
```

**Celda 4 – Análisis comparativo:**

```python
# Celda 4: Tabla comparativa de resultados
print("\n" + "=" * 70)
print("ANÁLISIS COMPARATIVO: CADENA vs AGENTE")
print("=" * 70)

comparacion = [
    ("Flujo de ejecución", "Fijo: prompt→LLM→parser", "Dinámico: Thought→Action→Obs"),
    ("Uso de herramientas", "No (responde desde LLM)", "Sí, selección automática"),
    ("Cálculo 1234×567", "Puede alucinar el resultado", "Usa calculadora: resultado exacto"),
    ("Preguntas de IA", "Responde desde conocimiento LLM", "Selecciona herramienta apropiada"),
    ("Predictibilidad", "Alta: siempre mismo flujo", "Variable: depende de la pregunta"),
    ("Transparencia", "Baja: caja negra", "Alta: muestra razonamiento paso a paso"),
    ("Casos de uso", "Tareas estructuradas fijas", "Tareas abiertas y complejas"),
]

print(f"\n{'Característica':<25} {'Cadena LCEL':<35} {'Agente ReAct':<35}")
print("-" * 95)
for caract, cadena, agente in comparacion:
    print(f"{caract:<25} {cadena:<35} {agente:<35}")

print("\n✅ CONCLUSIÓN CLAVE:")
print("   - Usa CADENAS cuando el flujo es conocido y estructurado.")
print("   - Usa AGENTES cuando necesitas flexibilidad y herramientas externas.")
```

**Salida Esperada (parcial):**

```
ANÁLISIS COMPARATIVO: CADENA vs AGENTE
======================================================================

Característica            Cadena LCEL                         Agente ReAct                       
-----------------------------------------------------------------------------------------------
Flujo de ejecución        Fijo: prompt→LLM→parser             Dinámico: Thought→Action→Obs       
Uso de herramientas       No (responde desde LLM)             Sí, selección automática           
Cálculo 1234×567          Puede alucinar el resultado         Usa calculadora: resultado exacto  
Preguntas de IA           Responde desde conocimiento LLM     Selecciona herramienta apropiada   
Predictibilidad           Alta: siempre mismo flujo           Variable: depende de la pregunta   
Transparencia             Baja: caja negra                    Alta: muestra razonamiento paso a paso
Casos de uso              Tareas estructuradas fijas          Tareas abiertas y complejas        

✅ CONCLUSIÓN CLAVE:
   - Usa CADENAS cuando el flujo es conocido y estructurado.
   - Usa AGENTES cuando necesitas flexibilidad y herramientas externas.
```

**Verificación:**

- La cadena responde todas las preguntas con el mismo flujo sin mostrar razonamiento
- El agente muestra explícitamente los pasos `Thought → Action → Observation → Final Answer`
- Para el cálculo matemático, el agente usa la herramienta `calculadora` y obtiene el resultado exacto (699678)

---

### Paso 3: Construir el Índice RAG con LlamaIndex

**Objetivo:** Indexar los documentos locales de la carpeta `documentos/` usando LlamaIndex con embeddings de OpenAI y almacenar el índice vectorial en FAISS para búsqueda semántica eficiente.

**Instrucciones:**

1. Instala el módulo Llama Index Embeddings para Azure OpenAI. Crea una nueva celda, agrega el siguiente script y ejecútala. 

```
!pip install llama_index-embeddings-azure-openai llama_index-llms-azure-openai
```
2. Regresa a Visual Studio Code, abre el archivo .env y agrega esta línea al final.

```
AZURE_OPENAI_DEPLOYMENT_EMBEDDING=text-embedding-ada-002
```

3. Crea un nuevo notebook llamado `lab_02_rag_index.ipynb` o continúa en el mismo notebook con nuevas celdas:

**Celda 5 – Construcción del índice RAG:**

```python
# Celda 5: Construcción del índice vectorial con LlamaIndex + FAISS
import os
import faiss
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from llama_index.vector_stores.faiss import FaissVectorStore

# Cargar las variables desde el archivo .env aplicando override
load_dotenv(override=True)

# 1. Credenciales y Despliegues dinámicos desde el archivo .env
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Separación clara de despliegues (LLM vs Embeddings)
LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") 
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING")

print("=" * 60)
print("FASE 2: Construcción del Índice RAG con LlamaIndex")
print("=" * 60)
print(f"   [CONFIG] LLM Deployment: {LLM_DEPLOYMENT}")
print(f"   [CONFIG] Embedding Deployment: {EMBED_DEPLOYMENT}")
print("-" * 60)

# 2. Configurar el modelo de embeddings Azure OpenAI utilizando la variable del .env
print("\n1. Configurando modelo de embeddings Azure OpenAI...")
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=EMBED_DEPLOYMENT,  
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_VERSION
)

# 3. Configurar el LLM para LlamaIndex
print("   Configurando modelo de lenguaje Azure OpenAI...")
llm_llamaindex = LlamaAzureOpenAI(
    model="gpt-4o-mini",
    deployment_name=LLM_DEPLOYMENT,   
    temperature=0,
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_VERSION
)

# Aplicar configuraciones globales en los Settings de LlamaIndex
Settings.embed_model = embed_model
Settings.llm = llm_llamaindex
Settings.chunk_size = 512      
Settings.chunk_overlap = 50    

# 4. Cargar documentos desde la carpeta local
print("\n2. Cargando documentos desde 'documentos/'...")
lector = SimpleDirectoryReader(
    input_dir="documentos",
    required_exts=[".txt"]
)
documentos_cargados = lector.load_data()
print(f"   Documentos cargados: {len(documentos_cargados)} archivos")

# 5. Crear el vector store FAISS
print("\n3. Creando vector store FAISS...")
dimension_embeddings = 1536  # Dimensión nativa para text-embedding-ada-002
indice_faiss = faiss.IndexFlatL2(dimension_embeddings)
vector_store = FaissVectorStore(faiss_index=indice_faiss)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 6. Construir el índice vectorial (Ahora sí llamando a la API correcta)
print("\n4. Generando embeddings e indexando documentos...")
indice = VectorStoreIndex.from_documents(
    documentos_cargados,
    storage_context=storage_context,
    show_progress=True
)
print("   ✅ Índice construido exitosamente.")

# 7. Persistir el índice en disco
print("\n5. Guardando índice en disco...")
os.makedirs("indice_faiss", exist_ok=True)
indice.storage_context.persist(persist_dir="indice_faiss")
print("   ✅ Índice guardado en 'indice_faiss/'")

# 8. Consulta de prueba de extremo a extremo
print("\n6. Probando el índice con consulta de prueba...")
motor_consulta = indice.as_query_engine(similarity_top_k=3)
respuesta_prueba = motor_consulta.query(
    "¿Qué es el aprendizaje supervisado y cuáles son sus algoritmos principales?"
)
print(f"\nConsulta: ¿Qué es el aprendizaje supervisado?")
print(f"Respuesta RAG: {str(respuesta_prueba)[:400]}...")

print("\n✅ Base de conocimiento RAG lista para ser usada como herramienta.")
```

**Celda 6 – Función de búsqueda RAG reutilizable:**

```python
# Celda 6: Carga del índice persistido y verificación del motor RAG
import os
import faiss
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore

def crear_motor_rag():
    """Carga el índice FAISS persistido y retorna el motor de consulta."""
    
    # NOTA: LlamaIndex consume automáticamente los modelos (embed_model y llm) 
    # asignados previamente a la clase global 'Settings' en la Celda 5.
    
    # 1. Cargar el contenedor de vectores FAISS desde el disco
    vector_store_cargado = FaissVectorStore.from_persist_dir("indice_faiss")
    
    # 2. Reconstruir el contexto de almacenamiento
    storage_context_cargado = StorageContext.from_defaults(
        vector_store=vector_store_cargado,
        persist_dir="indice_faiss"
    )
    
    # 3. Cargar la estructura del índice
    indice_cargado = load_index_from_storage(storage_context_cargado)
    
    # 4. Retornar el motor configurado para recuperar los 3 fragmentos más cercanos
    return indice_cargado.as_query_engine(similarity_top_k=3)

# --- Ejecución y Verificación ---
print("=" * 60)
print("FASE 3: Verificación del Motor RAG desde Disco")
print("=" * 60)

# Asegurar que las variables globales sigan vigentes (por si ejecutas esta celda de forma aislada)
if Settings.embed_model is None or Settings.llm is None:
    print("⚠️ Advertencia: Settings globales no detectados. Asegúrate de ejecutar la Celda 5 primero.")

print("Verificando carga del índice desde disco...")
motor_verificacion = crear_motor_rag()

# Ejecutar consulta de prueba
consulta_test = "¿Cuáles son los principios de la ética en IA?"
respuesta_verificacion = motor_verificacion.query(consulta_test)

print(f"\nConsulta de verificación: {consulta_test}")
print("-" * 60)
print(f"Respuesta RAG: {str(respuesta_verificacion)[:400]}...")
print("-" * 60)
print("\n✅ Motor RAG verificado y listo para ser integrado al Agente.")
```

**Salida Esperada:**

```
============================================================
FASE 2: Construcción del Índice RAG con LlamaIndex
============================================================

1. Configurando modelo de embeddings OpenAI...
   Embeddings: text-embedding-3-small
   LLM: gpt-3.5-turbo
   Chunk size: 512 tokens | Overlap: 50 tokens

2. Cargando documentos desde 'documentos/'...
   Documentos cargados: 5 archivos
   - deep_learning.txt (2847 caracteres)
   - etica_ia.txt (2931 caracteres)
   - ia_fundamentos.txt (2156 caracteres)
   - machine_learning.txt (2203 caracteres)
   - nlp_transformers.txt (2489 caracteres)

3. Creando vector store FAISS...
4. Generando embeddings e indexando documentos...
   (Este proceso puede tomar 30-60 segundos...)
Parsing nodes: 100%|████████████| 5/5 [00:00<00:00]
Generating embeddings: 100%|████████████| 18/18 [00:08<00:00]
   ✅ Índice construido exitosamente.

5. Guardando índice en disco...
   ✅ Índice guardado en 'indice_faiss/'

6. Probando el índice con consulta de prueba...
Respuesta RAG: El aprendizaje supervisado es un tipo de machine learning donde el algoritmo aprende de datos etiquetados...
```

**Verificación:**

- Confirma que la carpeta `indice_faiss/` fue creada con archivos en su interior: `ls -la indice_faiss/`
- La consulta de prueba devuelve una respuesta coherente con el contenido de `machine_learning.txt`
- No aparecen errores de API ni de dimensiones de embeddings

---

### Paso 4: Definir las Tres Herramientas del Agente

**Objetivo:** Crear las tres herramientas especializadas que el agente utilizará: búsqueda RAG en la base de conocimiento local, calculadora matemática y búsqueda en Wikipedia.

**Instrucciones:**

1. Continúa en el notebook con las siguientes celdas:

**Celda 7 – Herramienta 1: Búsqueda RAG en base de conocimiento local:**

```python
# Celda 7: Creación de la Herramienta RAG para el Agente LangChain
import os
from dotenv import load_dotenv
from langchain_core.tools import tool

# Importaciones base de LlamaIndex
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI

# Cargar las variables desde el archivo .env
load_dotenv(override=True)

# 1. Credenciales y Despliegues dinámicos desde el archivo .env
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Separación estricta de despliegues (LLM vs Embeddings)
LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") 
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING")

print("=" * 60)
print("FASE 4: Configuración de la Herramienta RAG LangChain")
print("=" * 60)

# 2. Asegurar configuración de LlamaIndex (necesario si se reinicia el kernel)
print("1. Sincronizando Settings globales de LlamaIndex...")
Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=EMBED_DEPLOYMENT,
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_VERSION
)

Settings.llm = LlamaAzureOpenAI(
    model="gpt-4o-mini",
    deployment_name=LLM_DEPLOYMENT,
    temperature=0,
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_VERSION
)

# 3. Inicializar el motor RAG (Asegúrate de haber ejecutado la función 'crear_motor_rag' previamente)
print("\n2. Inicializando motor RAG (Cargando índice FAISS desde disco)...")
try:
    motor_rag = crear_motor_rag()
    print("   ✅ Motor RAG inicializado con éxito.")
except NameError:
    print("   ❌ Error: La función 'crear_motor_rag' no está definida. Ejecuta la celda anterior primero.")
except Exception as e:
    print(f"   ❌ Error al cargar el motor RAG: {str(e)}")

# 4. Definición de la herramienta compatible con el Agente LangChain
@tool
def buscar_base_conocimiento(consulta: str) -> str:
    """Busca información en la base de conocimiento local sobre inteligencia artificial,
    machine learning, deep learning, procesamiento de lenguaje natural (NLP),
    transformers, redes neuronales y ética en IA.
    
    Usa esta herramienta cuando la pregunta sea sobre conceptos técnicos de IA,
    definiciones, historia, algoritmos o aplicaciones de inteligencia artificial.
    
    Parámetro consulta: pregunta o tema a buscar en la base de conocimiento."""
    try:
        respuesta = motor_rag.query(consulta)
        resultado = str(respuesta)
        if not resultado or resultado.strip() == "":
            return "No se encontró información relevante sobre ese tema en la base de conocimiento."
        return f"Información de la base de conocimiento:\n{resultado}"
    except Exception as e:
        return f"Error al consultar la base de conocimiento: {str(e)}"

# 5. Verificar e invocar la herramienta de prueba
print("\n3. Probando herramienta 'buscar_base_conocimiento':")
print(f"   -> Nombre: {buscar_base_conocimiento.name}")
print(f"   -> Descripción resumida: {buscar_base_conocimiento.description[:110]}...")

try:
    # Usamos invoke para simular la llamada del Agente
    resultado_prueba = buscar_base_conocimiento.invoke("¿Qué son los transformers en IA?")
    print(f"\n[TEST] Resultado de prueba (Primeros 300 caracteres):")
    print("-" * 60)
    print(f"{resultado_prueba[:300]}...")
    print("-" * 60)
    print("✅ Herramienta RAG empaquetada y lista.")
except Exception as e:
    print(f"   ❌ Fallo la ejecución de prueba de la herramienta: {str(e)}")
```

**Celda 8 – Herramienta 2: Calculadora matemática:**

```python
# Celda 8: Herramienta 2 - Calculadora matemática segura
import math
import re
from langchain_core.tools import tool  # 👈 CORRECCIÓN CRÍTICA: Importación del decorador tool

print("=" * 60)
print("FASE 5: Configuración de la Herramienta Calculadora Segura")
print("=" * 60)

@tool
def calculadora_matematica(expresion: str) -> str:
    """Realiza cálculos matemáticos con precisión. Evalúa expresiones numéricas.
    
    Usa esta herramienta para:
    - Operaciones aritméticas: suma, resta, multiplicación, división
    - Potencias y raíces cuadradas
    - Operaciones trigonométricas y logarítmicas
    - Cualquier cálculo numérico que requiera exactitud
    
    Ejemplos de expresiones válidas:
    - '1234 * 567' para multiplicación
    - '2 ** 10' para potencias
    - 'math.sqrt(144)' para raíz cuadrada
    - '(100 + 200) / 3' para operaciones combinadas
    - 'math.pi * 5 ** 2' para área de círculo con radio 5
    
    Parámetro expresion: expresión matemática como string."""
    try:
        # Limpiar la expresión de caracteres no permitidos
        expresion_limpia = expresion.strip()
        
        # Namespace seguro: solo permitir operaciones matemáticas
        namespace_seguro = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "pow": pow,
            "sum": sum,
            "min": min,
            "max": max,
        }
        
        resultado = eval(expresion_limpia, namespace_seguro)
        
        # Formatear el resultado
        if isinstance(resultado, float):
            if resultado == int(resultado):
                return f"Resultado de '{expresion_limpia}' = {int(resultado)}"
            else:
                return f"Resultado de '{expresion_limpia}' = {resultado:.6f}"
        return f"Resultado of '{expresion_limpia}' = {resultado}"
        
    except ZeroDivisionError:
        return "Error: División por cero no permitida."
    except SyntaxError:
        return f"Error de sintaxis en la expresión: '{expresion}'. Verifica que sea una expresión matemática válida."
    except Exception as e:
        return f"Error al calcular '{expresion}': {str(e)}"

# --- Ejecución y Verificación ---
print(f"Probando herramienta '{calculadora_matematica.name}':")
print(f"Descripción resumida: {calculadora_matematica.description[:90]}...\n")

casos_prueba = [
    "1234 * 567",
    "2 ** 10",
    "math.sqrt(144)",
    "math.pi * 5 ** 2",
    "(100 + 200) / 3"
]

for caso in casos_prueba:
    # Usamos invoke para simular la ejecución nativa del agente
    resultado = calculadora_matematica.invoke(caso)
    print(f"   [TEST] {caso} → {resultado}")

print("\n✅ Herramienta calculadora verificada y lista.")
```

**Celda 9 – Herramienta 3: Búsqueda en Wikipedia:**

```python
# Celda 9: Herramienta 3 - Búsqueda en Wikipedia
import wikipedia
from langchain_core.tools import tool  # 👈 CORRECCIÓN CRÍTICA: Importación obligatoria para el entorno

print("=" * 60)
print("FASE 6: Configuración de la Herramienta Wikipedia")
print("=" * 60)

# Configurar Wikipedia en español
wikipedia.set_lang("es")

@tool
def buscar_wikipedia(termino: str) -> str:
    """Busca información actualizada en Wikipedia sobre cualquier tema general.
    
    Usa esta herramienta cuando necesites información sobre:
    - Personas, organizaciones o empresas tecnológicas
    - Eventos históricos o actuales
    - Conceptos generales no relacionados con IA
    - Información geográfica, científica o cultural
    - Cualquier tema que NO esté en la base de conocimiento local de IA
    
    Parámetro termino: término o frase a buscar en Wikipedia (en español preferiblemente)."""
    try:
        # Buscar sugerencias de páginas
        sugerencias = wikipedia.search(termino, results=3)
        
        if not sugerencias:
            return f"No se encontraron resultados en Wikipedia para: '{termino}'"
        
        # Intentar obtener el resumen de la primera sugerencia
        for sugerencia in sugerencias:
            try:
                pagina = wikipedia.page(sugerencia, auto_suggest=False)
                resumen = wikipedia.summary(sugerencia, sentences=4, auto_suggest=False)
                return (
                    f"Wikipedia - {pagina.title}:\n"
                    f"{resumen}\n"
                    f"(Fuente: {pagina.url})"
                )
            except wikipedia.exceptions.DisambiguationError as e:
                # Si hay ambigüedad, intentar con la primera opción de la lista
                try:
                    primera_opcion = e.options[0]
                    resumen = wikipedia.summary(primera_opcion, sentences=4)
                    return f"Wikipedia - {primera_opcion}:\n{resumen}"
                except:
                    continue
            except wikipedia.exceptions.PageError:
                continue
        
        return f"No se pudo obtener información de Wikipedia para: '{termino}'"
        
    except Exception as e:
        return f"Error al consultar Wikipedia: {str(e)}"

# --- Ejecución y Verificación ---
print(f"Probando herramienta '{buscar_wikipedia.name}':")
print(f"Descripción resumida: {buscar_wikipedia.description[:95]}...\n")

# Ejecución simulada con invoke
try:
    resultado_wiki = buscar_wikipedia.invoke("Alan Turing")
    print(f"[TEST] Resultado de prueba (Primeros 300 caracteres):")
    print("-" * 60)
    print(f"{resultado_wiki[:300]}...")
    print("-" * 60)
    print("\n✅ Herramienta Wikipedia lista y verificada.")
except Exception as e:
    print(f"❌ Error al ejecutar la prueba de la herramienta: {str(e)}")
```

**Celda 10 – Resumen de herramientas definidas:**

```python
# Celda 10: Resumen de las 3 herramientas definidas
herramientas_agente = [
    buscar_base_conocimiento,
    calculadora_matematica,
    buscar_wikipedia
]

print("=" * 60)
print("RESUMEN: 3 HERRAMIENTAS DEFINIDAS PARA EL AGENTE")
print("=" * 60)
for i, herramienta in enumerate(herramientas_agente, 1):
    print(f"\nHerramienta {i}: {herramienta.name}")
    print(f"Descripción: {herramienta.description[:120]}...")
    print(f"Tipo de retorno: str")

print("\n✅ Las 3 herramientas están listas para ser integradas en el agente.")
print("\nRecuerda: La DESCRIPCIÓN de cada herramienta es crítica.")
print("El LLM usa esas descripciones para decidir cuándo invocar cada herramienta.")
```

**Salida Esperada:**

```
Inicializando motor RAG...
Motor RAG inicializado.

Probando herramienta 'buscar_base_conocimiento':
Nombre: buscar_base_conocimiento
Descripción: Busca información en la base de conocimiento local sobre inteligencia artificial...
Resultado de prueba: Información de la base de conocimiento:
Los transformers son una arquitectura revolucionaria...

Probando herramienta 'calculadora_matematica':
Nombre: calculadora_matematica
  1234 * 567 → Resultado de '1234 * 567' = 699678
  2 ** 10 → Resultado de '2 ** 10' = 1024
  math.sqrt(144) → Resultado de 'math.sqrt(144)' = 12
  math.pi * 5 ** 2 → Resultado de 'math.pi * 5 ** 2' = 78.539816
  (100 + 200) / 3 → Resultado de '(100 + 200) / 3' = 100.000000

Probando herramienta 'buscar_wikipedia':
Nombre: buscar_wikipedia
Resultado de prueba:
Wikipedia - Alan Turing:
Alan Mathison Turing fue un matemático, lógico, científico de la computación...

✅ Las 3 herramientas están listas para ser integradas en el agente.
```

**Verificación:**

- Las tres herramientas se prueban individualmente sin errores
- `calculadora_matematica` retorna `699678` para `1234 * 567`
- `buscar_base_conocimiento` retorna contenido relacionado con los documentos indexados
- `buscar_wikipedia` retorna información real de Wikipedia

---

### Paso 5: Ensamblar el Agente ReAct Completo

**Objetivo:** Integrar las tres herramientas en un agente ReAct completo usando `AgentExecutor` y verificar que selecciona automáticamente la herramienta correcta para cada tipo de pregunta.

**Instrucciones:**

1. Continúa en el notebook con las siguientes celdas:

**Celda 11 – Construcción del agente ReAct completo:**

```python
# Celda 11: Ensamblado del Agente ReAct Completo (Enfoque Nativo y Blindado)
import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.agents import AgentAction, AgentFinish

# Asegurar carga limpia de variables locales
load_dotenv(override=True)
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

print("=" * 60)
print("FASE 7: Ensamblado del Agente ReAct Completo")
print("=" * 60)

# 1. Prompt ReAct estándar e independiente (Inyectado localmente)
print("\n1. Configurando prompt ReAct local (Independiente del Hub)...")
prompt_react_local = ChatPromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)
print("   ✅ Prompt estructurado correctamente.")

# 2. Configurar el modelo LLM con las credenciales locales estrictas
print("\n2. Configurando modelo LLM (Azure OpenAI)...")
llm_agente = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    api_version=AZURE_VERSION,
    azure_deployment=LLM_DEPLOYMENT,
    temperature=0,
    max_tokens=2000,
)
print(f"   ✅ Modelo inicializado en el despliegue: {LLM_DEPLOYMENT}")

# 3. Consolidación de Herramientas Registradas
herramientas_agente = [
    buscar_base_conocimiento,
    calculadora_matematica,
    buscar_wikipedia
]

# Crear mapas e información textual para alimentar el razonamiento del Agente
tabla_herramientas = {tool.name: tool for tool in herramientas_agente}
texto_herramientas = "\n".join([f"{tool.name}: {tool.description}" for tool in herramientas_agente])
nombres_herramientas = ", ".join([tool.name for tool in herramientas_agente])

print(f"\n3. Herramientas registradas exitosamente: {[h.name for h in herramientas_agente]}")

# 4. Funciones locales de formateo y parseo personalizado de ReAct (Filosofía Celda 3)
def formatear_historial(pasos):
    string_log = ""
    for action, observation in pasos:
        string_log += f" {action.log}\nObservation: {observation}\nThought:"
    return string_log

def parsear_salida_react(resultado_llm):
    texto = resultado_llm.content
    
    if "Final Answer:" in texto:
        respuesta = texto.split("Final Answer:")[-1].strip()
        return AgentFinish(return_values={"output": respuesta}, log=texto)
    
    if "Action:" in texto and "Action Input:" in texto:
        tool_name = texto.split("Action:")[1].split("Action Input:")[0].strip()
        tool_input = texto.split("Action Input:")[1].strip().strip("'\"")
        return AgentAction(tool=tool_name, tool_input=tool_input, log=texto)
    
    return AgentFinish(return_values={"output": texto}, log=texto)

# 5. Construcción de la cadena del agente usando LCEL Puro
llm_con_stop = llm_agente.bind(stop=["\nObservation:"])

cadena_agente = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: formatear_historial(x["steps"]),
        "tools": lambda x: texto_herramientas,
        "tool_names": lambda x: nombres_herramientas
    }
    | prompt_react_local
    | llm_con_stop
    | parsear_salida_react
)
print("   ✅ Cadena LCEL del Agente ReAct construida.")

# 6. Orquestador Nativo del Agente (Equivalente robusto a AgentExecutor)
class OrquestadorAgente:
    def __init__(self, cadena, tabla_herramientas, max_iterations=8):
        self.cadena = cadena
        self.tabla_herramientas = tabla_herramientas
        self.max_iterations = max_iterations

    def invoke(self, pregunta_usuario: str):
        pasos_intermedios = []
        
        print(f"\n{'═' * 60}")
        print(f"PREGUNTA: {pregunta_usuario}")
        print('═' * 60)
        print("> Entering new Agent Executor chain...")
        
        for i in range(self.max_iterations):
            # El modelo genera el pensamiento y decide el siguiente paso
            salida = self.cadena.invoke({"input": pregunta_usuario, "steps": pasos_intermedios})
            
            # Si el agente determina que tiene la respuesta final
            if isinstance(salida, AgentFinish):
                print(f"\nThought: I now know the final answer.")
                print(f"Final Answer: {salida.return_values['output']}")
                return {
                    "output": salida.return_values["output"],
                    "intermediate_steps": pasos_intermedios
                }
            
            # Si decide que necesita ejecutar una acción (Herramienta)
            if isinstance(salida, AgentAction):
                print(f"\nThought: {salida.log.split('Action:')[0].strip()}")
                print(f"Action: {salida.tool}")
                print(f"Action Input: {salida.tool_input}")
                
                # Ejecutar la herramienta correspondiente de forma segura
                if salida.tool in self.tabla_herramientas:
                    observacion = self.tabla_herramientas[salida.tool].run(salida.tool_input)
                else:
                    observacion = f"Error: La herramienta '{salida.tool}' no existe en el catálogo."
                    
                print(f"Observation: {observacion}")
                
                # Almacenar el paso intermedio para la memoria de la iteración
                pasos_intermedios.append((salida, observacion))
                
        return {
            "output": "Error: Se alcanzó el límite máximo de iteraciones permitidas sin llegar a una conclusión.",
            "intermediate_steps": pasos_intermedios
        }

# Instanciar nuestro ejecutor personalizado equivalente a ejecutor_qa
ejecutor_qa = OrquestadorAgente(cadena_agente, tabla_herramientas, max_iterations=8)

print("\n✅ AgentExecutor Nativo configurado y listo.")
print("\nConfiguración del agente:")
print(f"  - LLM Deployment: {LLM_DEPLOYMENT}")
print(f"  - Herramientas integradas: {len(herramientas_agente)} {[h.name for h in herramientas_agente]}")
print(f"  - Max iteraciones configuradas: 8")
print(f"  - Verbose por defecto: True (Muestra la cadena de CoT completa)")
```

**Celda 12 – Función helper para ejecutar el agente:**

```python
# Celda 12: Función helper para ejecutar el agente y mostrar resultados
def ejecutar_pregunta(pregunta: str, numero: int = None) -> dict:
    """Ejecuta el agente con una pregunta y muestra el resultado formateado."""
    separador = "═" * 65
    if numero:
        print(f"\n{separador}")
        print(f"PREGUNTA {numero}: {pregunta}")
        print(separador)
    else:
        print(f"\n{separador}")
        print(f"PREGUNTA: {pregunta}")
        print(separador)
    
    try:
        # CORRECCIÓN CRÍTICA: Pasamos la 'pregunta' directamente como string 
        # para que coincida exactamente con la firma de nuestro OrquestadorAgente
        resultado = ejecutor_qa.invoke(pregunta)
        
        print(f"\n{'─' * 65}")
        print(f"✅ RESPUESTA FINAL:")
        print(f"{'─' * 65}")
        print(resultado["output"])
        
        # Mostrar herramientas utilizadas
        if "intermediate_steps" in resultado and resultado["intermediate_steps"]:
            herramientas_usadas = [
                paso[0].tool for paso in resultado["intermediate_steps"]
            ]
            print(f"\n🔧 Herramientas utilizadas: {herramientas_usadas}")
        else:
            print(f"\n🔧 Herramientas utilizadas: ninguna (respuesta directa)")
        
        return resultado
        
    except Exception as e:
        print(f"\n❌ Error al ejecutar la pregunta: {str(e)}")
        return {"output": f"Error: {str(e)}", "intermediate_steps": []}

print("Función 'ejecutar_pregunta' definida y lista.")
```

**Verificación:**

- El agente se crea sin errores
- `prompt_react.input_variables` incluye `['agent_scratchpad', 'input', 'tool_names', 'tools']`
- `AgentExecutor` se configura con `verbose=True` y `return_intermediate_steps=True`

---

### Paso 6: Evaluar el Agente con Banco de Preguntas

**Objetivo:** Ejecutar un banco de 10 preguntas variadas que fuercen el uso de diferentes herramientas, interpretando el razonamiento paso a paso en los logs de ejecución.

**Instrucciones:**

1. Continúa en el notebook con las siguientes celdas:

**Celda 13 – Banco de preguntas de evaluación:**

```python
# Celda 13: Banco de 10 preguntas para evaluar el agente
# Cada pregunta está diseñada para activar una herramienta específica

banco_preguntas = [
    # Preguntas que deben usar buscar_base_conocimiento (RAG)
    {
        "numero": 1,
        "pregunta": "¿Qué es el aprendizaje supervisado y cuáles son sus algoritmos principales?",
        "herramienta_esperada": "buscar_base_conocimiento",
        "categoria": "RAG - Machine Learning"
    },
    {
        "numero": 2,
        "pregunta": "Explica la arquitectura Transformer y por qué revolucionó el NLP",
        "herramienta_esperada": "buscar_base_conocimiento",
        "categoria": "RAG - Deep Learning / NLP"
    },
    {
        "numero": 3,
        "pregunta": "¿Cuáles son los principios fundamentales de la ética en inteligencia artificial?",
        "herramienta_esperada": "buscar_base_conocimiento",
        "categoria": "RAG - Ética IA"
    },
    # Preguntas que deben usar calculadora_matematica
    {
        "numero": 4,
        "pregunta": "¿Cuánto es 1234 multiplicado por 567?",
        "herramienta_esperada": "calculadora_matematica",
        "categoria": "Cálculo - Multiplicación"
    },
    {
        "numero": 5,
        "pregunta": "Calcula el área de un círculo con radio 15 metros",
        "herramienta_esperada": "calculadora_matematica",
        "categoria": "Cálculo - Geometría"
    },
    {
        "numero": 6,
        "pregunta": "¿Cuánto es 2 elevado a la potencia 16?",
        "herramienta_esperada": "calculadora_matematica",
        "categoria": "Cálculo - Potencia"
    },
    # Preguntas que deben usar buscar_wikipedia
    {
        "numero": 7,
        "pregunta": "¿Quién fue Alan Turing y cuál fue su contribución a la computación?",
        "herramienta_esperada": "buscar_wikipedia",
        "categoria": "Wikipedia - Persona histórica"
    },
    {
        "numero": 8,
        "pregunta": "¿Qué es OpenAI como empresa y cuándo fue fundada?",
        "herramienta_esperada": "buscar_wikipedia",
        "categoria": "Wikipedia - Empresa tecnológica"
    },
    # Preguntas que pueden requerir múltiples herramientas
    {
        "numero": 9,
        "pregunta": "Si una red neuronal tiene 3 capas ocultas con 128, 64 y 32 neuronas respectivamente, ¿cuántas neuronas tiene en total en las capas ocultas? Además, ¿qué es una red neuronal?",
        "herramienta_esperada": "calculadora_matematica + buscar_base_conocimiento",
        "categoria": "Multi-herramienta"
    },
    {
        "numero": 10,
        "pregunta": "¿Qué es el overfitting en machine learning y cómo se calcula el error cuadrático medio de las predicciones [2, 3, 4] vs valores reales [2.5, 3.5, 3.5]?",
        "herramienta_esperada": "buscar_base_conocimiento + calculadora_matematica",
        "categoria": "Multi-herramienta - ML + Cálculo"
    },
]

print("=" * 65)
print("BANCO DE EVALUACIÓN: 10 PREGUNTAS VARIADAS")
print("=" * 65)
print(f"\nTotal de preguntas: {len(banco_preguntas)}")
print("\nDistribución por categoría:")
categorias = {}
for p in banco_preguntas:
    cat = p["herramienta_esperada"]
    categorias[cat] = categorias.get(cat, 0) + 1
for cat, count in categorias.items():
    print(f"  - {cat}: {count} pregunta(s)")

print("\n⚠️  NOTA: Ejecutar todas las preguntas tomará 3-5 minutos.")
print("   Observa el razonamiento Thought → Action → Observation en cada respuesta.")
```

**Celda 14 – Ejecución de las primeras 5 preguntas:**

```python
# Celda 14: Ejecutar preguntas 1-5 (herramientas individuales)
print("EJECUTANDO PREGUNTAS 1-5 (Herramientas Individuales)")
print("=" * 65)

resultados = {}

for pregunta_info in banco_preguntas[:5]:
    resultado = ejecutar_pregunta(
        pregunta_info["pregunta"],
        pregunta_info["numero"]
    )
    resultados[pregunta_info["numero"]] = {
        "pregunta": pregunta_info["pregunta"],
        "respuesta": resultado.get("output", ""),
        "herramienta_esperada": pregunta_info["herramienta_esperada"],
        "pasos": resultado.get("intermediate_steps", [])
    }
    print(f"\n{'─' * 65}")
```

**Celda 15 – Ejecución de las preguntas 6-10:**

```python
# Celda 15: Ejecutar preguntas 6-10 (incluyendo multi-herramienta)
print("EJECUTANDO PREGUNTAS 6-10 (Incluyendo Multi-herramienta)")
print("=" * 65)

for pregunta_info in banco_preguntas[5:]:
    resultado = ejecutar_pregunta(
        pregunta_info["pregunta"],
        pregunta_info["numero"]
    )
    resultados[pregunta_info["numero"]] = {
        "pregunta": pregunta_info["pregunta"],
        "respuesta": resultado.get("output", ""),
        "herramienta_esperada": pregunta_info["herramienta_esperada"],
        "pasos": resultado.get("intermediate_steps", [])
    }
    print(f"\n{'─' * 65}")
```

**Celda 16 – Análisis de resultados:**

```python
# Celda 16: Análisis y reporte de resultados del banco de pruebas
print("\n" + "=" * 65)
print("REPORTE FINAL DE EVALUACIÓN DEL AGENTE")
print("=" * 65)

total_preguntas = len(resultados)
preguntas_con_herramienta = 0
herramientas_conteo = {}

for num, datos in resultados.items():
    pasos = datos["pasos"]
    if pasos:
        preguntas_con_herramienta += 1
        for paso in pasos:
            herramienta_usada = paso[0].tool
            herramientas_conteo[herramienta_usada] = herramientas_conteo.get(herramienta_usada, 0) + 1

print(f"\n📊 Estadísticas de Ejecución:")
print(f"   Total preguntas evaluadas: {total_preguntas}")
print(f"   Preguntas que usaron herramientas: {preguntas_con_herramienta}/{total_preguntas}")
print(f"   Respuestas directas (sin herramienta): {total_preguntas - preguntas_con_herramienta}")

print(f"\n🔧 Uso de Herramientas:")
for herramienta, count in sorted(herramientas_conteo.items(), key=lambda x: x[1], reverse=True):
    print(f"   - {herramienta}: {count} invocación(es)")

print(f"\n📋 Resumen por Pregunta:")
print(f"{'#':<4} {'Herramienta Esperada':<40} {'Herramientas Usadas':<30}")
print("-" * 74)
for num, datos in resultados.items():
    herramientas_usadas = list(set([paso[0].tool for paso in datos["pasos"]])) if datos["pasos"] else ["ninguna"]
    herramientas_str = ", ".join(herramientas_usadas)
    esperada = banco_preguntas[num-1]["herramienta_esperada"]
    print(f"{num:<4} {esperada:<40} {herramientas_str:<30}")

print(f"\n✅ Evaluación completada.")
print(f"\n🔍 REFLEXIÓN SOBRE EL CHAIN-OF-THOUGHT:")
print("   El patrón ReAct permite observar el razonamiento del agente:")
print("   1. THOUGHT: El modelo analiza la pregunta y decide la estrategia")
print("   2. ACTION: Selecciona e invoca la herramienta más apropiada")
print("   3. OBSERVATION: Recibe y procesa el resultado de la herramienta")
print("   4. REPEAT o FINAL ANSWER: Continúa razonando o entrega la respuesta")
print("\n   Esta transparencia es fundamental para depurar y auditar el agente.")
```

**Salida Esperada (parcial):**

```
═════════════════════════════════════════════════════════════════
PREGUNTA 1: ¿Qué es el aprendizaje supervisado y cuáles son sus algoritmos principales?
═════════════════════════════════════════════════════════════════

> Entering new AgentExecutor chain...
Thought: Esta pregunta es sobre machine learning, específicamente sobre aprendizaje supervisado. Debo buscar en la base de conocimiento local.
Action: buscar_base_conocimiento
Action Input: aprendizaje supervisado algoritmos principales
Observation: Información de la base de conocimiento:
El aprendizaje supervisado es un tipo de machine learning donde el algoritmo aprende de datos etiquetados...
Thought: Tengo información suficiente para responder.
Final Answer: El aprendizaje supervisado es un tipo de machine learning...

─────────────────────────────────────────────────────────────────
✅ RESPUESTA FINAL:
─────────────────────────────────────────────────────────────────
El aprendizaje supervisado es un tipo de machine learning donde el algoritmo aprende de datos etiquetados...

🔧 Herramientas utilizadas: ['buscar_base_conocimiento']
```

```
REPORTE FINAL DE EVALUACIÓN DEL AGENTE
═════════════════════════════════════════════════════════════════

📊 Estadísticas de Ejecución:
   Total preguntas evaluadas: 10
   Preguntas que usaron herramientas: 10/10
   Respuestas directas (sin herramienta): 0

🔧 Uso de Herramientas:
   - buscar_base_conocimiento: 5 invocación(es)
   - calculadora_matematica: 4 invocación(es)
   - buscar_wikipedia: 2 invocación(es)
```

**Verificación:**

- El agente ejecuta las 10 preguntas sin errores fatales
- Las preguntas matemáticas activan `calculadora_matematica`
- Las preguntas sobre IA activan `buscar_base_conocimiento`
- Las preguntas sobre personas/empresas activan `buscar_wikipedia`
- El reporte final muestra el conteo de uso de cada herramienta

---

### Paso 7: Guardar el Notebook y Documentar Observaciones

**Objetivo:** Guardar el trabajo realizado, documentar las observaciones clave sobre el comportamiento del agente y preparar el código para su reutilización en el Laboratorio 3.

**Instrucciones:**

1. Agrega una celda final de documentación en el notebook:

**Celda 17 – Documentación y observaciones finales:**

```python
# Celda 17: Documentación de observaciones y próximos pasos
observaciones = """
╔══════════════════════════════════════════════════════════════╗
║           OBSERVACIONES DEL LABORATORIO 2                   ║
╚══════════════════════════════════════════════════════════════╝

1. DIFERENCIA CADENA vs AGENTE:
    - La cadena LCEL siempre ejecuta el mismo flujo: prompt → LLM → parser
    - El agente ReAct razona dinámicamente: analiza la pregunta y decide
    qué herramienta usar (o si responder directamente)
    - Para cálculos matemáticos, el agente obtiene resultados exactos
    mientras que la cadena puede alucinar el resultado

2. IMPORTANCIA DE LAS DESCRIPCIONES DE HERRAMIENTAS:
    - El LLM usa SOLO las descripciones para decidir qué herramienta usar
    - Descripciones vagas → selección incorrecta de herramientas
    - Descripciones precisas con ejemplos → selección correcta

3. PATRÓN ReAct (Thought → Action → Observation):
    - Proporciona transparencia total del razonamiento del agente
    - Permite identificar errores en la lógica de selección
    - Facilita la depuración cuando el agente selecciona la herramienta incorrecta

4. ÍNDICE RAG con LlamaIndex + FAISS:
    - Los documentos se fragmentan en chunks de 512 tokens
    - Cada chunk se vectoriza con text-embedding-3-small
    - La búsqueda por similitud semántica recupera los 3 chunks más relevantes
    - El LLM sintetiza la respuesta basándose en esos chunks

5. CONSIDERACIONES ÉTICAS OBSERVADAS:
    - El agente cita la fuente (Wikipedia URL) cuando usa datos externos
    - La base de conocimiento RAG limita las respuestas a documentos verificados
    - El patrón verbose permite auditar cada decisión del agente
    - max_iterations=8 previene bucles infinitos y gastos excesivos de API

PRÓXIMOS PASOS (Laboratorio 3):
- Integrar APIs externas reales (OpenWeatherMap)
- Implementar herramienta de envío de correos
- Aplicar el Model Context Protocol (MCP)
- Usar ChromaDB como alternativa a FAISS para persistencia
"""

print(observaciones)

# Guardar un resumen en archivo de texto
with open("observaciones_lab02.txt", "w", encoding="utf-8") as f:
    f.write(observaciones)
print("\n✅ Observaciones guardadas en 'observaciones_lab02.txt'")
```

2. Guarda el notebook desde Jupyter (`File → Save and Checkpoint` o `Ctrl+S`)

3. En la parte superior de Jupyter selecciona **File -> New -> Terminal**, se abrirá una nueva pestaña con una terminal, envía y ejecuta este comando para exportar el notebook a script Python para reutilización:

```powershell
jupyter nbconvert --to script lab_02_cadenas_vs_agentes.ipynb --output agente_qa_completo
```

4. En la misma Terminal verifica la estructura final del proyecto:

```powershell
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch 'venv_lab02|\.git' } | Sort-Object FullName
```

**Salida Esperada:**

```
./agente_qa_completo.py
./documentos/deep_learning.txt
./documentos/etica_ia.txt
./documentos/ia_fundamentos.txt
./documentos/machine_learning.txt
./documentos/nlp_transformers.txt
./indice_faiss/default__vector_store.json
./indice_faiss/docstore.json
./indice_faiss/graph_store.json
./indice_faiss/image__vector_store.json
./indice_faiss/index_store.json
./lab_02_cadenas_vs_agentes.ipynb
./observaciones_lab02.txt
./.env
./.gitignore
```

**Verificación:**

- El archivo `agente_qa_completo.py` fue generado correctamente
- La carpeta `indice_faiss/` contiene los archivos de persistencia del índice
- El archivo `.env` NO aparece en commits de Git (verificar con `git status`)

## Solución de Problemas

### Problema 1: Error al instalar faiss-cpu en Windows

**Síntomas:**
- `ERROR: Could not build wheels for faiss-cpu`
- `error: Microsoft Visual C++ 14.0 or greater is required`
- El comando `pip install faiss-cpu` falla en Windows 10/11

**Causa:**
FAISS requiere compilación de código C++ en Windows, lo que necesita las herramientas de compilación de Microsoft Visual C++.

**Solución:**
```powershell
# Opción 1: Instalar Microsoft C++ Build Tools
# Descargar desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Instalar con la carga de trabajo "Desarrollo de escritorio con C++"

# Opción 2: Usar wheel precompilado (más rápido)
pip install faiss-cpu --only-binary :all:

# Opción 3: Instalar versión específica con wheel disponible
pip install faiss-cpu==1.7.4 --only-binary :all:

# Verificar instalación
python -c "import faiss; print(f'FAISS versión: {faiss.__version__}')"
```

---

### Problema 2: Error "ModuleNotFoundError: No module named 'llama_index.embeddings.openai'"

**Síntomas:**
- `ImportError: cannot import name 'AzureOpenAIEmbedding' from 'llama_index.embeddings.openai'`
- `ModuleNotFoundError: No module named 'llama_index.embeddings.openai'`

**Causa:**
LlamaIndex 0.10.x tiene una estructura de paquetes modular. Los embeddings de OpenAI están en un paquete separado que debe instalarse explícitamente.

**Solución:**
```powershell
# Instalar el paquete de embeddings OpenAI para LlamaIndex
pip install llama-index-embeddings-openai

# Instalar el paquete de vector store FAISS para LlamaIndex
pip install llama-index-vector-stores-faiss

# Instalar el LLM OpenAI para LlamaIndex
pip install llama-index-llms-openai

# Verificar importaciones
python -c "
from llama_index.embeddings.openai import AzureOpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import AzureOpenAI
print('Todas las importaciones de LlamaIndex Azure OK')
"
```

---

### Problema 3: El agente entra en bucle infinito o excede max_iterations

**Síntomas:**
- El agente imprime múltiples ciclos `Thought → Action → Observation` sin llegar a `Final Answer`
- Error: `Agent stopped due to iteration limit or time limit`
- El agente llama a la misma herramienta repetidamente con las mismas entradas

**Causa:**
El LLM no puede sintetizar una respuesta final a partir de los resultados de las herramientas, posiblemente porque la descripción de la herramienta es confusa o la pregunta es ambigua.

**Solución:**
```python
# 1. Reducir temperatura del LLM (más determinista)
llm_agente = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
)

# 2. Aumentar max_iterations temporalmente para diagnóstico
ejecutor_qa = AgentExecutor(
    agent=agente_qa,
    tools=herramientas_agente,
    verbose=True,
    max_iterations=3,  # Reducir para ver dónde falla
    handle_parsing_errors=True,
    early_stopping_method="generate"  # Forzar respuesta al alcanzar el límite
)

# 3. Mejorar las descripciones de las herramientas si el agente las confunde
# Ejemplo: ser más específico sobre cuándo NO usar cada herramienta
@tool
def buscar_base_conocimiento(consulta: str) -> str:
    """Busca en la base de conocimiento LOCAL sobre IA.
    USA ESTA HERRAMIENTA SOLO para: machine learning, deep learning, NLP, transformers, ética IA.
    NO uses esta herramienta para: cálculos matemáticos, personas históricas, noticias actuales."""
    # ... implementación
```

---

### Problema 4: Error de autenticación de OpenAI (AuthenticationError)

**Síntomas:**
- `openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}`
- `openai.AuthenticationError: No API key provided`

**Causa:**
La API key de OpenAI no está configurada correctamente en el archivo `.env` o no fue cargada antes de inicializar los clientes.

**Solución:**
```python
# 1. Verificar que .env existe y tiene el formato correcto
import os

# Verificar existencia del archivo
if not os.path.exists(".env"):
    print("ERROR: Archivo .env no encontrado en el directorio actual")
    print(f"Directorio actual: {os.getcwd()}")
else:
    print("Archivo .env encontrado")

# 2. Cargar explícitamente con override=True
from dotenv import load_dotenv
load_dotenv(override=True)

# 3. Verificar que la variable fue cargada
api_key = os.getenv("AZURE_OPENAI_API_KEY")
print(f"API Key: {'configurada' if api_key and api_key != 'sk-tu-api-key-aqui' else 'NO configurada'}")

# 4. Si ejecutas desde Jupyter, reinicia el kernel y ejecuta load_dotenv() primero
```

```powershell
# Verificar formato del archivo .env (debe ser exactamente así, sin espacios)
Get-Content .env
# Salida esperada: AZURE_OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

---

### Problema 5: Wikipedia retorna DisambiguationError para todos los términos

**Síntomas:**
- `wikipedia.exceptions.DisambiguationError: "Python" may refer to: Python (programming language), Python (genus)...`
- La herramienta `buscar_wikipedia` siempre retorna errores de ambigüedad

**Causa:**
Muchos términos en Wikipedia tienen múltiples páginas posibles. El wrapper de Python de Wikipedia no siempre resuelve automáticamente la ambigüedad.

**Solución:**
```python
# Actualizar la herramienta con manejo mejorado de ambigüedad
@tool
def buscar_wikipedia(termino: str) -> str:
    """Busca información en Wikipedia sobre cualquier tema general."""
    import wikipedia
    wikipedia.set_lang("es")
    
    try:
        # Intentar búsqueda directa primero
        resumen = wikipedia.summary(termino, sentences=4, auto_suggest=True)
        return f"Wikipedia:\n{resumen}"
    except wikipedia.exceptions.DisambiguationError as e:
        # Tomar la primera opción de desambiguación
        try:
            primera = e.options[0]
            resumen = wikipedia.summary(primera, sentences=4, auto_suggest=False)
            return f"Wikipedia - {primera}:\n{resumen}"
        except Exception:
            # Intentar en inglés como fallback
            wikipedia.set_lang("en")
            try:
                resumen = wikipedia.summary(termino, sentences=4, auto_suggest=True)
                wikipedia.set_lang("es")
                return f"Wikipedia (en inglés):\n{resumen}"
            except Exception as e2:
                return f"No se pudo obtener información de Wikipedia para '{termino}': {str(e2)}"
    except Exception as e:
        return f"Error al consultar Wikipedia: {str(e)}"
```

## Limpieza

```powershell
# Desactivar el entorno virtual
deactivate

# Opcional: Eliminar el entorno virtual (libera ~500MB)
# ADVERTENCIA: Solo si ya no necesitas el laboratorio
# Remove-Item -Recurse -Force venv_lab02

# Opcional: Eliminar el índice FAISS (se puede regenerar)
# Remove-Item -Recurse -Force indice_faiss

# Verificar que .env no está en staging de Git
git status
# El archivo .env NO debe aparecer en la lista

# Si accidentalmente agregaste .env a Git, eliminarlo del tracking
# git rm --cached .env
# git commit -m "Eliminar .env del tracking de Git"

# Comprimir el laboratorio para respaldo (excluyendo entorno virtual e índice)
cd ..
zip -r lab_02_backup.zip lab_02_00_01/ \
    --exclude "lab_02_00_01/venv_lab02/*" \
    --exclude "lab_02_00_01/.env" \
    --exclude "lab_02_00_01/__pycache__/*"
echo "Respaldo creado: lab_02_backup.zip"
```

> ⚠️ **Advertencia:** El archivo `.env` contiene tu API key de OpenAI. Nunca lo incluyas en commits de Git, correos electrónicos, screenshots o repositorios públicos. Verifica siempre que `.gitignore` lo excluya antes de cualquier operación de Git. Si accidentalmente expusiste tu API key, revócala inmediatamente en [platform.openai.com/api-keys](https://platform.openai.com/api-keys) y genera una nueva.

> ⚠️ **Costos de API:** Este laboratorio consume aproximadamente $0.30–$0.80 USD en llamadas a la API de OpenAI (embeddings + completions). Si ejecutas el banco de 10 preguntas múltiples veces, el costo se multiplica. Monitorea tu uso en [platform.openai.com/usage](https://platform.openai.com/usage).

## Resumen

### Lo que Lograste

- Creaste una base de conocimiento local con 5 documentos sobre inteligencia artificial y la indexaste con LlamaIndex + FAISS para búsqueda semántica eficiente
- Implementaste y comparaste dos enfoques distintos: una cadena LCEL con flujo fijo y un agente ReAct con razonamiento dinámico, observando sus diferencias en código ejecutable
- Definiste 3 herramientas especializadas usando el decorador `@tool`: búsqueda RAG en documentos locales, calculadora matemática segura y búsqueda en Wikipedia
- Ensamblaste un agente de QA completo con `create_react_agent` y `AgentExecutor` capaz de seleccionar automáticamente la herramienta correcta para cada tipo de pregunta
- Evaluaste el agente con un banco de 10 preguntas variadas, interpretando el razonamiento paso a paso (chain-of-thought) en los logs de ejecución

### Conceptos Clave Aprendidos

- **Cadenas vs Agentes:** Las cadenas ejecutan flujos fijos y predecibles; los agentes razonan dinámicamente usando el patrón ReAct (Thought → Action → Observation → Final Answer)
- **Importancia de las descripciones de herramientas:** El LLM usa exclusivamente las descripciones para decidir qué herramienta invocar; descripciones precisas son fundamentales para el correcto funcionamiento del agente
- **RAG con LlamaIndex + FAISS:** Los documentos se fragmentan, vectorizan y almacenan en un índice local; la búsqueda semántica recupera los fragmentos más relevantes para cada consulta
- **Transparencia del patrón ReAct:** El modo `verbose=True` expone el razonamiento completo del agente, facilitando la depuración, auditoría y comprensión de sus decisiones
- **Seguridad y límites:** `max_iterations` previene bucles infinitos, el namespace seguro en la calculadora previene ejecución de código malicioso, y `.env` + `.gitignore` protegen las credenciales

### Próximos Pasos

- **Laboratorio 3:** Integrar APIs externas reales (OpenWeatherMap) y herramientas de comunicación (correo electrónico) al agente, aplicando el Model Context Protocol (MCP) y explorando ChromaDB como alternativa a FAISS
- **Exploración adicional:** Experimenta con diferentes valores de `chunk_size` y `similarity_top_k` en el índice RAG para observar cómo afectan la calidad de las respuestas
- **Mejora de herramientas:** Agrega una cuarta herramienta personalizada (por ejemplo, búsqueda en arXiv o consulta a una API REST pública) para practicar la extensibilidad del agente

## Recursos Adicionales

- [Documentación oficial de LangChain – Agentes ReAct](https://python.langchain.com/docs/modules/agents/) - Guía completa sobre tipos de agentes, ejecutores y configuración avanzada del patrón ReAct
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) - Tutorial oficial para construir cadenas modernas con el operador pipe y sus ventajas sobre enfoques anteriores
- [Documentación de LlamaIndex – VectorStoreIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/) - Referencia completa para construcción de índices vectoriales, configuración de chunks y motores de consulta
- [FAISS – Facebook AI Similarity Search](https://faiss.ai/index.html) - Documentación técnica del motor de búsqueda vectorial usado en este laboratorio
- [Paper ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) - Paper original de Yao et al. (2022) que introduce el patrón ReAct en el que se basa el agente de este laboratorio
- [OpenAI Embeddings – text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings) - Documentación del modelo de embeddings usado para vectorizar los documentos de la base de conocimiento
- [Decorador @tool de LangChain](https://python.langchain.com/docs/modules/tools/) - Referencia sobre cómo definir, registrar y optimizar herramientas para agentes en LangChain
