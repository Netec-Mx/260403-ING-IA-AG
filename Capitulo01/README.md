# Construir un Agente Básico con LangChain (Chatbot Simple de Preguntas y Respuestas)

## Metadatos

| Propiedad | Valor |
|-----------|-------|
| **Duración** | 65 minutos |
| **Complejidad** | Fácil |
| **Nivel Bloom** | Aplicar |
| **Laboratorio** | 1 de 4 |

## Descripción General

En este laboratorio construirás desde cero un agente conversacional básico utilizando LangChain, pasando por cada uno de los componentes fundamentales que conforman un agente de IA: el modelo de lenguaje, el template de prompt, la memoria y la cadena de orquestación. Comenzarás con un chatbot simple sin memoria y progresivamente incorporarás `ConversationBufferMemory` para mantener el contexto entre turnos de conversación.

Este laboratorio tiene un valor práctico inmediato: al finalizar, comprenderás la diferencia entre un modelo de lenguaje estático y un agente conversacional real, y habrás experimentado de primera mano cómo el diseño del prompt afecta directamente la calidad de las respuestas. Estas habilidades son la base para construir sistemas de IA más sofisticados como los agentes con herramientas externas y RAG que se desarrollarán en los laboratorios siguientes.

## Objetivos de Aprendizaje

Al completar este laboratorio, serás capaz de:

- [ ] Identificar y describir los componentes fundamentales de un agente de IA (LLM, prompt template, memoria, cadena de ejecución) y su rol en el flujo de procesamiento
- [ ] Configurar correctamente el entorno de desarrollo con LangChain y sus dependencias, incluyendo la autenticación segura con la API de OpenAI mediante archivos `.env`
- [ ] Construir un chatbot funcional de preguntas y respuestas utilizando `ChatOpenAI` y `ChatPromptTemplate` de LangChain
- [ ] Diseñar y aplicar prompts eficaces con instrucciones claras, roles definidos (`system`, `human`, `ai`) y contexto adecuado para mejorar la calidad de las respuestas
- [ ] Implementar memoria conversacional básica con `ConversationBufferMemory` para mantener el contexto entre turnos de conversación
- [ ] Evaluar y comparar el comportamiento del agente con diferentes configuraciones de prompts, documentando las diferencias observadas

## Prerrequisitos

### Conocimientos Requeridos

- Programación en Python 3.10+: funciones, clases, manejo de excepciones y uso de librerías externas con `pip`
- Conceptos básicos de IA y modelos de lenguaje: qué es un LLM, tokens y parámetro de temperatura
- Comprensión básica de APIs REST y cómo funcionan las API keys para autenticación
- Manejo básico de la terminal/línea de comandos para instalar paquetes y ejecutar scripts
- Conocimiento del formato JSON para interpretar respuestas de APIs

### Acceso Requerido

- Cuenta activa en OpenAI con créditos disponibles (estimado: $0.10–$0.50 USD para este laboratorio usando `gpt-3.5-turbo`)
- Acceso a Internet estable para consumir la API de OpenAI y descargar paquetes
- Permisos de escritura en el sistema de archivos local para crear el entorno virtual y archivos de configuración

> ⚠️ **Nota de Costos:** Este laboratorio utiliza la API de OpenAI, lo que genera costos reales. Se estima un gasto de $0.10–$0.50 USD usando `gpt-3.5-turbo`. Configura un límite de gasto en tu cuenta de OpenAI antes de comenzar: [https://platform.openai.com/account/limits](https://platform.openai.com/account/limits)

> 💡 **Alternativa Gratuita:** Si no tienes acceso a OpenAI, puedes usar Ollama con `llama3.2` de forma local. El instructor proveerá las instrucciones de configuración alternativa. Requiere mínimo 16 GB de RAM.

## Entorno de Laboratorio

### Requisitos de Hardware

| Componente | Especificación |
|-----------|----------------|
| Procesador | CPU de 64 bits, mínimo 4 núcleos |
| Memoria RAM | Mínimo 8 GB |
| Almacenamiento | Mínimo 2 GB libres para este laboratorio |
| Conexión a Internet | Mínimo 10 Mbps para consumo de la API de OpenAI |

### Requisitos de Software

| Software | Versión | Propósito |
|----------|---------|-----------|
| Python | 3.10 o 3.11 | Lenguaje principal del laboratorio |
| pip | 23.x o superior | Gestión de paquetes |
| venv | Incluido en Python 3.10+ | Aislamiento del entorno |
| langchain | 0.2.x o superior | Framework principal de orquestación |
| langchain-openai | 0.1.x o superior | Integración con modelos OpenAI |
| python-dotenv | 1.0.x | Gestión segura de variables de entorno |
| Jupyter Notebook | 7.x o JupyterLab 4.x | Entorno interactivo de desarrollo |
| Visual Studio Code | 1.85 o superior | IDE recomendado (opcional) |

### Configuración Inicial

```bash
# 1. Crear el directorio del laboratorio
mkdir lab01_agente_basico
cd lab01_agente_basico

# 2. Crear el entorno virtual
python3 -m venv venv

# 3. Activar el entorno virtual
# En Linux/macOS:
source venv/bin/activate
# En Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
# En Windows (CMD):
# venv\Scripts\activate.bat

# 4. Actualizar pip
pip install --upgrade pip

# 5. Instalar las dependencias del laboratorio
pip install langchain==0.2.16 langchain-openai==0.1.25 langchain-core==0.2.41 python-dotenv==1.0.1 jupyter==1.0.0 notebook==7.2.2

# 6. Verificar la instalación
pip list | grep -E "langchain|openai|dotenv|jupyter"
```

## Instrucciones Paso a Paso

### Paso 1: Configurar el Entorno y Gestionar Credenciales de Forma Segura

**Objetivo:** Crear la estructura del proyecto, configurar las variables de entorno con la API key de OpenAI de forma segura y verificar que LangChain está correctamente instalado.

**Instrucciones:**

1. Asegúrate de estar dentro del directorio `lab01_agente_basico` con el entorno virtual activado. Crea la estructura de archivos del proyecto:

   ```bash
   # Crear archivos necesarios del proyecto
   touch .env
   touch .gitignore
   touch agente_basico.ipynb
   ```

2. Configura el archivo `.gitignore` para proteger tus credenciales. Este paso es **obligatorio** antes de escribir cualquier API key:

   ```bash
   cat > .gitignore << 'EOF'
   # Variables de entorno y credenciales - NUNCA subir al repositorio
   .env
   .env.local
   .env.*.local

   # Entorno virtual
   venv/
   __pycache__/
   *.pyc
   *.pyo

   # Jupyter checkpoints
   .ipynb_checkpoints/

   # Archivos del sistema operativo
   .DS_Store
   Thumbs.db
   EOF
   ```

3. Abre el archivo `.env` con tu editor de texto preferido y agrega tu API key de OpenAI. **Nunca escribas la API key directamente en el código Python o en el notebook:**

   ```bash
   # Abrir el archivo .env con nano (Linux/macOS)
   nano .env
   ```

   Escribe el siguiente contenido en el archivo `.env` (reemplaza `tu_api_key_aqui` con tu clave real de OpenAI):

   ```
   OPENAI_API_KEY=tu_api_key_aqui
   OPENAI_MODEL=gpt-3.5-turbo
   ```

   Guarda el archivo con `Ctrl+O` y cierra con `Ctrl+X`.

4. Verifica que el archivo `.env` fue creado correctamente sin mostrar el contenido completo de la API key:

   ```bash
   # Verificar que el archivo existe y tiene contenido
   ls -la .env
   
   # Verificar que las variables están definidas (sin mostrar el valor completo)
   grep -c "OPENAI_API_KEY" .env
   ```

5. Inicia Jupyter Notebook para trabajar de forma interactiva:

   ```bash
   jupyter notebook agente_basico.ipynb
   ```

**Salida Esperada:**

```
-rw-r--r-- 1 usuario grupo 52 fecha .env
1
[I 2024-01-15 10:00:00.000 ServerApp] Jupyter Server 2.x.x is running at:
[I 2024-01-15 10:00:00.000 ServerApp] http://localhost:8888/tree
```

**Verificación:**

- El archivo `.env` existe y tiene al menos 1 línea con `OPENAI_API_KEY`
- El archivo `.gitignore` contiene la entrada `.env`
- Jupyter Notebook se abre en el navegador sin errores

---

### Paso 2: Verificar la Instalación de LangChain y Cargar Variables de Entorno

**Objetivo:** Confirmar que todas las dependencias están instaladas correctamente y que la API key de OpenAI se carga desde el archivo `.env` sin exponerla en el código.

**Instrucciones:**

1. En el notebook de Jupyter, crea una nueva celda y escribe el siguiente código para verificar las versiones instaladas:

   ```python
   # Celda 1: Verificación de versiones
   import langchain
   import langchain_openai
   import dotenv
   
   print(f"LangChain versión: {langchain.__version__}")
   print(f"LangChain-OpenAI versión: {langchain_openai.__version__}")
   print(f"Python-dotenv versión: {dotenv.__version__}")
   print("✅ Todas las dependencias están instaladas correctamente")
   ```

2. Crea una nueva celda para cargar las variables de entorno y verificar que la API key está disponible sin mostrarla completa:

   ```python
   # Celda 2: Cargar variables de entorno de forma segura
   import os
   from dotenv import load_dotenv
   
   # Cargar variables desde el archivo .env
   load_dotenv()
   
   # Verificar que la API key está disponible sin mostrarla completa
   api_key = os.getenv("OPENAI_API_KEY")
   modelo = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
   
   if api_key:
       # Mostrar solo los primeros y últimos 4 caracteres para verificar
       key_preview = f"{api_key[:4]}...{api_key[-4:]}"
       print(f"✅ API Key cargada correctamente: {key_preview}")
       print(f"✅ Modelo configurado: {modelo}")
   else:
       print("❌ ERROR: No se encontró OPENAI_API_KEY en el archivo .env")
       print("Verifica que el archivo .env existe y contiene la variable OPENAI_API_KEY")
   ```

3. Ejecuta ambas celdas presionando `Shift+Enter` en cada una.

**Salida Esperada:**

```
LangChain versión: 0.2.16
LangChain-OpenAI versión: 0.1.25
Python-dotenv versión: 1.0.1
✅ Todas las dependencias están instaladas correctamente

✅ API Key cargada correctamente: sk-p...xK3m
✅ Modelo configurado: gpt-3.5-turbo
```

**Verificación:**

- Las versiones de LangChain son 0.2.x o superior
- La API key muestra los primeros y últimos 4 caracteres (no el valor completo)
- El modelo está configurado como `gpt-3.5-turbo` o el que hayas definido

---

### Paso 3: Crear tu Primer Chatbot Simple (Sin Memoria)

**Objetivo:** Construir un chatbot funcional básico usando `ChatOpenAI` y `ChatPromptTemplate`, comprendiendo cómo el LLM y el prompt template trabajan juntos como los primeros dos componentes del agente.

**Instrucciones:**

1. Crea una nueva celda en el notebook e importa los componentes necesarios de LangChain:

   ```python
   # Celda 3: Importar componentes de LangChain
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_core.output_parsers import StrOutputParser
   
   print("✅ Componentes de LangChain importados correctamente")
   ```

2. Crea una nueva celda para inicializar el modelo de lenguaje. Este es el **cerebro** del agente:

   ```python
   # Celda 4: Inicializar el modelo de lenguaje (el "cerebro" del agente)
   llm = ChatOpenAI(
       model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
       temperature=0.7,        # Creatividad de las respuestas (0=determinista, 1=creativo)
       max_tokens=500,         # Límite de tokens en la respuesta
       api_key=os.getenv("OPENAI_API_KEY")
   )
   
   print(f"✅ Modelo inicializado: {llm.model_name}")
   print(f"   Temperatura: {llm.temperature}")
   print(f"   Máximo de tokens: {llm.max_tokens}")
   ```

3. Crea una nueva celda para diseñar el **prompt template**. Este es el componente que define cómo el agente recibe y estructura la información:

   ```python
   # Celda 5: Diseñar el Prompt Template (instrucciones del agente)
   # El ChatPromptTemplate define los roles de la conversación:
   # - "system": instrucciones permanentes que definen el comportamiento del agente
   # - "human": el mensaje del usuario en cada turno
   
   prompt_template = ChatPromptTemplate.from_messages([
       (
           "system",
           """Eres un asistente de IA amable y profesional especializado en responder 
           preguntas sobre tecnología y programación. 
           
           Tus características:
           - Respondes siempre en español
           - Eres conciso pero completo en tus explicaciones
           - Cuando no sabes algo, lo dices claramente en lugar de inventar
           - Usas ejemplos prácticos para explicar conceptos complejos
           - Siempre mantienes un tono profesional y respetuoso"""
       ),
       ("human", "{pregunta}")
   ])
   
   print("✅ Prompt template creado con rol de sistema definido")
   print(f"   Mensajes en el template: {len(prompt_template.messages)}")
   ```

4. Crea una nueva celda para construir la **cadena de ejecución** (chain) conectando el prompt, el LLM y el parser de salida:

   ```python
   # Celda 6: Construir la cadena de ejecución (el "orquestador" básico)
   # La cadena conecta: prompt_template → llm → output_parser
   # El operador | (pipe) es la sintaxis LCEL (LangChain Expression Language)
   
   output_parser = StrOutputParser()
   
   cadena_simple = prompt_template | llm | output_parser
   
   print("✅ Cadena de ejecución construida:")
   print("   ChatPromptTemplate → ChatOpenAI → StrOutputParser")
   ```

5. Crea una nueva celda para probar el chatbot con tu primera pregunta:

   ```python
   # Celda 7: Probar el chatbot con la primera pregunta
   print("=" * 60)
   print("CHATBOT BÁSICO - PRUEBA 1")
   print("=" * 60)
   
   pregunta_1 = "¿Qué es un agente de IA y en qué se diferencia de un chatbot simple?"
   
   print(f"👤 Usuario: {pregunta_1}")
   print()
   
   respuesta_1 = cadena_simple.invoke({"pregunta": pregunta_1})
   
   print(f"🤖 Agente: {respuesta_1}")
   print()
   print("=" * 60)
   ```

6. Prueba con una segunda pregunta para observar que el chatbot **no recuerda** la conversación anterior:

   ```python
   # Celda 8: Probar la segunda pregunta (observar falta de memoria)
   print("=" * 60)
   print("CHATBOT BÁSICO - PRUEBA 2 (Sin memoria)")
   print("=" * 60)
   
   pregunta_2 = "¿Puedes darme un ejemplo de lo que acabas de explicar?"
   
   print(f"👤 Usuario: {pregunta_2}")
   print()
   
   respuesta_2 = cadena_simple.invoke({"pregunta": pregunta_2})
   
   print(f"🤖 Agente: {respuesta_2}")
   print()
   print("⚠️  OBSERVACIÓN: El agente no recuerda la pregunta anterior.")
   print("   Esto se debe a que no tiene memoria configurada.")
   print("=" * 60)
   ```

**Salida Esperada:**

```
============================================================
CHATBOT BÁSICO - PRUEBA 1
============================================================
👤 Usuario: ¿Qué es un agente de IA y en qué se diferencia de un chatbot simple?

🤖 Agente: Un agente de IA es un sistema autónomo que puede percibir su entorno, 
razonar sobre él y ejecutar acciones para cumplir objetivos específicos. 

A diferencia de un chatbot simple que solo responde a preguntas con respuestas 
predefinidas o generadas por un LLM, un agente puede:

1. **Tomar decisiones** sobre qué herramientas usar
2. **Ejecutar acciones** como consultar APIs o bases de datos
3. **Recordar contexto** de interacciones anteriores
4. **Planificar pasos** para resolver tareas complejas

Por ejemplo, si le preguntas a un chatbot simple "¿Cuál es el clima en Madrid?", 
te dará información general. Un agente, en cambio, consultaría una API meteorológica 
real y te daría el clima actual.
============================================================
```

**Verificación:**

- La respuesta está en español y es coherente con la pregunta
- No aparecen errores de autenticación (`AuthenticationError`)
- La segunda respuesta no hace referencia al contenido de la primera pregunta (confirma la ausencia de memoria)

---

### Paso 4: Implementar Memoria Conversacional con ConversationBufferMemory

**Objetivo:** Incorporar `ConversationBufferMemory` al agente para que mantenga el contexto entre turnos de conversación, transformando el chatbot simple en un agente conversacional real.

**Instrucciones:**

1. Crea una nueva celda e importa los componentes de memoria de LangChain:

   ```python
   # Celda 9: Importar componentes de memoria
   from langchain.memory import ConversationBufferMemory
   from langchain.chains import ConversationChain
   from langchain_core.prompts import PromptTemplate
   
   print("✅ Componentes de memoria importados correctamente")
   ```

2. Crea una nueva celda para inicializar la memoria y construir el agente con memoria:

   ```python
   # Celda 10: Configurar la memoria conversacional
   # ConversationBufferMemory almacena TODOS los turnos de conversación en memoria RAM
   # memory_key="history" define el nombre de la variable en el prompt
   # return_messages=False devuelve el historial como texto plano
   
   memoria = ConversationBufferMemory(
       memory_key="history",
       return_messages=False
   )
   
   print("✅ Memoria conversacional inicializada")
   print(f"   Tipo de memoria: ConversationBufferMemory")
   print(f"   Clave de memoria: {memoria.memory_key}")
   print(f"   Historial inicial: '{memoria.load_memory_variables({})['history']}'")
   ```

3. Crea el prompt template que incluye el historial de conversación como variable:

   ```python
   # Celda 11: Crear prompt template con soporte para historial
   # La variable {history} será reemplazada automáticamente por el contenido de la memoria
   # La variable {input} contiene el mensaje actual del usuario
   
   prompt_con_memoria = PromptTemplate(
       input_variables=["history", "input"],
       template="""Eres un asistente de IA amable y profesional especializado en 
   tecnología y programación.
   
   Tus características:
   - Respondes siempre en español
   - Eres conciso pero completo en tus explicaciones  
   - Cuando no sabes algo, lo dices claramente
   - Usas ejemplos prácticos para explicar conceptos complejos
   - Recuerdas y referencias el contexto de la conversación cuando es relevante
   
   Historial de conversación:
   {history}
   
   Usuario: {input}
   Asistente:"""
   )
   
   print("✅ Prompt template con memoria creado")
   print(f"   Variables del template: {prompt_con_memoria.input_variables}")
   ```

4. Construye el agente conversacional con memoria usando `ConversationChain`:

   ```python
   # Celda 12: Construir el agente con memoria (ConversationChain)
   # ConversationChain orquesta automáticamente: memoria → prompt → LLM → respuesta → memoria
   
   agente_con_memoria = ConversationChain(
       llm=llm,
       memory=memoria,
       prompt=prompt_con_memoria,
       verbose=False  # Cambiar a True para ver el prompt completo en cada llamada
   )
   
   print("✅ Agente conversacional con memoria construido")
   print("   Componentes:")
   print("   1. LLM (cerebro): ChatOpenAI gpt-3.5-turbo")
   print("   2. Memoria: ConversationBufferMemory")
   print("   3. Prompt: Template con historial")
   print("   4. Orquestador: ConversationChain")
   ```

5. Prueba el agente con memoria usando una conversación de múltiples turnos:

   ```python
   # Celda 13: Probar el agente con memoria - Turno 1
   print("=" * 60)
   print("AGENTE CON MEMORIA - CONVERSACIÓN MULTI-TURNO")
   print("=" * 60)
   
   # Turno 1: Pregunta inicial
   turno_1 = "¿Cuáles son los cuatro componentes básicos de un agente de IA?"
   print(f"\n[Turno 1]")
   print(f"👤 Usuario: {turno_1}")
   
   respuesta_turno_1 = agente_con_memoria.predict(input=turno_1)
   print(f"🤖 Agente: {respuesta_turno_1}")
   ```

   ```python
   # Celda 14: Turno 2 - El agente debe recordar el turno anterior
   turno_2 = "¿Puedes profundizar en el segundo componente que mencionaste?"
   print(f"\n[Turno 2]")
   print(f"👤 Usuario: {turno_2}")
   
   respuesta_turno_2 = agente_con_memoria.predict(input=turno_2)
   print(f"🤖 Agente: {respuesta_turno_2}")
   ```

   ```python
   # Celda 15: Turno 3 - Referencia a contexto previo
   turno_3 = "Dame un ejemplo práctico de todo lo que hemos discutido hasta ahora"
   print(f"\n[Turno 3]")
   print(f"👤 Usuario: {turno_3}")
   
   respuesta_turno_3 = agente_con_memoria.predict(input=turno_3)
   print(f"🤖 Agente: {respuesta_turno_3}")
   print("\n" + "=" * 60)
   ```

6. Inspecciona el contenido de la memoria para entender cómo almacena el historial:

   ```python
   # Celda 16: Inspeccionar el contenido de la memoria
   print("\n" + "=" * 60)
   print("INSPECCIÓN DE LA MEMORIA DEL AGENTE")
   print("=" * 60)
   
   historial = memoria.load_memory_variables({})
   print(f"\nContenido completo de la memoria:")
   print("-" * 40)
   print(historial["history"])
   print("-" * 40)
   
   # Contar los turnos almacenados
   turnos = historial["history"].count("Human:")
   print(f"\n📊 Estadísticas de memoria:")
   print(f"   Turnos de conversación almacenados: {turnos}")
   print(f"   Caracteres en memoria: {len(historial['history'])}")
   ```

**Salida Esperada:**

```
============================================================
AGENTE CON MEMORIA - CONVERSACIÓN MULTI-TURNO
============================================================

[Turno 1]
👤 Usuario: ¿Cuáles son los cuatro componentes básicos de un agente de IA?
🤖 Agente: Los cuatro componentes básicos de un agente de IA son:

1. **Modelo de Lenguaje (LLM)**: El "cerebro" del agente que interpreta instrucciones y razona sobre problemas.
2. **Memoria**: Permite recordar información de conversaciones anteriores (corto y largo plazo).
3. **Herramientas**: Funciones externas que el agente puede invocar (APIs, buscadores, calculadoras).
4. **Orquestador**: Coordina el flujo entre LLM, memoria y herramientas.

[Turno 2]
👤 Usuario: ¿Puedes profundizar en el segundo componente que mencionaste?
🤖 Agente: Por supuesto. Hablando del segundo componente que mencioné, la **Memoria**, 
existen dos tipos principales:
- **Memoria a corto plazo**: El historial de la conversación actual...
[continúa referenciando el contexto previo]

[Turno 3]
👤 Usuario: Dame un ejemplo práctico de todo lo que hemos discutido hasta ahora
🤖 Agente: Basándome en nuestra conversación sobre los cuatro componentes...
[el agente integra información de los tres turnos anteriores]
```

**Verificación:**

- En el Turno 2, el agente hace referencia al "segundo componente" sin que el usuario lo especificara
- En el Turno 3, el agente menciona elementos de los turnos anteriores
- La inspección de memoria muestra el historial completo con los prefijos `Human:` y `AI:`

---

### Paso 5: Experimentar con Diseño de Prompts y Comparar Resultados

**Objetivo:** Diseñar y comparar diferentes configuraciones de prompts para observar cómo cada cambio afecta la calidad, tono y estructura de las respuestas del agente, aplicando principios de prompting efectivo.

**Instrucciones:**

1. Crea una nueva celda con una función auxiliar para facilitar las comparaciones:

   ```python
   # Celda 17: Función auxiliar para comparar prompts
   def comparar_prompts(pregunta, configuraciones):
       """
       Compara diferentes configuraciones de prompt para la misma pregunta.
       
       Args:
           pregunta (str): La pregunta a responder
           configuraciones (list): Lista de dicts con 'nombre' y 'system_message'
       """
       print("=" * 70)
       print(f"COMPARACIÓN DE PROMPTS")
       print(f"Pregunta: {pregunta}")
       print("=" * 70)
       
       for config in configuraciones:
           print(f"\n{'─' * 70}")
           print(f"📋 Configuración: {config['nombre']}")
           print(f"{'─' * 70}")
           
           # Crear prompt con esta configuración
           prompt_test = ChatPromptTemplate.from_messages([
               ("system", config['system_message']),
               ("human", "{pregunta}")
           ])
           
           # Crear cadena temporal
           cadena_test = prompt_test | llm | StrOutputParser()
           
           # Obtener respuesta
           respuesta = cadena_test.invoke({"pregunta": pregunta})
           
           print(f"🤖 Respuesta:\n{respuesta}")
       
       print("\n" + "=" * 70)
   
   print("✅ Función de comparación definida")
   ```

2. Ejecuta el experimento comparando tres configuraciones de prompt distintas:

   ```python
   # Celda 18: Experimento 1 - Comparar nivel de especificidad del prompt
   
   pregunta_experimento = "Explícame qué es la memoria en un agente de IA"
   
   configuraciones_especificidad = [
       {
           "nombre": "Prompt Mínimo (sin instrucciones específicas)",
           "system_message": "Eres un asistente útil."
       },
       {
           "nombre": "Prompt Intermedio (con rol y idioma)",
           "system_message": """Eres un instructor de programación. 
           Responde siempre en español y sé claro y conciso."""
       },
       {
           "nombre": "Prompt Detallado (con rol, formato y audiencia)",
           "system_message": """Eres un instructor senior de inteligencia artificial 
           con 10 años de experiencia enseñando a desarrolladores junior.
           
           Instrucciones:
           - Responde SIEMPRE en español
           - Adapta tu explicación para alguien con conocimientos básicos de Python
           - Estructura tu respuesta con: definición breve, cómo funciona, y ejemplo práctico
           - Usa analogías del mundo real para conceptos abstractos
           - Limita tu respuesta a máximo 200 palabras"""
       }
   ]
   
   comparar_prompts(pregunta_experimento, configuraciones_especificidad)
   ```

3. Realiza un segundo experimento comparando diferentes definiciones de rol:

   ```python
   # Celda 19: Experimento 2 - Comparar definición de roles
   
   pregunta_roles = "¿Debería aprender LangChain o construir mi propio framework de agentes?"
   
   configuraciones_roles = [
       {
           "nombre": "Sin rol definido",
           "system_message": "Responde la siguiente pregunta en español."
       },
       {
           "nombre": "Rol: Vendedor de tecnología",
           "system_message": """Eres un vendedor entusiasta de productos de IA. 
           Siempre recomiendas las últimas tecnologías y frameworks populares.
           Responde en español."""
       },
       {
           "nombre": "Rol: Arquitecto de software pragmático",
           "system_message": """Eres un arquitecto de software senior con 15 años de experiencia.
           Tu enfoque es pragmático: priorizas la solución más simple que funcione.
           Consideras siempre: tiempo de desarrollo, mantenibilidad y necesidades reales del proyecto.
           Responde en español con una perspectiva balanceada y honesta."""
       }
   ]
   
   comparar_prompts(pregunta_roles, configuraciones_roles)
   ```

4. Documenta tus observaciones en una celda de texto (Markdown):

   ```python
   # Celda 20: Documentar observaciones del experimento
   print("=" * 70)
   print("📝 REFLEXIÓN: ¿Qué observaste en los experimentos?")
   print("=" * 70)
   print("""
   Preguntas para reflexionar:
   
   1. ¿Cómo cambió la estructura de la respuesta entre el prompt mínimo y el detallado?
      → 
   
   2. ¿El prompt detallado siempre produce mejores resultados? ¿Cuándo podría ser excesivo?
      → 
   
   3. ¿Cómo afectó la definición del rol a la perspectiva de las respuestas?
      → 
   
   4. ¿Qué elementos del prompt consideras más importantes para tu caso de uso?
      → 
   
   Principios aprendidos:
   ✅ Mayor especificidad en el prompt → respuestas más predecibles y estructuradas
   ✅ El rol del sistema define la "personalidad" y perspectiva del agente
   ✅ Las instrucciones de formato (longitud, estructura) mejoran la consistencia
   ✅ El contexto de la audiencia adapta el nivel de complejidad de las respuestas
   """)
   ```

**Salida Esperada:**

```
======================================================================
COMPARACIÓN DE PROMPTS
Pregunta: Explícame qué es la memoria en un agente de IA
======================================================================

──────────────────────────────────────────────────────────────────────
📋 Configuración: Prompt Mínimo (sin instrucciones específicas)
──────────────────────────────────────────────────────────────────────
🤖 Respuesta:
Memory in an AI agent refers to... [posiblemente en inglés, sin estructura]

──────────────────────────────────────────────────────────────────────
📋 Configuración: Prompt Intermedio (con rol y idioma)
──────────────────────────────────────────────────────────────────────
🤖 Respuesta:
La memoria en un agente de IA es el componente que permite... [en español, más estructurado]

──────────────────────────────────────────────────────────────────────
📋 Configuración: Prompt Detallado (con rol, formato y audiencia)
──────────────────────────────────────────────────────────────────────
🤖 Respuesta:
**Definición:** La memoria en un agente de IA es...
**Cómo funciona:** Imagina que la memoria es como...
**Ejemplo práctico:** En código Python con LangChain...
[Respuesta estructurada, en español, con analogía y ejemplo]
```

**Verificación:**

- Las tres configuraciones producen respuestas notablemente diferentes para la misma pregunta
- El prompt detallado genera una respuesta más estructurada y en el idioma especificado
- Los diferentes roles producen perspectivas claramente distintas en el Experimento 2

---

### Paso 6: Construir el Agente Final Integrado

**Objetivo:** Integrar todos los componentes aprendidos (LLM, prompt template optimizado, memoria) en un agente conversacional completo y funcional que demuestre los cuatro componentes fundamentales de un agente de IA.

**Instrucciones:**

1. Crea una nueva celda con el agente final completamente configurado:

   ```python
   # Celda 21: Agente Final Integrado - Todos los componentes
   from langchain.memory import ConversationBufferMemory
   from langchain.chains import ConversationChain
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import PromptTemplate
   import os
   
   # ─────────────────────────────────────────────
   # COMPONENTE 1: Modelo de Lenguaje (el "cerebro")
   # ─────────────────────────────────────────────
   llm_final = ChatOpenAI(
       model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
       temperature=0.7,
       max_tokens=600,
       api_key=os.getenv("OPENAI_API_KEY")
   )
   
   # ─────────────────────────────────────────────
   # COMPONENTE 2: Memoria (historial conversacional)
   # ─────────────────────────────────────────────
   memoria_final = ConversationBufferMemory(
       memory_key="history",
       return_messages=False,
       human_prefix="Usuario",
       ai_prefix="Asistente"
   )
   
   # ─────────────────────────────────────────────
   # COMPONENTE 3: Prompt Template (instrucciones del agente)
   # ─────────────────────────────────────────────
   prompt_final = PromptTemplate(
       input_variables=["history", "input"],
       template="""Eres TechBot, un asistente de IA especializado en tecnología y 
   programación, diseñado para ayudar a desarrolladores a aprender sobre agentes de IA.
   
   Características de TechBot:
   - Respondes SIEMPRE en español
   - Eres experto en LangChain, Python y arquitecturas de agentes de IA
   - Explicas conceptos complejos con analogías del mundo real
   - Cuando el usuario hace referencia a algo previo, lo incorporas en tu respuesta
   - Eres honesto: si no sabes algo, lo dices claramente
   - Tus respuestas son entre 100-300 palabras, estructuradas y fáciles de leer
   - Usas emojis ocasionalmente para hacer la conversación más amigable
   
   Historial de conversación:
   {history}
   
   Usuario: {input}
   Asistente:"""
   )
   
   # ─────────────────────────────────────────────
   # COMPONENTE 4: Orquestador (coordina el flujo)
   # ─────────────────────────────────────────────
   techbot = ConversationChain(
       llm=llm_final,
       memory=memoria_final,
       prompt=prompt_final,
       verbose=False
   )
   
   print("✅ TechBot - Agente Conversacional Final inicializado")
   print()
   print("Arquitectura del agente:")
   print("┌─────────────────────────────────────────────┐")
   print("│           TECHBOT - AGENTE DE IA            │")
   print("├─────────────────────────────────────────────┤")
   print("│ 🧠 LLM:         ChatOpenAI (gpt-3.5-turbo)  │")
   print("│ 💾 Memoria:     ConversationBufferMemory     │")
   print("│ 📋 Prompt:      Template personalizado       │")
   print("│ ⚙️  Orquestador: ConversationChain           │")
   print("└─────────────────────────────────────────────┘")
   ```

2. Implementa una función de chat interactivo para probar el agente final:

   ```python
   # Celda 22: Función de chat para interactuar con TechBot
   def chat_con_techbot(mensaje):
       """
       Envía un mensaje a TechBot y muestra la respuesta formateada.
       
       Args:
           mensaje (str): El mensaje del usuario
           
       Returns:
           str: La respuesta del agente
       """
       print(f"\n{'─' * 60}")
       print(f"👤 Tú: {mensaje}")
       print(f"{'─' * 60}")
       
       respuesta = techbot.predict(input=mensaje)
       
       print(f"🤖 TechBot: {respuesta}")
       
       # Mostrar estadísticas de memoria
       historial = memoria_final.load_memory_variables({})
       turnos = historial["history"].count("Usuario:")
       print(f"\n   [💾 Memoria: {turnos} turno(s) almacenado(s)]")
       
       return respuesta
   
   print("✅ Función chat_con_techbot() lista para usar")
   ```

3. Ejecuta una sesión de demostración completa con TechBot:

   ```python
   # Celda 23: Sesión de demostración con TechBot
   print("🚀 INICIANDO SESIÓN DE DEMOSTRACIÓN CON TECHBOT")
   print("=" * 60)
   
   # Conversación de demostración
   chat_con_techbot("Hola TechBot, ¿puedes explicarme brevemente qué es LangChain?")
   ```

   ```python
   # Celda 24: Segundo turno de la demostración
   chat_con_techbot("¿Cómo se relaciona lo que me explicaste con el concepto de agente de IA?")
   ```

   ```python
   # Celda 25: Tercer turno - referencia a contexto previo
   chat_con_techbot("¿Cuál sería el primer paso práctico para aprender todo esto que hemos hablado?")
   ```

   ```python
   # Celda 26: Ver resumen de la sesión
   print("\n" + "=" * 60)
   print("📊 RESUMEN DE LA SESIÓN DE DEMOSTRACIÓN")
   print("=" * 60)
   
   historial_final = memoria_final.load_memory_variables({})
   turnos_totales = historial_final["history"].count("Usuario:")
   
   print(f"✅ Turnos de conversación completados: {turnos_totales}")
   print(f"✅ Memoria funcionando: El agente recordó el contexto en cada turno")
   print(f"✅ Prompt personalizado: TechBot mantuvo su rol y características")
   print(f"✅ Respuestas en español: Instrucción del prompt respetada")
   print()
   print("Componentes del agente verificados:")
   print("  🧠 LLM → Generó respuestas coherentes y contextuales")
   print("  💾 Memoria → Mantuvo el historial de la conversación")
   print("  📋 Prompt → Definió el comportamiento y personalidad del agente")
   print("  ⚙️  Orquestador → Coordinó el flujo entre todos los componentes")
   ```

**Salida Esperada:**

```
🚀 INICIANDO SESIÓN DE DEMOSTRACIÓN CON TECHBOT
============================================================

────────────────────────────────────────────────────────────
👤 Tú: Hola TechBot, ¿puedes explicarme brevemente qué es LangChain?
────────────────────────────────────────────────────────────
🤖 TechBot: ¡Hola! 👋 Con gusto te explico.

LangChain es un framework de Python diseñado para construir aplicaciones 
con modelos de lenguaje grandes (LLMs)...

   [💾 Memoria: 1 turno(s) almacenado(s)]

────────────────────────────────────────────────────────────
👤 Tú: ¿Cómo se relaciona lo que me explicaste con el concepto de agente de IA?
────────────────────────────────────────────────────────────
🤖 TechBot: Excelente pregunta 🎯 Como te comenté sobre LangChain...
[El agente referencia la explicación anterior]

   [💾 Memoria: 2 turno(s) almacenado(s)]
```

**Verificación:**

- TechBot responde siempre en español
- En el segundo y tercer turno, el agente hace referencia explícita a conversaciones anteriores
- El contador de turnos en memoria aumenta correctamente con cada interacción
- El resumen final muestra los 4 componentes verificados

---

## Validación y Pruebas

### Criterios de Éxito

- [ ] El archivo `.env` existe con `OPENAI_API_KEY` y está en el `.gitignore`
- [ ] LangChain 0.2.x y sus dependencias están instaladas correctamente
- [ ] El chatbot simple sin memoria responde preguntas en español de forma coherente
- [ ] El agente con `ConversationBufferMemory` mantiene el contexto entre al menos 3 turnos
- [ ] Los experimentos de comparación de prompts muestran diferencias observables entre configuraciones
- [ ] El agente final (TechBot) integra los 4 componentes y funciona correctamente
- [ ] No hay API keys expuestas en ninguna celda del notebook

### Procedimiento de Pruebas

1. **Prueba de seguridad de credenciales:** Verifica que la API key no aparece en el notebook:

   ```bash
   # Ejecutar desde la terminal en el directorio del laboratorio
   grep -r "sk-" agente_basico.ipynb && echo "⚠️ ALERTA: API key encontrada en el notebook" || echo "✅ No se encontraron API keys en el notebook"
   ```
   **Resultado Esperado:** `✅ No se encontraron API keys en el notebook`

2. **Prueba de memoria conversacional:** Ejecuta esta secuencia en el notebook para verificar que la memoria funciona:

   ```python
   # Prueba de verificación de memoria
   memoria_prueba = ConversationBufferMemory(memory_key="history", return_messages=False)
   
   agente_prueba = ConversationChain(
       llm=llm,
       memory=memoria_prueba,
       verbose=False
   )
   
   # Turno 1: Introducir información específica
   agente_prueba.predict(input="Mi nombre es Carlos y soy desarrollador Python")
   
   # Turno 2: Verificar que recuerda la información
   respuesta_verificacion = agente_prueba.predict(input="¿Cuál es mi nombre y qué hago?")
   
   # Validar que la respuesta contiene "Carlos"
   assert "Carlos" in respuesta_verificacion, "❌ FALLO: La memoria no está funcionando"
   print("✅ PRUEBA DE MEMORIA: EXITOSA - El agente recordó el nombre del usuario")
   print(f"   Respuesta del agente: {respuesta_verificacion[:100]}...")
   ```
   **Resultado Esperado:** `✅ PRUEBA DE MEMORIA: EXITOSA - El agente recordó el nombre del usuario`

3. **Prueba de configuración del entorno:** Verifica que todas las dependencias están en las versiones correctas:

   ```python
   # Prueba de versiones
   import langchain
   import langchain_openai
   
   version_langchain = tuple(int(x) for x in langchain.__version__.split(".")[:2])
   assert version_langchain >= (0, 2), f"❌ LangChain debe ser 0.2.x o superior, encontrado: {langchain.__version__}"
   
   print(f"✅ LangChain {langchain.__version__} - Versión correcta")
   print(f"✅ LangChain-OpenAI {langchain_openai.__version__} - Instalado")
   print("✅ Todas las pruebas de entorno pasaron correctamente")
   ```
   **Resultado Esperado:** Todas las líneas muestran `✅`

4. **Prueba de comparación de prompts:** Verifica que diferentes prompts producen respuestas diferentes:

   ```python
   # Prueba de efectividad de prompts
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_core.output_parsers import StrOutputParser
   
   pregunta_test = "¿Qué es Python?"
   
   prompt_sin_idioma = ChatPromptTemplate.from_messages([
       ("system", "You are a helpful assistant."),
       ("human", "{pregunta}")
   ])
   
   prompt_con_idioma = ChatPromptTemplate.from_messages([
       ("system", "Eres un asistente. Responde SIEMPRE en español."),
       ("human", "{pregunta}")
   ])
   
   cadena_sin = prompt_sin_idioma | llm | StrOutputParser()
   cadena_con = prompt_con_idioma | llm | StrOutputParser()
   
   resp_sin = cadena_sin.invoke({"pregunta": pregunta_test})
   resp_con = cadena_con.invoke({"pregunta": pregunta_test})
   
   assert resp_sin != resp_con, "❌ Los prompts deberían producir respuestas diferentes"
   print("✅ PRUEBA DE PROMPTS: Los diferentes prompts producen respuestas distintas")
   print(f"   Respuesta sin instrucción de idioma (primeras 50 chars): {resp_sin[:50]}...")
   print(f"   Respuesta con instrucción de idioma (primeras 50 chars): {resp_con[:50]}...")
   ```
   **Resultado Esperado:** `✅ PRUEBA DE PROMPTS: Los diferentes prompts producen respuestas distintas`

## Solución de Problemas

### Problema 1: AuthenticationError - API Key Inválida o No Encontrada

**Síntomas:**
- El código lanza `openai.AuthenticationError: Incorrect API key provided`
- El mensaje de error menciona `invalid_api_key`
- La API key muestra `None` al ejecutar `os.getenv("OPENAI_API_KEY")`

**Causa:**
El archivo `.env` no existe, tiene un error de formato, o `load_dotenv()` no fue ejecutado antes de acceder a la variable de entorno. También puede ocurrir si la API key fue copiada con espacios adicionales.

**Solución:**

```bash
# 1. Verificar que el archivo .env existe en el directorio correcto
ls -la .env

# 2. Verificar el contenido del archivo (sin mostrar la key completa)
head -1 .env | cut -c1-20

# 3. Verificar que no hay espacios en la key
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv('OPENAI_API_KEY', '')
print(f'Longitud de la key: {len(key)} caracteres')
print(f'Tiene espacios al inicio: {key != key.lstrip()}')
print(f'Tiene espacios al final: {key != key.rstrip()}')
"

# 4. Si hay espacios, editar el .env y eliminarlos
# La línea debe ser exactamente: OPENAI_API_KEY=sk-xxxxx (sin espacios)
nano .env
```

---

### Problema 2: ModuleNotFoundError - Paquetes No Instalados

**Síntomas:**
- `ModuleNotFoundError: No module named 'langchain'`
- `ModuleNotFoundError: No module named 'langchain_openai'`
- `ImportError: cannot import name 'ConversationBufferMemory' from 'langchain.memory'`

**Causa:**
El entorno virtual no está activado, los paquetes no fueron instalados en el entorno correcto, o se está usando una versión de LangChain incompatible (0.1.x en lugar de 0.2.x).

**Solución:**

```bash
# 1. Verificar que el entorno virtual está activado
# El prompt de la terminal debe mostrar (venv) al inicio
which python3

# 2. Si no está activado, activarlo
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. Verificar qué paquetes están instalados
pip list | grep -E "langchain|openai"

# 4. Si langchain no aparece o es versión 0.1.x, reinstalar
pip uninstall langchain langchain-openai langchain-core -y
pip install langchain==0.2.16 langchain-openai==0.1.25 langchain-core==0.2.41

# 5. Reiniciar el kernel de Jupyter después de instalar
# En Jupyter: Kernel → Restart Kernel
```

---

### Problema 3: RateLimitError - Límite de Solicitudes de OpenAI

**Síntomas:**
- `openai.RateLimitError: You exceeded your current quota`
- `openai.RateLimitError: Rate limit reached for requests`
- Las respuestas se detienen después de varias llamadas

**Causa:**
Se alcanzó el límite de solicitudes por minuto de la API de OpenAI (por defecto 3 RPM en cuentas nuevas) o se agotaron los créditos disponibles.

**Solución:**

```python
# Agregar manejo de errores y reintentos en el notebook
import time
from openai import RateLimitError

def llamar_agente_con_reintento(agente, mensaje, max_reintentos=3):
    """
    Llama al agente con reintentos automáticos ante RateLimitError.
    """
    for intento in range(max_reintentos):
        try:
            return agente.predict(input=mensaje)
        except RateLimitError as e:
            if intento < max_reintentos - 1:
                tiempo_espera = (intento + 1) * 20  # 20, 40, 60 segundos
                print(f"⏳ Límite de API alcanzado. Esperando {tiempo_espera}s antes de reintentar...")
                time.sleep(tiempo_espera)
            else:
                print("❌ Se agotaron los reintentos. Verifica tus créditos en:")
                print("   https://platform.openai.com/account/usage")
                raise e

# Usar la función en lugar de .predict() directamente
# respuesta = llamar_agente_con_reintento(techbot, "Tu pregunta aquí")
```

```bash
# Para verificar el estado de tu cuenta de OpenAI:
# 1. Ir a https://platform.openai.com/account/usage
# 2. Verificar los créditos disponibles
# 3. Si es cuenta nueva, esperar 60 segundos entre llamadas
```

---

### Problema 4: ConversationBufferMemory - El Agente No Recuerda el Contexto

**Síntomas:**
- El agente responde como si no hubiera conversación previa
- Las referencias a "lo que mencionaste antes" no funcionan
- La memoria muestra 0 turnos después de varias interacciones

**Causa:**
Se está creando una nueva instancia de `ConversationBufferMemory` o `ConversationChain` en cada celda, en lugar de reutilizar la misma instancia. También puede ocurrir si se reinicia el kernel de Jupyter.

**Solución:**

```python
# ❌ INCORRECTO: Crea nueva memoria en cada ejecución
def responder_mal(pregunta):
    memoria_nueva = ConversationBufferMemory()  # ← Nueva instancia cada vez
    agente_nuevo = ConversationChain(llm=llm, memory=memoria_nueva)
    return agente_nuevo.predict(input=pregunta)

# ✅ CORRECTO: Reutiliza la misma instancia de memoria
# Definir UNA SOLA VEZ fuera de la función:
memoria_compartida = ConversationBufferMemory(memory_key="history", return_messages=False)
agente_compartido = ConversationChain(llm=llm, memory=memoria_compartida)

def responder_bien(pregunta):
    return agente_compartido.predict(input=pregunta)  # ← Usa la instancia compartida

# Verificar que la memoria persiste
print(f"Turnos en memoria: {memoria_compartida.load_memory_variables({})['history'].count('Human:')}")
```

---

### Problema 5: Jupyter Notebook No Inicia o Puerto Ocupado

**Síntomas:**
- `OSError: [Errno 98] Address already in use`
- El navegador no abre automáticamente
- `jupyter notebook` cuelga sin mostrar URL

**Causa:**
Hay otra instancia de Jupyter corriendo en el puerto 8888, o el proceso anterior no se cerró correctamente.

**Solución:**

```bash
# 1. Verificar qué proceso usa el puerto 8888
lsof -i :8888  # Linux/macOS
# netstat -ano | findstr :8888  # Windows

# 2. Iniciar Jupyter en un puerto diferente
jupyter notebook --port=8889 agente_basico.ipynb

# 3. O terminar el proceso existente y reiniciar
# En Linux/macOS:
pkill -f jupyter
jupyter notebook agente_basico.ipynb

# 4. Si prefieres JupyterLab como alternativa
jupyter lab agente_basico.ipynb
```

## Limpieza

```bash
# 1. Guardar y cerrar el notebook en Jupyter (File → Save → Close)
# Luego, desde la terminal:

# 2. Desactivar el entorno virtual
deactivate

# 3. (Opcional) Limpiar archivos de caché de Python
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# 4. Verificar que el archivo .env NO está en el área de staging de Git (si usas Git)
# git status  # El archivo .env NO debe aparecer en la lista

# 5. (Opcional) Para liberar espacio, eliminar el entorno virtual
# ADVERTENCIA: Deberás reinstalar las dependencias si quieres continuar
# rm -rf venv/

# 6. Verificar el estado final del directorio
ls -la
```

> ⚠️ **Advertencia de Seguridad:** **NUNCA** elimines el archivo `.env` si planeas continuar con el laboratorio. Sin embargo, **NUNCA** subas el archivo `.env` a un repositorio Git público o privado compartido. El archivo `.gitignore` ya está configurado para ignorarlo, pero verifica siempre con `git status` antes de hacer commit.

> ⚠️ **Advertencia de Costos:** Cada llamada a la API de OpenAI genera un costo. Si no planeas continuar trabajando, cierra el notebook para evitar llamadas accidentales. Puedes revisar tu consumo en: [https://platform.openai.com/account/usage](https://platform.openai.com/account/usage)

## Resumen

### Lo que Lograste

- **Configuraste un entorno de desarrollo seguro** con Python, LangChain y gestión de credenciales mediante archivos `.env`, siguiendo las mejores prácticas de seguridad para no exponer API keys en el código
- **Construiste un chatbot funcional sin memoria** usando `ChatOpenAI`, `ChatPromptTemplate` y la sintaxis LCEL (operador `|`), comprendiendo cómo el prompt template y el LLM trabajan juntos
- **Implementaste memoria conversacional** con `ConversationBufferMemory` y `ConversationChain`, transformando el chatbot estático en un agente que mantiene el contexto entre turnos
- **Experimentaste con diseño de prompts** comparando configuraciones de diferente especificidad y roles, observando cómo cada cambio afecta la calidad, estructura e idioma de las respuestas
- **Construiste TechBot**, un agente conversacional completo que integra los cuatro componentes fundamentales: LLM, memoria, prompt template y orquestador

### Conceptos Clave Aprendidos

- Un **agente de IA** supera al chatbot simple porque mantiene contexto, puede razonar sobre conversaciones previas y adaptar sus respuestas según el historial
- Los **cuatro componentes fundamentales** (LLM, memoria, prompt template, orquestador) trabajan en conjunto siguiendo el ciclo Percibir → Razonar → Actuar
- El **diseño del prompt** (especificidad, rol, formato, audiencia) es tan importante como el modelo: el mismo LLM puede producir respuestas radicalmente diferentes según las instrucciones recibidas
- `ConversationBufferMemory` almacena el historial completo en RAM; es simple y efectiva para conversaciones cortas, pero puede saturar la ventana de contexto en conversaciones largas (limitación a explorar en laboratorios posteriores)
- La **gestión segura de credenciales** mediante `.env` y `.gitignore` es una práctica no negociable en el desarrollo de aplicaciones con IA

### Próximos Pasos

- **Laboratorio 2:** Extenderás este agente básico incorporando herramientas externas (tools) y técnicas de RAG con LlamaIndex para que el agente pueda consultar bases de conocimiento locales
- **Exploración adicional:** Prueba el parámetro `verbose=True` en `ConversationChain` para ver el prompt completo que se envía al LLM en cada turno; esto te ayudará a depurar comportamientos inesperados
- **Optimización de costos:** Experimenta con `temperature=0` para respuestas más deterministas y `max_tokens=200` para reducir el consumo de la API
- **Lectura recomendada:** Revisa la documentación oficial de LangChain sobre LCEL (LangChain Expression Language) para entender el operador `|` en profundidad: [https://python.langchain.com/docs/concepts/lcel/](https://python.langchain.com/docs/concepts/lcel/)

## Recursos Adicionales

- **Documentación oficial de LangChain - Agentes:** Introducción a los agentes con ejemplos de código y arquitecturas recomendadas — [https://python.langchain.com/docs/concepts/agents/](https://python.langchain.com/docs/concepts/agents/)
- **LangChain Expression Language (LCEL):** Guía completa del operador `|` y la composición de cadenas — [https://python.langchain.com/docs/concepts/lcel/](https://python.langchain.com/docs/concepts/lcel/)
- **ChatPromptTemplate - Referencia API:** Documentación detallada de todos los tipos de mensajes y métodos disponibles — [https://python.langchain.com/docs/concepts/prompt_templates/](https://python.langchain.com/docs/concepts/prompt_templates/)
- **ConversationBufferMemory - Tipos de Memoria:** Comparativa de los diferentes tipos de memoria disponibles en LangChain (Buffer, Summary, Window) — [https://python.langchain.com/docs/concepts/memory/](https://python.langchain.com/docs/concepts/memory/)
- **ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022):** El paper académico que formalizó el patrón de razonamiento que usan los agentes modernos — [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- **OpenAI Platform - Usage Dashboard:** Monitorea tu consumo de tokens y costos en tiempo real — [https://platform.openai.com/account/usage](https://platform.openai.com/account/usage)
- **Prompt Engineering Guide:** Guía completa de técnicas de prompting con ejemplos prácticos — [https://www.promptingguide.ai/es](https://www.promptingguide.ai/es)
