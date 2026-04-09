# Agente Inteligente de Consulta Meteorológica con LlamaIndex y OpenWeatherMap

## Metadatos

| Propiedad | Valor |
|-----------|-------|
| **Duración** | 95 minutos |
| **Complejidad** | Intermedio |
| **Nivel Bloom** | Crear |
| **Módulo** | 3 — Integración de Servicios Externos y RAG |
| **Tecnologías principales** | LlamaIndex, OpenAI GPT-4o, OpenWeatherMap API, FastAPI |

---

## Descripción General

En este laboratorio construirás desde cero un agente conversacional inteligente capaz de responder preguntas sobre el clima de cualquier ciudad del mundo en lenguaje natural. Utilizando LlamaIndex como framework de orquestación y el patrón ReAct (Reasoning + Acting), el agente razonará sobre qué herramienta invocar, consultará la API real de OpenWeatherMap para obtener datos meteorológicos actualizados y generará respuestas contextualizadas y naturales para el usuario.

El laboratorio tiene relevancia práctica directa: aprenderás a conectar un LLM con servicios externos reales siguiendo el patrón Model Context Protocol (MCP), que es la base de cualquier agente de IA que necesite operar en el mundo real más allá de su conocimiento de entrenamiento. Al finalizar, habrás empaquetado el agente como un microservicio REST con FastAPI, listo para integrarse en aplicaciones de producción.

---

## Objetivos de Aprendizaje

Al completar este laboratorio, serás capaz de:

- [ ] Construir un agente conversacional funcional usando `ReActAgent` de LlamaIndex que integre la API de OpenWeatherMap como herramienta externa
- [ ] Definir y registrar herramientas personalizadas (`FunctionTool`) que el agente pueda invocar dinámicamente según el contexto de la conversación
- [ ] Implementar el patrón Model Context Protocol (MCP) para conectar el agente a servicios externos de forma estructurada y segura
- [ ] Aplicar el ciclo Razonamiento-Acción-Observación (ReAct) para resolver consultas que requieren datos en tiempo real
- [ ] Exponer el agente como un microservicio REST con FastAPI y validar su comportamiento mediante pruebas de integración end-to-end

---

## Prerrequisitos

### Conocimientos Requeridos

- Programación en Python: funciones, clases, manejo de excepciones, f-strings y type hints básicos
- Familiaridad con el formato JSON y cómo parsear respuestas de APIs REST con la librería `requests`
- Comprensión conceptual de LLMs: qué son los modelos de lenguaje, cómo funcionan los tokens y el rol del prompt
- Experiencia básica con `pip` y entornos virtuales de Python (`venv`)
- Haber completado los Laboratorios 1 y 2 del curso (fundamentos de agentes LangChain y cadenas de procesamiento)

### Accesos Requeridos

- **Cuenta OpenAI activa** con crédito disponible y clave API (`OPENAI_API_KEY`). Costo estimado para este laboratorio: $0.10–$0.30 USD usando GPT-4o-mini
- **Cuenta gratuita en OpenWeatherMap** con clave API generada (`OPENWEATHERMAP_API_KEY`). Registro gratuito en [https://openweathermap.org/api](https://openweathermap.org/api)
- Conexión a Internet estable (mínimo 10 Mbps) para consumir ambas APIs

> ⚠️ **Advertencia de costos:** El uso de la API de OpenAI genera costos reales. Configura un límite de gasto mensual en tu cuenta de OpenAI en [https://platform.openai.com/account/billing/limits](https://platform.openai.com/account/billing/limits) antes de comenzar. Se recomienda GPT-4o-mini para minimizar costos durante el desarrollo.

---

## Entorno de Laboratorio

### Requisitos de Hardware

| Componente | Especificación |
|------------|----------------|
| Procesador | CPU 64 bits, mínimo 4 núcleos |
| Memoria RAM | Mínimo 8 GB (recomendado 16 GB) |
| Almacenamiento | Mínimo 2 GB libres para dependencias |
| Conexión a Internet | Mínimo 10 Mbps estable |

### Requisitos de Software

| Software | Versión | Propósito |
|----------|---------|-----------|
| Python | 3.10 o 3.11 | Lenguaje principal del laboratorio |
| pip | 23.x o superior | Gestor de paquetes |
| llama-index-core | 0.10.x o superior | Framework de orquestación del agente |
| llama-index-llms-openai | 0.1.x o superior | Integración LlamaIndex con OpenAI |
| openai | 1.x o superior | Cliente SDK de OpenAI |
| requests | 2.31.x o superior | Cliente HTTP para OpenWeatherMap |
| python-dotenv | 1.0.x | Gestión segura de variables de entorno |
| fastapi | 0.110.x o superior | Framework para microservicio REST |
| uvicorn | 0.29.x o superior | Servidor ASGI para FastAPI |
| pydantic | 2.x | Validación de datos de entrada/salida |

### Configuración Inicial

```bash
# 1. Crear el directorio del laboratorio
mkdir lab03-agente-clima
cd lab03-agente-clima

# 2. Crear y activar el entorno virtual
python3 -m venv venv

# En Linux/macOS:
source venv/bin/activate

# En Windows (PowerShell):
# venv\Scripts\Activate.ps1

# 3. Verificar que el entorno virtual está activo
# Deberías ver (venv) al inicio del prompt

# 4. Actualizar pip
pip install --upgrade pip

# 5. Instalar todas las dependencias
pip install llama-index-core==0.10.68 \
            llama-index-llms-openai==0.1.31 \
            llama-index-embeddings-openai==0.1.11 \
            openai==1.40.0 \
            requests==2.31.0 \
            python-dotenv==1.0.1 \
            fastapi==0.111.0 \
            uvicorn==0.30.1 \
            pydantic==2.7.4

# 6. Verificar instalación
pip list | grep -E "llama|openai|fastapi|requests|dotenv"
```

**Salida esperada de verificación:**

```
llama-index-core          0.10.68
llama-index-embeddings-openai  0.1.11
llama-index-llms-openai   0.1.31
openai                    1.40.0
python-dotenv             1.0.1
fastapi                   0.111.0
requests                  2.31.0
```

---

## Instrucciones Paso a Paso

### Paso 1: Configurar Variables de Entorno de Forma Segura

**Objetivo:** Crear el archivo `.env` con las credenciales necesarias y configurar `.gitignore` para asegurar que nunca se suban a un repositorio Git.

**Instrucciones:**

1. Crear el archivo `.gitignore` para proteger las credenciales:

   ```bash
   cat > .gitignore << 'EOF'
   # Variables de entorno y secretos - NUNCA subir a Git
   .env
   .env.local
   .env.*.local
   
   # Entorno virtual
   venv/
   __pycache__/
   *.pyc
   *.pyo
   
   # Archivos de índice vectorial (pueden ser grandes)
   storage/
   *.faiss
   
   # Jupyter checkpoints
   .ipynb_checkpoints/
   EOF
   ```

2. Crear el archivo `.env` con tus credenciales reales:

   ```bash
   cat > .env << 'EOF'
   # OpenAI - Obtener en https://platform.openai.com/api-keys
   OPENAI_API_KEY=sk-proj-TU_CLAVE_OPENAI_AQUI
   
   # OpenWeatherMap - Obtener en https://home.openweathermap.org/api_keys
   OPENWEATHERMAP_API_KEY=TU_CLAVE_OPENWEATHERMAP_AQUI
   
   # Configuración del modelo
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_TEMPERATURE=0.1
   
   # Configuración de la API de clima
   WEATHER_BASE_URL=https://api.openweathermap.org/data/2.5
   WEATHER_UNITS=metric
   WEATHER_LANG=es
   EOF
   ```

   > ⚠️ **IMPORTANTE:** Reemplaza `sk-proj-TU_CLAVE_OPENAI_AQUI` y `TU_CLAVE_OPENWEATHERMAP_AQUI` con tus claves reales. Nunca compartas este archivo ni lo subas a Git.

3. Verificar que `.gitignore` protege el archivo `.env`:

   ```bash
   git init
   git status
   ```

**Salida esperada:**

```
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .gitignore

nothing added to commit but untracked files present
```

**Verificación:**

- El archivo `.env` NO debe aparecer en `git status` — si aparece, revisa el `.gitignore`
- El archivo `.gitignore` SÍ debe aparecer como archivo a trackear

---

### Paso 2: Explorar la API de OpenWeatherMap Directamente

**Objetivo:** Comprender el formato de respuesta JSON de la API de clima antes de integrarlo en el agente, identificando los campos más relevantes para construir respuestas naturales.

**Instrucciones:**

1. Crear el script de exploración de la API:

   ```bash
   cat > explorar_api_clima.py << 'EOF'
   """
   Paso 2: Exploración directa de la API de OpenWeatherMap
   Objetivo: Entender el formato JSON antes de integrarlo en el agente
   """
   
   import os
   import json
   import requests
   from dotenv import load_dotenv
   
   # Cargar variables de entorno
   load_dotenv()
   
   API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
   BASE_URL = os.getenv("WEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
   UNITS = os.getenv("WEATHER_UNITS", "metric")
   LANG = os.getenv("WEATHER_LANG", "es")
   
   
   def obtener_clima_raw(ciudad: str) -> dict:
       """Consulta directa a la API sin procesamiento."""
       url = f"{BASE_URL}/weather"
       params = {
           "q": ciudad,
           "appid": API_KEY,
           "units": UNITS,
           "lang": LANG
       }
       
       print(f"\n🌐 Consultando: {url}")
       print(f"📋 Parámetros: ciudad={ciudad}, units={UNITS}, lang={LANG}")
       
       response = requests.get(url, params=params, timeout=10)
       response.raise_for_status()
       return response.json()
   
   
   def explorar_estructura_json(ciudad: str = "Madrid"):
       """Explora y muestra la estructura completa de la respuesta."""
       print(f"\n{'='*60}")
       print(f"EXPLORANDO API DE CLIMA PARA: {ciudad}")
       print(f"{'='*60}")
       
       datos = obtener_clima_raw(ciudad)
       
       print("\n📦 RESPUESTA JSON COMPLETA:")
       print(json.dumps(datos, indent=2, ensure_ascii=False))
       
       print("\n🔍 CAMPOS CLAVE IDENTIFICADOS:")
       print(f"  • Temperatura actual: {datos['main']['temp']}°C")
       print(f"  • Sensación térmica: {datos['main']['feels_like']}°C")
       print(f"  • Temperatura mínima: {datos['main']['temp_min']}°C")
       print(f"  • Temperatura máxima: {datos['main']['temp_max']}°C")
       print(f"  • Humedad: {datos['main']['humidity']}%")
       print(f"  • Presión: {datos['main']['pressure']} hPa")
       print(f"  • Descripción: {datos['weather'][0]['description']}")
       print(f"  • Velocidad del viento: {datos['wind']['speed']} m/s")
       print(f"  • Visibilidad: {datos.get('visibility', 'N/A')} metros")
       print(f"  • País: {datos['sys']['country']}")
       print(f"  • Ciudad: {datos['name']}")
       
       return datos
   
   
   if __name__ == "__main__":
       # Probar con varias ciudades para ver variaciones
       ciudades = ["Madrid", "Mexico City", "Buenos Aires"]
       
       for ciudad in ciudades:
           try:
               explorar_estructura_json(ciudad)
           except requests.exceptions.HTTPError as e:
               print(f"❌ Error HTTP para {ciudad}: {e}")
           except requests.exceptions.ConnectionError:
               print(f"❌ Error de conexión. Verifica tu Internet.")
               break
   EOF
   ```

2. Ejecutar el script de exploración:

   ```bash
   python explorar_api_clima.py
   ```

**Salida esperada:**

```
============================================================
EXPLORANDO API DE CLIMA PARA: Madrid
============================================================

🌐 Consultando: https://api.openweathermap.org/data/2.5/weather
📋 Parámetros: ciudad=Madrid, units=metric, lang=es

📦 RESPUESTA JSON COMPLETA:
{
  "coord": {"lon": -3.7026, "lat": 40.4165},
  "weather": [{"id": 800, "main": "Clear", "description": "cielo claro", "icon": "01d"}],
  "main": {
    "temp": 22.5,
    "feels_like": 21.8,
    "temp_min": 19.2,
    "temp_max": 25.1,
    "pressure": 1015,
    "humidity": 45
  },
  "wind": {"speed": 3.6, "deg": 230},
  "visibility": 10000,
  "name": "Madrid",
  "sys": {"country": "ES"}
}

🔍 CAMPOS CLAVE IDENTIFICADOS:
  • Temperatura actual: 22.5°C
  • Sensación térmica: 21.8°C
  • Temperatura mínima: 19.2°C
  • Temperatura máxima: 25.1°C
  • Humedad: 45%
  • Presión: 1015 hPa
  • Descripción: cielo claro
  • Velocidad del viento: 3.6 m/s
  • Visibilidad: 10000 metros
  • País: ES
  • Ciudad: Madrid
```

**Verificación:**

- Debes ver datos reales de temperatura y clima para cada ciudad
- Si ves `401 Unauthorized`, tu clave de OpenWeatherMap es incorrecta o aún no está activada (puede tomar hasta 2 horas tras el registro)
- Si ves `404 Not Found`, el nombre de la ciudad no fue reconocido — intenta con el nombre en inglés

---

### Paso 3: Implementar las Herramientas del Agente (FunctionTools)

**Objetivo:** Crear las funciones Python que actuarán como herramientas del agente, con documentación clara que permita al LLM entender cuándo y cómo usarlas.

**Instrucciones:**

1. Crear el módulo de herramientas de clima:

   ```bash
   cat > herramientas_clima.py << 'EOF'
   """
   Paso 3: Herramientas del agente para consulta meteorológica
   
   Cada función en este módulo se registrará como FunctionTool en LlamaIndex.
   La documentación (docstring) de cada función es CRÍTICA: el LLM la usa
   para decidir cuándo invocar cada herramienta y con qué parámetros.
   
   Patrón MCP (Model Context Protocol):
   - Input: parámetros estructurados y tipados
   - Processing: llamada a API externa con manejo de errores
   - Output: string descriptivo que el LLM puede interpretar
   """
   
   import os
   import requests
   from dotenv import load_dotenv
   
   load_dotenv()
   
   # Configuración global de la API
   _API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
   _BASE_URL = os.getenv("WEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
   _UNITS = os.getenv("WEATHER_UNITS", "metric")
   _LANG = os.getenv("WEATHER_LANG", "es")
   
   
   def obtener_clima_actual(ciudad: str) -> str:
       """
       Obtiene el clima actual de una ciudad específica.
       
       Usa esta herramienta cuando el usuario pregunte sobre el clima actual,
       la temperatura de hoy, las condiciones meteorológicas presentes o
       cualquier información del tiempo en este momento para una ciudad.
       
       Args:
           ciudad: Nombre de la ciudad en español o inglés. Ejemplos válidos:
                   'Madrid', 'Mexico City', 'Buenos Aires', 'New York', 'Tokyo'.
                   Para ciudades con espacios, úsalas tal cual.
       
       Returns:
           String con descripción completa del clima actual incluyendo temperatura,
           humedad, viento y condiciones generales. En caso de error, retorna
           un mensaje descriptivo del problema encontrado.
       """
       if not _API_KEY:
           return "Error: No se encontró la clave de API de OpenWeatherMap. Verifica el archivo .env"
       
       try:
           url = f"{_BASE_URL}/weather"
           params = {
               "q": ciudad,
               "appid": _API_KEY,
               "units": _UNITS,
               "lang": _LANG
           }
           
           response = requests.get(url, params=params, timeout=10)
           
           # Manejo específico de errores HTTP
           if response.status_code == 401:
               return "Error: Clave de API inválida o no activada. Las claves nuevas pueden tardar hasta 2 horas en activarse."
           elif response.status_code == 404:
               return f"Error: No se encontró la ciudad '{ciudad}'. Intenta con el nombre en inglés o verifica la ortografía."
           elif response.status_code == 429:
               return "Error: Límite de solicitudes alcanzado. Espera un momento antes de intentar nuevamente."
           
           response.raise_for_status()
           datos = response.json()
           
           # Extraer y formatear los datos relevantes
           nombre_ciudad = datos.get("name", ciudad)
           pais = datos["sys"]["country"]
           temp = datos["main"]["temp"]
           sensacion = datos["main"]["feels_like"]
           temp_min = datos["main"]["temp_min"]
           temp_max = datos["main"]["temp_max"]
           humedad = datos["main"]["humidity"]
           presion = datos["main"]["pressure"]
           descripcion = datos["weather"][0]["description"]
           viento_velocidad = datos["wind"]["speed"]
           viento_dir = datos["wind"].get("deg", "N/A")
           visibilidad = datos.get("visibility", "N/A")
           
           # Convertir visibilidad a km si está disponible
           if isinstance(visibilidad, (int, float)):
               visibilidad = f"{visibilidad / 1000:.1f} km"
           
           # Construir respuesta estructurada
           resultado = f"""
   🌍 Clima actual en {nombre_ciudad}, {pais}:
   
   🌡️  Temperatura: {temp:.1f}°C (sensación térmica: {sensacion:.1f}°C)
   📊  Rango del día: {temp_min:.1f}°C — {temp_max:.1f}°C
   🌤️  Condiciones: {descripcion.capitalize()}
   💧  Humedad: {humedad}%
   🌬️  Viento: {viento_velocidad} m/s ({viento_velocidad * 3.6:.1f} km/h), dirección: {viento_dir}°
   📍  Presión atmosférica: {presion} hPa
   👁️  Visibilidad: {visibilidad}
   """.strip()
           
           return resultado
           
       except requests.exceptions.ConnectionError:
           return "Error de conexión: No se pudo conectar a la API de OpenWeatherMap. Verifica tu conexión a Internet."
       except requests.exceptions.Timeout:
           return "Error de tiempo de espera: La API tardó demasiado en responder. Intenta nuevamente."
       except requests.exceptions.RequestException as e:
           return f"Error al consultar la API: {str(e)}"
       except KeyError as e:
           return f"Error al procesar la respuesta de la API: campo inesperado {str(e)}"
   
   
   def obtener_pronostico_5dias(ciudad: str) -> str:
       """
       Obtiene el pronóstico del tiempo para los próximos 5 días de una ciudad.
       
       Usa esta herramienta cuando el usuario pregunte sobre el clima futuro,
       el pronóstico de la semana, si va a llover en los próximos días,
       o cualquier consulta sobre condiciones meteorológicas futuras.
       
       Args:
           ciudad: Nombre de la ciudad. Ejemplos: 'Barcelona', 'Bogota', 'Lima'.
       
       Returns:
           String con el pronóstico resumido para los próximos 5 días,
           mostrando temperatura y condiciones por día.
       """
       if not _API_KEY:
           return "Error: No se encontró la clave de API de OpenWeatherMap."
       
       try:
           url = f"{_BASE_URL}/forecast"
           params = {
               "q": ciudad,
               "appid": _API_KEY,
               "units": _UNITS,
               "lang": _LANG,
               "cnt": 40  # 5 días × 8 registros por día (cada 3 horas)
           }
           
           response = requests.get(url, params=params, timeout=10)
           
           if response.status_code == 401:
               return "Error: Clave de API inválida."
           elif response.status_code == 404:
               return f"Error: Ciudad '{ciudad}' no encontrada."
           
           response.raise_for_status()
           datos = response.json()
           
           nombre_ciudad = datos["city"]["name"]
           pais = datos["city"]["country"]
           
           # Agrupar pronósticos por día (tomar el registro del mediodía o el primero disponible)
           pronosticos_por_dia = {}
           for item in datos["list"]:
               fecha = item["dt_txt"].split(" ")[0]
               hora = item["dt_txt"].split(" ")[1]
               
               # Preferir el registro de las 12:00 como representativo del día
               if fecha not in pronosticos_por_dia or hora == "12:00:00":
                   pronosticos_por_dia[fecha] = {
                       "temp_min": item["main"]["temp_min"],
                       "temp_max": item["main"]["temp_max"],
                       "descripcion": item["weather"][0]["description"],
                       "humedad": item["main"]["humidity"],
                       "viento": item["wind"]["speed"]
                   }
           
           # Construir respuesta
           lineas = [f"📅 Pronóstico 5 días para {nombre_ciudad}, {pais}:\n"]
           
           dias_semana = {
               "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
               "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado",
               "Sunday": "Domingo"
           }
           
           from datetime import datetime
           for fecha, info in list(pronosticos_por_dia.items())[:5]:
               dt = datetime.strptime(fecha, "%Y-%m-%d")
               nombre_dia = dias_semana.get(dt.strftime("%A"), dt.strftime("%A"))
               
               lineas.append(
                   f"📆 {nombre_dia} {dt.strftime('%d/%m')}: "
                   f"{info['temp_min']:.0f}°C — {info['temp_max']:.0f}°C | "
                   f"{info['descripcion'].capitalize()} | "
                   f"💧{info['humedad']}% | "
                   f"🌬️{info['viento']:.1f} m/s"
               )
           
           return "\n".join(lineas)
           
       except requests.exceptions.RequestException as e:
           return f"Error al obtener pronóstico: {str(e)}"
       except (KeyError, ValueError) as e:
           return f"Error al procesar datos del pronóstico: {str(e)}"
   
   
   def comparar_clima_ciudades(ciudad1: str, ciudad2: str) -> str:
       """
       Compara el clima actual entre dos ciudades simultáneamente.
       
       Usa esta herramienta cuando el usuario quiera comparar el clima
       entre dos lugares, por ejemplo: '¿Dónde hace más calor, en Madrid
       o en Barcelona?' o '¿Cuál ciudad tiene mejor clima hoy?'
       
       Args:
           ciudad1: Primera ciudad para comparar.
           ciudad2: Segunda ciudad para comparar.
       
       Returns:
           String con comparación detallada del clima entre ambas ciudades,
           indicando cuál tiene mayor temperatura, menor humedad, etc.
       """
       resultado1 = obtener_clima_actual(ciudad1)
       resultado2 = obtener_clima_actual(ciudad2)
       
       if resultado1.startswith("Error") or resultado2.startswith("Error"):
           errores = []
           if resultado1.startswith("Error"):
               errores.append(f"• {ciudad1}: {resultado1}")
           if resultado2.startswith("Error"):
               errores.append(f"• {ciudad2}: {resultado2}")
           return "No se pudo completar la comparación:\n" + "\n".join(errores)
       
       comparacion = f"""
   ⚖️  COMPARACIÓN METEOROLÓGICA
   {'='*45}
   
   {resultado1}
   
   {'─'*45}
   
   {resultado2}
   
   {'='*45}
   📊 Datos obtenidos en tiempo real de OpenWeatherMap
   """.strip()
       
       return comparacion
   
   
   # Prueba rápida del módulo
   if __name__ == "__main__":
       print("🧪 Probando herramientas individualmente...\n")
       
       print("--- HERRAMIENTA 1: Clima actual ---")
       print(obtener_clima_actual("Madrid"))
       
       print("\n--- HERRAMIENTA 2: Pronóstico 5 días ---")
       print(obtener_pronostico_5dias("Barcelona"))
       
       print("\n--- HERRAMIENTA 3: Comparar ciudades ---")
       print(comparar_clima_ciudades("Lima", "Bogota"))
   EOF
   ```

2. Probar las herramientas de forma independiente:

   ```bash
   python herramientas_clima.py
   ```

**Salida esperada:**

```
🧪 Probando herramientas individualmente...

--- HERRAMIENTA 1: Clima actual ---
🌍 Clima actual en Madrid, ES:

🌡️  Temperatura: 22.5°C (sensación térmica: 21.8°C)
📊  Rango del día: 19.2°C — 25.1°C
🌤️  Condiciones: Cielo claro
💧  Humedad: 45%
🌬️  Viento: 3.6 m/s (13.0 km/h), dirección: 230°
📍  Presión atmosférica: 1015 hPa
👁️  Visibilidad: 10.0 km

--- HERRAMIENTA 2: Pronóstico 5 días ---
📅 Pronóstico 5 días para Barcelona, ES:

📆 Martes 15/07: 18°C — 27°C | Nubes dispersas | 💧52% | 🌬️4.2 m/s
...

--- HERRAMIENTA 3: Comparar ciudades ---
⚖️  COMPARACIÓN METEOROLÓGICA
...
```

**Verificación:**

- Las tres herramientas deben retornar datos reales sin errores
- Los datos de temperatura deben ser coherentes con la época del año
- Si alguna herramienta retorna `Error:`, revisa las claves en `.env`

---

### Paso 4: Construir el Agente ReAct con LlamaIndex

**Objetivo:** Integrar las herramientas en un `ReActAgent` de LlamaIndex, configurar el LLM y observar el ciclo Razonamiento-Acción-Observación en acción.

**Instrucciones:**

1. Crear el módulo principal del agente:

   ```bash
   cat > agente_clima.py << 'EOF'
   """
   Paso 4: Agente ReAct de consulta meteorológica con LlamaIndex
   
   Este módulo implementa el patrón ReAct (Reasoning + Acting):
   1. REASONING: El agente analiza la pregunta y decide qué herramienta usar
   2. ACTION: Invoca la herramienta con los parámetros correctos
   3. OBSERVATION: Procesa el resultado de la herramienta
   4. RESPONSE: Genera una respuesta natural al usuario
   
   El patrón MCP (Model Context Protocol) se implementa a través de:
   - FunctionTool: abstracción que encapsula cada función como herramienta
   - Docstrings descriptivos: el LLM los usa para decidir cuándo usar cada tool
   - Tipos de datos: input/output tipados para validación estructurada
   """
   
   import os
   from dotenv import load_dotenv
   
   # LlamaIndex - Framework de orquestación
   from llama_index.core.agent import ReActAgent
   from llama_index.core.tools import FunctionTool
   from llama_index.llms.openai import OpenAI
   
   # Importar nuestras herramientas personalizadas
   from herramientas_clima import (
       obtener_clima_actual,
       obtener_pronostico_5dias,
       comparar_clima_ciudades
   )
   
   # Cargar variables de entorno
   load_dotenv()
   
   
   def crear_agente_clima(verbose: bool = True) -> ReActAgent:
       """
       Crea y configura el agente meteorológico con todas sus herramientas.
       
       Args:
           verbose: Si True, muestra el proceso de razonamiento paso a paso.
                    Útil para debugging y aprendizaje del ciclo ReAct.
       
       Returns:
           ReActAgent configurado y listo para recibir consultas.
       """
       
       # ── 1. CONFIGURAR EL MODELO DE LENGUAJE ──────────────────────────────
       modelo = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
       temperatura = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
       
       llm = OpenAI(
           model=modelo,
           temperature=temperatura,
           # Temperatura baja (0.1) para respuestas más precisas y consistentes
           # con datos meteorológicos (no queremos "creatividad" en datos de clima)
       )
       
       print(f"✅ LLM configurado: {modelo} (temperatura: {temperatura})")
       
       # ── 2. REGISTRAR HERRAMIENTAS COMO FunctionTools ─────────────────────
       # FunctionTool extrae automáticamente el nombre y descripción del docstring
       # El LLM usa esta información para decidir cuándo invocar cada herramienta
       
       herramienta_clima_actual = FunctionTool.from_defaults(
           fn=obtener_clima_actual,
           name="obtener_clima_actual",
           # La descripción adicional refuerza el docstring de la función
           description=(
               "Obtiene el clima actual y condiciones meteorológicas en tiempo real "
               "para cualquier ciudad del mundo. Retorna temperatura, humedad, "
               "viento y descripción del cielo. Úsala para preguntas sobre el "
               "clima de HOY o AHORA MISMO."
           )
       )
       
       herramienta_pronostico = FunctionTool.from_defaults(
           fn=obtener_pronostico_5dias,
           name="obtener_pronostico_5dias",
           description=(
               "Obtiene el pronóstico meteorológico para los próximos 5 días "
               "de una ciudad. Incluye temperaturas mínimas y máximas, "
               "condiciones del cielo y probabilidad de lluvia por día. "
               "Úsala para preguntas sobre el clima FUTURO o de los PRÓXIMOS DÍAS."
           )
       )
       
       herramienta_comparacion = FunctionTool.from_defaults(
           fn=comparar_clima_ciudades,
           name="comparar_clima_ciudades",
           description=(
               "Compara el clima actual entre DOS ciudades simultáneamente. "
               "Úsala cuando el usuario quiera saber cuál ciudad tiene mejor "
               "clima, más calor, más frío o cualquier comparación entre dos lugares."
           )
       )
       
       herramientas = [
           herramienta_clima_actual,
           herramienta_pronostico,
           herramienta_comparacion
       ]
       
       print(f"✅ {len(herramientas)} herramientas registradas:")
       for h in herramientas:
           print(f"   • {h.metadata.name}: {h.metadata.description[:60]}...")
       
       # ── 3. CONFIGURAR EL PROMPT DEL SISTEMA ──────────────────────────────
       system_prompt = """
   Eres MeteoBot, un asistente meteorológico experto y amigable.
   
   Tu función es proporcionar información climática precisa y actualizada
   utilizando tus herramientas de consulta meteorológica.
   
   DIRECTRICES:
   - Siempre usa las herramientas disponibles para obtener datos reales y actualizados
   - Nunca inventes datos meteorológicos — si no puedes consultar la API, dilo claramente
   - Presenta la información de forma clara, amigable y contextualizada
   - Si el usuario menciona actividades (picnic, viaje, deporte), adapta tu respuesta
   - Para ciudades con nombres ambiguos, usa el contexto para determinar el país correcto
   - Responde siempre en el mismo idioma que el usuario (español por defecto)
   
   CONSIDERACIONES ÉTICAS:
   - Los datos meteorológicos tienen incertidumbre — especialmente los pronósticos
   - Para decisiones importantes (viajes, eventos), recomienda verificar fuentes oficiales
   - No hagas predicciones más allá de los 5 días disponibles en la API
   """
       
       # ── 4. CREAR EL AGENTE ReAct ──────────────────────────────────────────
       agente = ReActAgent.from_tools(
           tools=herramientas,
           llm=llm,
           verbose=verbose,  # Muestra el proceso de razonamiento en consola
           system_prompt=system_prompt,
           max_iterations=10  # Límite de iteraciones para evitar bucles infinitos
       )
       
       print("\n✅ Agente ReAct creado exitosamente")
       print("   Patrón: Razonamiento → Acción → Observación → Respuesta")
       
       return agente
   
   
   def ejecutar_consulta_interactiva(agente: ReActAgent, pregunta: str) -> str:
       """
       Ejecuta una consulta y muestra el proceso de razonamiento completo.
       
       Args:
           agente: El ReActAgent configurado
           pregunta: Pregunta del usuario en lenguaje natural
       
       Returns:
           Respuesta final generada por el agente
       """
       print(f"\n{'='*60}")
       print(f"👤 USUARIO: {pregunta}")
       print(f"{'='*60}")
       print("🤖 PROCESO DE RAZONAMIENTO DEL AGENTE:")
       print("─"*60)
       
       respuesta = agente.chat(pregunta)
       
       print("─"*60)
       print(f"🤖 RESPUESTA FINAL:\n{respuesta.response}")
       print("="*60)
       
       return respuesta.response
   
   
   # ── MODO INTERACTIVO ─────────────────────────────────────────────────────
   if __name__ == "__main__":
       print("\n🌤️  METEOBOT - Agente Meteorológico con LlamaIndex ReAct")
       print("="*60)
       
       # Crear el agente con verbose=True para ver el razonamiento
       agente = crear_agente_clima(verbose=True)
       
       # Conjunto de preguntas de prueba que ejercitan diferentes herramientas
       preguntas_prueba = [
           "¿Cómo está el clima en Ciudad de México ahora mismo?",
           "¿Va a llover en Bogotá esta semana?",
           "¿Dónde hace más calor hoy, en Miami o en Cancún?",
       ]
       
       print("\n🧪 EJECUTANDO CONSULTAS DE PRUEBA...")
       
       for pregunta in preguntas_prueba:
           ejecutar_consulta_interactiva(agente, pregunta)
           input("\n⏸️  Presiona ENTER para continuar con la siguiente consulta...")
       
       # Modo conversación libre
       print("\n💬 MODO CONVERSACIÓN LIBRE")
       print("Escribe tus preguntas sobre el clima (escribe 'salir' para terminar)\n")
       
       while True:
           pregunta = input("👤 Tú: ").strip()
           if pregunta.lower() in ["salir", "exit", "quit", ""]:
               print("👋 ¡Hasta luego! MeteoBot se despide.")
               break
           ejecutar_consulta_interactiva(agente, pregunta)
   EOF
   ```

2. Ejecutar el agente y observar el ciclo ReAct:

   ```bash
   python agente_clima.py
   ```

**Salida esperada:**

```
🌤️  METEOBOT - Agente Meteorológico con LlamaIndex ReAct
============================================================
✅ LLM configurado: gpt-4o-mini (temperatura: 0.1)
✅ 3 herramientas registradas:
   • obtener_clima_actual: Obtiene el clima actual y condiciones meteorológ...
   • obtener_pronostico_5dias: Obtiene el pronóstico meteorológico para los...
   • comparar_clima_ciudades: Compara el clima actual entre DOS ciudades si...

✅ Agente ReAct creado exitosamente
   Patrón: Razonamiento → Acción → Observación → Respuesta

🧪 EJECUTANDO CONSULTAS DE PRUEBA...

============================================================
👤 USUARIO: ¿Cómo está el clima en Ciudad de México ahora mismo?
============================================================
🤖 PROCESO DE RAZONAMIENTO DEL AGENTE:
────────────────────────────────────────────────────────────
> Running step ...
Thought: El usuario pregunta por el clima actual en Ciudad de México.
Debo usar la herramienta obtener_clima_actual.
Action: obtener_clima_actual
Action Input: {"ciudad": "Mexico City"}
Observation: 🌍 Clima actual en Mexico City, MX:
🌡️  Temperatura: 18.3°C ...
Thought: Tengo los datos necesarios para responder.
Answer: ...
────────────────────────────────────────────────────────────
🤖 RESPUESTA FINAL:
¡Hola! Aquí tienes el clima actual en Ciudad de México...
```

**Verificación:**

- Debes ver el proceso de razonamiento con `Thought:`, `Action:`, `Action Input:`, `Observation:` y `Answer:`
- Cada pregunta de prueba debe invocar una herramienta diferente
- Las respuestas finales deben ser naturales y contextualizar los datos meteorológicos

---

### Paso 5: Exponer el Agente como Microservicio REST con FastAPI

**Objetivo:** Empaquetar el agente en una API REST profesional usando FastAPI, con validación de entrada mediante Pydantic y manejo robusto de errores, siguiendo patrones de producción.

**Instrucciones:**

1. Crear el microservicio FastAPI:

   ```bash
   cat > servicio_agente.py << 'EOF'
   """
   Paso 5: Microservicio REST del Agente Meteorológico
   
   Expone el agente como una API REST con FastAPI, permitiendo:
   - Consultas individuales via POST /consulta
   - Health check via GET /health
   - Información del agente via GET /info
   - Historial de conversación via GET /historial
   
   Implementa:
   - Validación de entrada con Pydantic v2
   - Manejo de errores HTTP apropiado
   - CORS para integración con frontends
   - Documentación automática en /docs
   """
   
   import os
   import time
   from datetime import datetime
   from typing import Optional
   from contextlib import asynccontextmanager
   
   from fastapi import FastAPI, HTTPException, Request
   from fastapi.middleware.cors import CORSMiddleware
   from fastapi.responses import JSONResponse
   from pydantic import BaseModel, Field
   from dotenv import load_dotenv
   
   # Importar el agente
   from agente_clima import crear_agente_clima
   
   load_dotenv()
   
   # ── MODELOS PYDANTIC ─────────────────────────────────────────────────────
   
   class ConsultaRequest(BaseModel):
       """Modelo de entrada para consultas al agente."""
       pregunta: str = Field(
           ...,
           min_length=3,
           max_length=500,
           description="Pregunta sobre el clima en lenguaje natural",
           examples=["¿Cómo está el clima en Madrid?"]
       )
       session_id: Optional[str] = Field(
           default=None,
           description="ID de sesión para mantener contexto de conversación"
       )
   
   
   class ConsultaResponse(BaseModel):
       """Modelo de respuesta del agente."""
       respuesta: str = Field(description="Respuesta generada por el agente")
       session_id: Optional[str] = Field(description="ID de sesión")
       tiempo_procesamiento_ms: float = Field(description="Tiempo de procesamiento en ms")
       timestamp: str = Field(description="Timestamp de la respuesta")
       modelo_usado: str = Field(description="Modelo LLM utilizado")
   
   
   class HealthResponse(BaseModel):
       """Modelo de respuesta para health check."""
       status: str
       agente_activo: bool
       modelo: str
       herramientas_disponibles: list[str]
       timestamp: str
   
   
   # ── ESTADO GLOBAL DEL SERVICIO ────────────────────────────────────────────
   
   estado_servicio = {
       "agente": None,
       "inicio": None,
       "total_consultas": 0,
       "historial": []
   }
   
   
   # ── CICLO DE VIDA DE LA APLICACIÓN ────────────────────────────────────────
   
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       """Inicializa el agente al arrancar el servicio."""
       print("\n🚀 Iniciando MeteoBot Service...")
       print("─"*50)
       
       try:
           # Crear el agente (verbose=False en producción para no saturar logs)
           estado_servicio["agente"] = crear_agente_clima(verbose=False)
           estado_servicio["inicio"] = datetime.now().isoformat()
           print("✅ Agente inicializado correctamente")
           print("📡 Servicio listo para recibir consultas")
           print("─"*50)
       except Exception as e:
           print(f"❌ Error al inicializar el agente: {e}")
           raise
       
       yield  # El servicio está corriendo
       
       # Cleanup al cerrar
       print("\n🛑 Cerrando MeteoBot Service...")
   
   
   # ── APLICACIÓN FASTAPI ────────────────────────────────────────────────────
   
   app = FastAPI(
       title="MeteoBot API",
       description="""
   ## Agente Meteorológico Inteligente
   
   API REST que expone un agente de IA capaz de responder preguntas sobre
   el clima en lenguaje natural, consultando datos en tiempo real de
   OpenWeatherMap mediante el patrón ReAct (Reasoning + Acting).
   
   ### Herramientas disponibles:
   - **obtener_clima_actual**: Clima en tiempo real de cualquier ciudad
   - **obtener_pronostico_5dias**: Pronóstico para los próximos 5 días  
   - **comparar_clima_ciudades**: Comparación entre dos ciudades
   
   ### Ejemplo de uso:
   ```json
   POST /consulta
   {"pregunta": "¿Va a llover en Barcelona esta semana?"}
   ```
   """,
       version="1.0.0",
       lifespan=lifespan
   )
   
   # Configurar CORS para permitir acceso desde frontends
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # En producción, especificar dominios permitidos
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   
   
   # ── ENDPOINTS ─────────────────────────────────────────────────────────────
   
   @app.get("/health", response_model=HealthResponse, tags=["Sistema"])
   async def health_check():
       """
       Verifica el estado del servicio y el agente.
       Útil para monitoreo y load balancers.
       """
       agente = estado_servicio["agente"]
       agente_activo = agente is not None
       
       herramientas = []
       if agente_activo:
           herramientas = [
               "obtener_clima_actual",
               "obtener_pronostico_5dias",
               "comparar_clima_ciudades"
           ]
       
       return HealthResponse(
           status="healthy" if agente_activo else "degraded",
           agente_activo=agente_activo,
           modelo=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
           herramientas_disponibles=herramientas,
           timestamp=datetime.now().isoformat()
       )
   
   
   @app.get("/info", tags=["Sistema"])
   async def info_servicio():
       """Retorna información general del servicio."""
       return {
           "nombre": "MeteoBot API",
           "version": "1.0.0",
           "descripcion": "Agente meteorológico con LlamaIndex ReAct + OpenWeatherMap",
           "inicio_servicio": estado_servicio["inicio"],
           "total_consultas_procesadas": estado_servicio["total_consultas"],
           "modelo_llm": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
           "patron_agente": "ReAct (Reasoning + Acting)",
           "framework": "LlamaIndex 0.10.x"
       }
   
   
   @app.post("/consulta", response_model=ConsultaResponse, tags=["Agente"])
   async def procesar_consulta(request: ConsultaRequest):
       """
       Procesa una consulta meteorológica en lenguaje natural.
       
       El agente razonará sobre qué herramienta usar, consultará la API
       de OpenWeatherMap y generará una respuesta contextualizada.
       """
       agente = estado_servicio["agente"]
       
       if agente is None:
           raise HTTPException(
               status_code=503,
               detail="El agente no está disponible. Contacta al administrador."
           )
       
       inicio = time.time()
       
       try:
           print(f"\n📨 Nueva consulta: '{request.pregunta[:50]}...'")
           
           # Procesar con el agente
           respuesta_agente = agente.chat(request.pregunta)
           
           tiempo_ms = (time.time() - inicio) * 1000
           timestamp = datetime.now().isoformat()
           
           # Registrar en historial
           estado_servicio["total_consultas"] += 1
           estado_servicio["historial"].append({
               "pregunta": request.pregunta,
               "respuesta_preview": respuesta_agente.response[:100],
               "timestamp": timestamp,
               "tiempo_ms": round(tiempo_ms, 2)
           })
           
           # Mantener solo las últimas 50 consultas en memoria
           if len(estado_servicio["historial"]) > 50:
               estado_servicio["historial"] = estado_servicio["historial"][-50:]
           
           print(f"✅ Consulta procesada en {tiempo_ms:.0f}ms")
           
           return ConsultaResponse(
               respuesta=respuesta_agente.response,
               session_id=request.session_id,
               tiempo_procesamiento_ms=round(tiempo_ms, 2),
               timestamp=timestamp,
               modelo_usado=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
           )
           
       except Exception as e:
           tiempo_ms = (time.time() - inicio) * 1000
           print(f"❌ Error procesando consulta: {str(e)}")
           
           raise HTTPException(
               status_code=500,
               detail=f"Error al procesar la consulta: {str(e)}"
           )
   
   
   @app.get("/historial", tags=["Agente"])
   async def obtener_historial(limite: int = 10):
       """
       Retorna el historial de las últimas consultas procesadas.
       
       Args:
           limite: Número máximo de entradas a retornar (máximo 50)
       """
       limite = min(limite, 50)
       historial = estado_servicio["historial"][-limite:]
       
       return {
           "total_consultas": estado_servicio["total_consultas"],
           "mostrando": len(historial),
           "historial": historial
       }
   
   
   # ── MANEJADOR DE ERRORES GLOBAL ───────────────────────────────────────────
   
   @app.exception_handler(Exception)
   async def manejador_error_global(request: Request, exc: Exception):
       """Captura errores no manejados y retorna respuesta JSON apropiada."""
       return JSONResponse(
           status_code=500,
           content={
               "error": "Error interno del servidor",
               "detalle": str(exc),
               "timestamp": datetime.now().isoformat()
           }
       )
   
   
   # ── PUNTO DE ENTRADA ──────────────────────────────────────────────────────
   
   if __name__ == "__main__":
       import uvicorn
       
       print("\n🌤️  Iniciando MeteoBot Service en http://localhost:8000")
       print("📚 Documentación disponible en http://localhost:8000/docs")
       print("🔍 API alternativa en http://localhost:8000/redoc\n")
       
       uvicorn.run(
           "servicio_agente:app",
           host="0.0.0.0",
           port=8000,
           reload=False,  # False en producción
           log_level="info"
       )
   EOF
   ```

2. Iniciar el microservicio:

   ```bash
   python servicio_agente.py
   ```

**Salida esperada:**

```
🌤️  Iniciando MeteoBot Service en http://localhost:8000
📚 Documentación disponible en http://localhost:8000/docs
🔍 API alternativa en http://localhost:8000/redoc

🚀 Iniciando MeteoBot Service...
──────────────────────────────────────────────────
✅ LLM configurado: gpt-4o-mini (temperatura: 0.1)
✅ 3 herramientas registradas:
   • obtener_clima_actual: ...
   • obtener_pronostico_5dias: ...
   • comparar_clima_ciudades: ...
✅ Agente inicializado correctamente
📡 Servicio listo para recibir consultas
──────────────────────────────────────────────────
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Verificación:**

- El servidor debe arrancar sin errores en el puerto 8000
- Abre un navegador en `http://localhost:8000/docs` para ver la documentación Swagger automática
- El endpoint `/health` debe responder con `"status": "healthy"`

---

### Paso 6: Pruebas de Integración End-to-End

**Objetivo:** Validar el comportamiento completo del agente a través de la API REST mediante pruebas automatizadas que cubran los casos de uso principales y los casos de error.

**Instrucciones:**

1. Abrir una **nueva terminal** (mantener el servidor corriendo en la primera) y activar el entorno virtual:

   ```bash
   # Nueva terminal - navegar al directorio del laboratorio
   cd lab03-agente-clima
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\Activate.ps1  # Windows
   ```

2. Crear el script de pruebas de integración:

   ```bash
   cat > pruebas_integracion.py << 'EOF'
   """
   Paso 6: Pruebas de integración end-to-end del servicio MeteoBot
   
   Este script valida:
   1. Health check del servicio
   2. Consultas que invocan cada herramienta
   3. Manejo de entradas inválidas
   4. Respuestas para ciudades no encontradas
   5. Verificación del historial
   """
   
   import requests
   import json
   import time
   
   BASE_URL = "http://localhost:8000"
   
   
   def imprimir_separador(titulo: str):
       print(f"\n{'='*60}")
       print(f"  {titulo}")
       print(f"{'='*60}")
   
   
   def imprimir_resultado(test_nombre: str, exitoso: bool, detalle: str = ""):
       emoji = "✅" if exitoso else "❌"
       print(f"{emoji} {test_nombre}")
       if detalle:
           print(f"   └─ {detalle}")
   
   
   def test_health_check():
       """Prueba 1: Verificar que el servicio está activo."""
       imprimir_separador("PRUEBA 1: Health Check")
       
       response = requests.get(f"{BASE_URL}/health", timeout=5)
       
       assert response.status_code == 200, f"Status code: {response.status_code}"
       datos = response.json()
       
       assert datos["status"] == "healthy", f"Estado: {datos['status']}"
       assert datos["agente_activo"] == True, "Agente no activo"
       assert len(datos["herramientas_disponibles"]) == 3, "Herramientas faltantes"
       
       imprimir_resultado("Health check retorna 200", True, f"Status: {datos['status']}")
       imprimir_resultado("Agente está activo", True, f"Modelo: {datos['modelo']}")
       imprimir_resultado(
           "3 herramientas disponibles", True,
           f"{', '.join(datos['herramientas_disponibles'])}"
       )
       return True
   
   
   def test_consulta_clima_actual():
       """Prueba 2: Consulta que debe invocar obtener_clima_actual."""
       imprimir_separador("PRUEBA 2: Consulta de Clima Actual")
       
       payload = {"pregunta": "¿Cuál es la temperatura actual en Madrid?"}
       
       inicio = time.time()
       response = requests.post(f"{BASE_URL}/consulta", json=payload, timeout=60)
       tiempo = (time.time() - inicio) * 1000
       
       assert response.status_code == 200, f"Error: {response.status_code} - {response.text}"
       datos = response.json()
       
       assert "respuesta" in datos, "Falta campo 'respuesta'"
       assert len(datos["respuesta"]) > 20, "Respuesta demasiado corta"
       assert datos["tiempo_procesamiento_ms"] > 0, "Tiempo de procesamiento inválido"
       
       imprimir_resultado("Endpoint /consulta responde 200", True)
       imprimir_resultado(
           "Respuesta contiene información meteorológica", True,
           f"Preview: {datos['respuesta'][:80]}..."
       )
       imprimir_resultado(
           "Tiempo de procesamiento registrado", True,
           f"{datos['tiempo_procesamiento_ms']:.0f}ms (total: {tiempo:.0f}ms)"
       )
       
       print(f"\n📝 Respuesta completa del agente:\n{datos['respuesta']}")
       return True
   
   
   def test_consulta_pronostico():
       """Prueba 3: Consulta que debe invocar obtener_pronostico_5dias."""
       imprimir_separador("PRUEBA 3: Consulta de Pronóstico")
       
       payload = {"pregunta": "¿Va a llover en Bogotá esta semana?"}
       
       response = requests.post(f"{BASE_URL}/consulta", json=payload, timeout=60)
       
       assert response.status_code == 200, f"Error: {response.status_code}"
       datos = response.json()
       
       assert len(datos["respuesta"]) > 20, "Respuesta demasiado corta"
       
       imprimir_resultado("Consulta de pronóstico procesada", True)
       imprimir_resultado(
           "Respuesta generada correctamente", True,
           f"Preview: {datos['respuesta'][:80]}..."
       )
       
       print(f"\n📝 Respuesta completa:\n{datos['respuesta']}")
       return True
   
   
   def test_consulta_comparacion():
       """Prueba 4: Consulta que debe invocar comparar_clima_ciudades."""
       imprimir_separador("PRUEBA 4: Comparación de Ciudades")
       
       payload = {
           "pregunta": "¿Dónde hace más calor ahora, en Miami o en Cancún?",
           "session_id": "test-session-001"
       }
       
       response = requests.post(f"{BASE_URL}/consulta", json=payload, timeout=60)
       
       assert response.status_code == 200, f"Error: {response.status_code}"
       datos = response.json()
       
       assert datos["session_id"] == "test-session-001", "Session ID no se preservó"
       
       imprimir_resultado("Consulta de comparación procesada", True)
       imprimir_resultado("Session ID preservado en respuesta", True, datos["session_id"])
       
       print(f"\n📝 Respuesta completa:\n{datos['respuesta']}")
       return True
   
   
   def test_entrada_invalida():
       """Prueba 5: Validación de entrada con Pydantic."""
       imprimir_separador("PRUEBA 5: Validación de Entrada Inválida")
       
       # Prueba con pregunta vacía
       response = requests.post(
           f"{BASE_URL}/consulta",
           json={"pregunta": ""},
           timeout=10
       )
       
       assert response.status_code == 422, f"Debería ser 422, recibió: {response.status_code}"
       imprimir_resultado(
           "Pregunta vacía rechazada con 422", True,
           "Validación Pydantic funcionando"
       )
       
       # Prueba con pregunta demasiado corta
       response = requests.post(
           f"{BASE_URL}/consulta",
           json={"pregunta": "ab"},
           timeout=10
       )
       
       assert response.status_code == 422, f"Debería ser 422, recibió: {response.status_code}"
       imprimir_resultado(
           "Pregunta muy corta rechazada con 422", True,
           "min_length=3 funcionando"
       )
       
       # Prueba sin campo requerido
       response = requests.post(
           f"{BASE_URL}/consulta",
           json={"otro_campo": "valor"},
           timeout=10
       )
       
       assert response.status_code == 422, f"Debería ser 422, recibió: {response.status_code}"
       imprimir_resultado(
           "Payload sin 'pregunta' rechazado con 422", True,
           "Campo requerido validado"
       )
       
       return True
   
   
   def test_historial():
       """Prueba 6: Verificar que el historial se actualiza."""
       imprimir_separador("PRUEBA 6: Historial de Consultas")
       
       response = requests.get(f"{BASE_URL}/historial?limite=5", timeout=10)
       
       assert response.status_code == 200, f"Error: {response.status_code}"
       datos = response.json()
       
       assert "total_consultas" in datos, "Falta campo total_consultas"
       assert datos["total_consultas"] >= 3, f"Deberían haber al menos 3 consultas, hay: {datos['total_consultas']}"
       
       imprimir_resultado(
           "Historial accesible", True,
           f"Total consultas registradas: {datos['total_consultas']}"
       )
       imprimir_resultado(
           "Historial contiene consultas previas", True,
           f"Mostrando {datos['mostrando']} de {datos['total_consultas']}"
       )
       
       print("\n📋 Últimas consultas:")
       for entrada in datos["historial"][-3:]:
           print(f"  • [{entrada['timestamp'][:19]}] {entrada['pregunta'][:50]}...")
       
       return True
   
   
   def ejecutar_todas_las_pruebas():
       """Ejecuta todas las pruebas y genera reporte final."""
       print("\n🧪 INICIANDO SUITE DE PRUEBAS DE INTEGRACIÓN - MeteoBot API")
       print("="*60)
       
       pruebas = [
           ("Health Check", test_health_check),
           ("Clima Actual", test_consulta_clima_actual),
           ("Pronóstico 5 Días", test_consulta_pronostico),
           ("Comparación de Ciudades", test_consulta_comparacion),
           ("Validación de Entrada", test_entrada_invalida),
           ("Historial", test_historial),
       ]
       
       resultados = []
       
       for nombre, prueba in pruebas:
           try:
               prueba()
               resultados.append((nombre, True, None))
           except AssertionError as e:
               print(f"\n❌ FALLO en {nombre}: {str(e)}")
               resultados.append((nombre, False, str(e)))
           except requests.exceptions.ConnectionError:
               print(f"\n❌ ERROR: No se pudo conectar al servidor en {BASE_URL}")
               print("   Asegúrate de que el servidor está corriendo: python servicio_agente.py")
               resultados.append((nombre, False, "Servidor no disponible"))
               break
           except Exception as e:
               print(f"\n❌ ERROR inesperado en {nombre}: {str(e)}")
               resultados.append((nombre, False, str(e)))
       
       # Reporte final
       print(f"\n{'='*60}")
       print("📊 REPORTE FINAL DE PRUEBAS")
       print(f"{'='*60}")
       
       exitosas = sum(1 for _, ok, _ in resultados if ok)
       total = len(resultados)
       
       for nombre, ok, error in resultados:
           emoji = "✅" if ok else "❌"
           estado = "PASÓ" if ok else f"FALLÓ: {error}"
           print(f"{emoji} {nombre}: {estado}")
       
       print(f"\n{'─'*60}")
       print(f"Resultado: {exitosas}/{total} pruebas exitosas")
       
       if exitosas == total:
           print("🎉 ¡Todas las pruebas pasaron! El agente está funcionando correctamente.")
       else:
           print(f"⚠️  {total - exitosas} prueba(s) fallaron. Revisa los errores anteriores.")
       
       return exitosas == total
   
   
   if __name__ == "__main__":
       exito = ejecutar_todas_las_pruebas()
       exit(0 if exito else 1)
   EOF
   ```

3. Ejecutar las pruebas de integración (con el servidor corriendo en la otra terminal):

   ```bash
   python pruebas_integracion.py
   ```

**Salida esperada:**

```
🧪 INICIANDO SUITE DE PRUEBAS DE INTEGRACIÓN - MeteoBot API
============================================================

============================================================
  PRUEBA 1: Health Check
============================================================
✅ Health check retorna 200
   └─ Status: healthy
✅ Agente está activo
   └─ Modelo: gpt-4o-mini
✅ 3 herramientas disponibles
   └─ obtener_clima_actual, obtener_pronostico_5dias, comparar_clima_ciudades

...

============================================================
📊 REPORTE FINAL DE PRUEBAS
============================================================
✅ Health Check: PASÓ
✅ Clima Actual: PASÓ
✅ Pronóstico 5 Días: PASÓ
✅ Comparación de Ciudades: PASÓ
✅ Validación de Entrada: PASÓ
✅ Historial: PASÓ

──────────────────────────────────────────────────────────
Resultado: 6/6 pruebas exitosas
🎉 ¡Todas las pruebas pasaron! El agente está funcionando correctamente.
```

**Verificación:**

- Todas las 6 pruebas deben pasar
- Las consultas al agente deben generar respuestas coherentes con los datos meteorológicos
- La validación Pydantic debe rechazar entradas inválidas con código 422

---

### Paso 7: Documentar la Estructura del Proyecto y Crear requirements.txt

**Objetivo:** Finalizar el laboratorio con buenas prácticas de ingeniería: documentar la arquitectura, crear el `requirements.txt` reproducible y el `README.md` del proyecto.

**Instrucciones:**

1. Generar el `requirements.txt` con versiones exactas:

   ```bash
   pip freeze > requirements.txt
   
   # Verificar que contiene los paquetes principales
   grep -E "llama|openai|fastapi|requests|dotenv|pydantic|uvicorn" requirements.txt
   ```

2. Crear el README del proyecto:

   ```bash
   cat > README.md << 'EOF'
   # MeteoBot — Agente Meteorológico Inteligente
   
   ## Descripción
   
   Agente conversacional que responde preguntas sobre el clima en lenguaje natural,
   integrando la API de OpenWeatherMap mediante el patrón ReAct (LlamaIndex).
   
   ## Arquitectura
   
   ```
   Usuario → FastAPI REST → ReActAgent (LlamaIndex)
                                    ↓
                         FunctionTool × 3
                                    ↓
                         OpenWeatherMap API
                                    ↓
                         OpenAI GPT-4o-mini
                                    ↓
                         Respuesta Natural
   ```
   
   ## Patrón ReAct (Reasoning + Acting)
   
   1. **Thought**: El LLM analiza la pregunta y decide qué herramienta usar
   2. **Action**: Invoca la herramienta con parámetros estructurados
   3. **Observation**: Procesa el resultado de la API
   4. **Answer**: Genera respuesta natural con los datos obtenidos
   
   ## Herramientas del Agente
   
   | Herramienta | Descripción | Cuándo se usa |
   |-------------|-------------|---------------|
   | `obtener_clima_actual` | Clima en tiempo real | Preguntas sobre "ahora" |
   | `obtener_pronostico_5dias` | Pronóstico 5 días | Preguntas sobre el futuro |
   | `comparar_clima_ciudades` | Comparar dos ciudades | Preguntas comparativas |
   
   ## Instalación
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # Editar con tus claves
   ```
   
   ## Configuración
   
   Crear archivo `.env` (ver `.env.example`):
   ```
   OPENAI_API_KEY=sk-proj-...
   OPENWEATHERMAP_API_KEY=...
   OPENAI_MODEL=gpt-4o-mini
   ```
   
   ## Uso
   
   ### Modo agente directo:
   ```bash
   python agente_clima.py
   ```
   
   ### Modo microservicio REST:
   ```bash
   python servicio_agente.py
   # API disponible en http://localhost:8000
   # Docs en http://localhost:8000/docs
   ```
   
   ### Pruebas de integración:
   ```bash
   python pruebas_integracion.py
   ```
   
   ## Estructura del Proyecto
   
   ```
   lab03-agente-clima/
   ├── .env                    # Credenciales (NO subir a Git)
   ├── .env.example            # Plantilla de variables de entorno
   ├── .gitignore              # Excluye .env y venv
   ├── README.md               # Este archivo
   ├── requirements.txt        # Dependencias con versiones exactas
   ├── herramientas_clima.py   # FunctionTools del agente
   ├── agente_clima.py         # ReActAgent con LlamaIndex
   ├── servicio_agente.py      # Microservicio FastAPI
   ├── explorar_api_clima.py   # Script de exploración de la API
   └── pruebas_integracion.py  # Suite de pruebas end-to-end
   ```
   
   ## Consideraciones Éticas
   
   - Los datos meteorológicos tienen incertidumbre inherente
   - Los pronósticos a 5 días tienen menor precisión que los de 24h
   - Para decisiones críticas, siempre verificar fuentes meteorológicas oficiales
   - Las claves de API nunca deben aparecer en el código fuente
   
   ## Laboratorio 3 — Módulo de Agentes de IA
   EOF
   ```

3. Crear el archivo `.env.example` como plantilla segura:

   ```bash
   cat > .env.example << 'EOF'
   # Plantilla de variables de entorno - COPIAR como .env y completar con valores reales
   # NUNCA subir el archivo .env a Git
   
   # OpenAI - https://platform.openai.com/api-keys
   OPENAI_API_KEY=sk-proj-REEMPLAZAR_CON_TU_CLAVE
   
   # OpenWeatherMap - https://home.openweathermap.org/api_keys
   OPENWEATHERMAP_API_KEY=REEMPLAZAR_CON_TU_CLAVE
   
   # Configuración del modelo (opciones: gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_TEMPERATURE=0.1
   
   # Configuración de la API de clima
   WEATHER_BASE_URL=https://api.openweathermap.org/data/2.5
   WEATHER_UNITS=metric
   WEATHER_LANG=es
   EOF
   ```

4. Verificar la estructura final del proyecto:

   ```bash
   ls -la
   ```

**Salida esperada:**

```
total 72
drwxr-xr-x  3 usuario  staff   256 Jul 15 10:30 .
drwxr-xr-x  8 usuario  staff   256 Jul 15 09:00 ..
-rw-r--r--  1 usuario  staff   450 Jul 15 10:30 .env
-rw-r--r--  1 usuario  staff   380 Jul 15 10:30 .env.example
-rw-r--r--  1 usuario  staff   180 Jul 15 10:30 .gitignore
-rw-r--r--  1 usuario  staff  2100 Jul 15 10:30 README.md
-rw-r--r--  1 usuario  staff  1850 Jul 15 10:30 agente_clima.py
-rw-r--r--  1 usuario  staff   890 Jul 15 10:30 explorar_api_clima.py
-rw-r--r--  1 usuario  staff  3200 Jul 15 10:30 herramientas_clima.py
-rw-r--r--  1 usuario  staff  2700 Jul 15 10:30 pruebas_integracion.py
-rw-r--r--  1 usuario  staff  1200 Jul 15 10:30 requirements.txt
-rw-r--r--  1 usuario  staff  3100 Jul 15 10:30 servicio_agente.py
drwxr-xr-x  5 usuario  staff   160 Jul 15 09:15 venv
```

**Verificación:**

- El archivo `.env` NO debe estar trackeado por Git (`git status` no debe mostrarlo)
- El `requirements.txt` debe existir con versiones específicas
- El `README.md` debe describir correctamente la arquitectura del proyecto

---

## Validación y Pruebas

### Criterios de Éxito

- [ ] El archivo `.env` contiene las claves de API y NO aparece en `git status`
- [ ] `python explorar_api_clima.py` retorna datos reales de temperatura para al menos 2 ciudades
- [ ] `python herramientas_clima.py` ejecuta las 3 herramientas sin errores
- [ ] `python agente_clima.py` muestra el ciclo ReAct completo (Thought → Action → Observation → Answer)
- [ ] El servidor FastAPI arranca en el puerto 8000 sin errores
- [ ] `http://localhost:8000/health` retorna `"status": "healthy"`
- [ ] `python pruebas_integracion.py` reporta 6/6 pruebas exitosas
- [ ] El `requirements.txt` existe con versiones exactas de los paquetes

### Procedimiento de Pruebas

1. Verificar protección de credenciales:

   ```bash
   git status
   ```
   **Resultado esperado:** El archivo `.env` NO debe aparecer en la lista de archivos

2. Verificar conectividad con OpenWeatherMap:

   ```bash
   python -c "
   import os, requests
   from dotenv import load_dotenv
   load_dotenv()
   r = requests.get('https://api.openweathermap.org/data/2.5/weather',
       params={'q': 'London', 'appid': os.getenv('OPENWEATHERMAP_API_KEY'), 'units': 'metric'})
   print('OpenWeatherMap:', r.status_code, '- OK' if r.status_code == 200 else r.text[:100])
   "
   ```
   **Resultado esperado:** `OpenWeatherMap: 200 - OK`

3. Verificar conectividad con OpenAI:

   ```bash
   python -c "
   import os
   from openai import OpenAI
   from dotenv import load_dotenv
   load_dotenv()
   client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
   r = client.chat.completions.create(
       model='gpt-4o-mini',
       messages=[{'role': 'user', 'content': 'Di solo: OK'}],
       max_tokens=5
   )
   print('OpenAI:', r.choices[0].message.content)
   "
   ```
   **Resultado esperado:** `OpenAI: OK`

4. Prueba rápida del agente completo:

   ```bash
   python -c "
   from agente_clima import crear_agente_clima
   agente = crear_agente_clima(verbose=False)
   r = agente.chat('¿Cuál es la temperatura en Paris ahora?')
   print('Agente respondió:', len(r.response), 'caracteres')
   print(r.response[:200])
   "
   ```
   **Resultado esperado:** Respuesta con temperatura real de París

5. Prueba del microservicio (con el servidor corriendo):

   ```bash
   curl -s -X POST http://localhost:8000/consulta \
     -H "Content-Type: application/json" \
     -d '{"pregunta": "¿Cómo está el clima en Tokyo?"}' | python -m json.tool
   ```
   **Resultado esperado:** JSON con campo `"respuesta"` conteniendo datos meteorológicos de Tokio

---

## Solución de Problemas

### Problema 1: Error 401 de OpenWeatherMap — Clave API Inválida

**Síntomas:**
- `herramientas_clima.py` retorna `"Error: Clave de API inválida o no activada"`
- `explorar_api_clima.py` muestra `HTTPError: 401 Client Error: Unauthorized`

**Causa:**
Las claves de API de OpenWeatherMap recién creadas pueden tardar entre 10 minutos y 2 horas en activarse en sus servidores. También puede ser que la clave esté mal copiada en el archivo `.env`.

**Solución:**

```bash
# Verificar que la clave está correctamente configurada en .env
grep OPENWEATHERMAP_API_KEY .env

# Probar la clave directamente en el navegador (reemplazar TU_CLAVE):
# https://api.openweathermap.org/data/2.5/weather?q=London&appid=TU_CLAVE&units=metric

# Si la clave es nueva, esperar 2 horas y reintentar
# Si el problema persiste, generar una nueva clave en:
# https://home.openweathermap.org/api_keys

# Verificar que no hay espacios o caracteres extra en la clave
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv('OPENWEATHERMAP_API_KEY', '')
print(f'Longitud de clave: {len(key)} caracteres')
print(f'Primeros 6 chars: {key[:6]}...')
print(f'Hay espacios: {\" \" in key}')
"
```

---

### Problema 2: ImportError al Importar LlamaIndex

**Síntomas:**
- `ModuleNotFoundError: No module named 'llama_index'`
- `ImportError: cannot import name 'ReActAgent' from 'llama_index.core.agent'`

**Causa:**
LlamaIndex 0.10.x cambió su estructura de módulos respecto a versiones anteriores. El paquete monolítico `llama-index` fue reemplazado por paquetes separados como `llama-index-core`, `llama-index-llms-openai`, etc.

**Solución:**

```bash
# Verificar qué versión está instalada
pip show llama-index-core

# Si no está instalado o tiene versión incorrecta:
pip uninstall llama-index llama-index-core -y

# Instalar los paquetes correctos con versiones exactas
pip install llama-index-core==0.10.68 \
            llama-index-llms-openai==0.1.31 \
            llama-index-embeddings-openai==0.1.11

# Verificar la importación
python -c "
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
print('✅ Todas las importaciones de LlamaIndex funcionan correctamente')
"
```

---

### Problema 3: El Agente No Invoca las Herramientas (Responde sin Consultar la API)

**Síntomas:**
- El agente responde con datos genéricos o inventados sin mostrar `Action:` en el log
- No se ven llamadas a la API de OpenWeatherMap en los logs
- El ciclo ReAct muestra solo `Thought:` y `Answer:` sin `Action:`

**Causa:**
El LLM puede decidir responder directamente si la descripción de la herramienta no es suficientemente clara, si la temperatura es demasiado alta (respuestas creativas), o si el prompt del sistema no enfatiza el uso obligatorio de herramientas para datos en tiempo real.

**Solución:**

```bash
# Opción 1: Reducir temperatura del modelo para respuestas más deterministas
# Editar .env:
# OPENAI_TEMPERATURE=0.0

# Opción 2: Hacer la pregunta más explícita
python -c "
from agente_clima import crear_agente_clima
agente = crear_agente_clima(verbose=True)
# Pregunta más directa que fuerza el uso de herramientas
r = agente.chat('Consulta la API y dime la temperatura exacta en Tokio ahora mismo')
print(r.response)
"

# Opción 3: Verificar que las herramientas están registradas correctamente
python -c "
from agente_clima import crear_agente_clima
agente = crear_agente_clima(verbose=True)
print('Herramientas registradas:')
for tool in agente.tools:
    print(f'  - {tool.metadata.name}')
"
```

---

### Problema 4: FastAPI No Arranca — Puerto 8000 en Uso

**Síntomas:**
- `ERROR: [Errno 48] Address already in use` (macOS/Linux)
- `ERROR: [WinError 10048] Only one usage of each socket address` (Windows)

**Causa:**
Otra instancia del servidor ya está corriendo en el puerto 8000, o una sesión previa no se cerró correctamente.

**Solución:**

```bash
# Linux/macOS: Encontrar y terminar el proceso que usa el puerto 8000
lsof -ti:8000 | xargs kill -9

# Windows (PowerShell): Encontrar el proceso
netstat -ano | findstr :8000
# Luego terminar con el PID encontrado:
# taskkill /PID <numero_pid> /F

# Alternativa: Usar un puerto diferente
python -c "
import uvicorn
from servicio_agente import app
uvicorn.run(app, host='0.0.0.0', port=8001)
"

# Y actualizar las pruebas:
# Editar pruebas_integracion.py, cambiar BASE_URL = 'http://localhost:8001'
```

---

### Problema 5: Error de Compatibilidad en Windows con FAISS o ChromaDB

**Síntomas:**
- `ERROR: Could not build wheels for faiss-cpu`
- `error: Microsoft Visual C++ 14.0 or greater is required`

**Causa:**
Algunos paquetes de Python requieren compiladores C++ en Windows que no están instalados por defecto.

**Solución:**

```bash
# Instalar Microsoft C++ Build Tools 2022
# Descargar desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Seleccionar: "Desktop development with C++"

# Después de instalar, reiniciar la terminal y:
pip install faiss-cpu --no-cache-dir

# Alternativa sin compilación: usar versión precompilada
pip install faiss-cpu==1.7.4 --extra-index-url https://download.pytorch.org/whl/cpu

# Si el problema persiste en este laboratorio específico (no usa FAISS directamente):
# Este lab NO requiere FAISS. Si aparece este error, es por una dependencia transitiva.
# Instalar solo los paquetes necesarios:
pip install llama-index-core llama-index-llms-openai openai requests python-dotenv fastapi uvicorn pydantic
```

---

## Limpieza

```bash
# 1. Detener el servidor FastAPI (Ctrl+C en la terminal donde está corriendo)

# 2. Desactivar el entorno virtual
deactivate

# 3. (Opcional) Eliminar el entorno virtual para liberar espacio
# ADVERTENCIA: Esto elimina todas las dependencias instaladas
# Deberás reinstalarlas si quieres continuar el laboratorio
rm -rf venv/

# 4. (Opcional) Eliminar archivos de caché de Python
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# 5. Verificar que .env NO fue commiteado accidentalmente
git log --oneline 2>/dev/null | head -5
git show --stat HEAD 2>/dev/null | grep ".env" && echo "⚠️ ADVERTENCIA: .env en commits!" || echo "✅ .env no está en Git"
```

> ⚠️ **Advertencia de Seguridad:** Nunca elimines el archivo `.env` sin antes asegurarte de tener las claves de API guardadas en un gestor de contraseñas seguro. Las claves de OpenAI y OpenWeatherMap son difíciles de recuperar una vez perdidas. Si sospechas que tus claves fueron expuestas en un repositorio Git, revócalas inmediatamente en los portales respectivos y genera nuevas.

> ⚠️ **Advertencia de Costos:** Recuerda que cada llamada al agente consume tokens de OpenAI. Asegúrate de haber configurado un límite de gasto mensual en [https://platform.openai.com/account/billing/limits](https://platform.openai.com/account/billing/limits). El costo total de este laboratorio debería ser menor a $0.50 USD con GPT-4o-mini.

---

## Resumen

### Lo que Lograste

- **Configuraste un entorno seguro** con gestión de credenciales mediante `.env` y `.gitignore`, siguiendo las mejores prácticas de seguridad para proyectos con APIs externas
- **Exploraste la API de OpenWeatherMap** directamente, comprendiendo la estructura JSON de respuesta y los campos meteorológicos relevantes antes de integrarlos en el agente
- **Implementaste tres FunctionTools** con documentación (docstrings) que el LLM usa para decidir cuándo invocar cada herramienta, aplicando el patrón Model Context Protocol (MCP)
- **Construiste un ReActAgent con LlamaIndex** que ejecuta el ciclo completo Razonamiento → Acción → Observación → Respuesta para resolver consultas meteorológicas en lenguaje natural
- **Empaquetaste el agente como microservicio REST** con FastAPI, incluyendo validación Pydantic, manejo de errores HTTP apropiado y documentación automática Swagger
- **Validaste el sistema end-to-end** con una suite de 6 pruebas de integración que cubren casos de uso normales y casos de error

### Conceptos Clave Aprendidos

- **Patrón ReAct**: El agente no responde directamente sino que razona sobre qué herramienta usar, la invoca, observa el resultado y luego genera la respuesta — esto es fundamentalmente diferente a un LLM sin herramientas
- **FunctionTool y MCP**: Los docstrings de las funciones actúan como el "contrato" entre el agente y las herramientas; una descripción clara y precisa es crítica para que el LLM tome decisiones correctas
- **Temperatura baja para datos factuales**: Cuando el agente trabaja con datos de APIs (no creatividad), una temperatura de 0.0–0.1 produce respuestas más consistentes y precisas
- **Separación de responsabilidades**: Las herramientas (`herramientas_clima.py`) son independientes del agente (`agente_clima.py`), que es independiente del servicio (`servicio_agente.py`) — esto facilita el testing y mantenimiento
- **Conexión con RAG**: El mismo patrón de "recuperar información externa → inyectar en contexto → generar respuesta" que usamos con APIs también se aplica con documentos indexados (LlamaIndex VectorStoreIndex del módulo 3.1)

### Próximos Pasos

- **Laboratorio 3 — Extensión opcional**: Agrega una cuarta herramienta `obtener_alertas_meteorologicas(ciudad)` que consulte el endpoint `/alerts` de OpenWeatherMap para ciudades con alertas activas, reforzando el concepto de agentes multi-herramienta
- **Lección 3.2 — Model Context Protocol (MCP)**: Profundizarás en el estándar formal MCP para conectar agentes a servicios externos de forma estructurada, comprendiendo cómo el patrón que implementaste en este laboratorio se formaliza en un protocolo de comunicación
- **Laboratorio 4 — LangSmith**: Aprenderás a observar, depurar y auditar el comportamiento del agente usando trazas completas de LangSmith, identificando ineficiencias en el ciclo ReAct y sesgos en las respuestas generadas

---

## Recursos Adicionales

- **Documentación oficial de LlamaIndex — ReActAgent**: Guía completa sobre configuración del agente, herramientas y parámetros avanzados. Disponible en [https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)
- **OpenWeatherMap API Documentation**: Referencia completa de endpoints, parámetros y formatos de respuesta. Disponible en [https://openweathermap.org/api/one-call-3](https://openweathermap.org/api/one-call-3)
- **"ReAct: Synergizing Reasoning and Acting in Language Models"** (Yao et al., 2023): Paper original que introduce el patrón ReAct. Disponible en [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- **FastAPI Documentation**: Guía oficial para construir APIs REST con validación Pydantic automática. Disponible en [https://fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **LlamaIndex FunctionTool Reference**: Documentación sobre cómo definir herramientas personalizadas para agentes. Disponible en [https://docs.llamaindex.ai/en/stable/api_reference/tools/](https://docs.llamaindex.ai/en/stable/api_reference/tools/)
- **Pydantic v2 Migration Guide**: Si encuentras errores de compatibilidad entre Pydantic v1 y v2. Disponible en [https://docs.pydantic.dev/latest/migration/](https://docs.pydantic.dev/latest/migration/)
