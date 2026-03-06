# Construyendo desde cero un RAG con LangChain, FastAPI y React 🚀

> 📖 **[Ver el tutorial completo en LinkedIn](https://www.linkedin.com/pulse/construyendo-desde-cero-un-rag-con-langchainpython-llm-canales-mora-08a0e/)** por Roberto Canales Mora.

Bienvenido a este proyecto, diseñado paso a paso de forma didáctica para entender y construir una aplicación interactiva **RAG (Retrieval-Augmented Generation)** utilizando componentes modernos y modelos de IA en local.

## 🎯 Objetivo del Proyecto

El objetivo es crear un sistema interactivo completo que te permita **hacer preguntas sobre un documento PDF específico** (`libro.pdf`) a través de un chat en la web. La IA buscará primero respuestas en el contenido de ese documento y luego responderá tu pregunta, evitando así improvisar la respuesta (alucinaciones).

---

## 🏗️ Arquitectura y Componentes Clave

Para lograr este resultado, hemos conectado diferentes tecnologías potentes:

1. **Modelos de Inteligencia Artificial Locales (Ollama)** 🦙
   - No necesitas conectarte a APIs de pago, todo funciona en tu ordenador.
   - `llama3.1`: Modelo principal que genera las respuestas.
   - `mxbai-embed-large`: Modelo especializado en transformar texto en "vectores" (números) para que las búsquedas semánticas sean rápidas y precisas.

2. **Backend: Procesamiento y Búsqueda (Python + LangChain + FAISS)** 🐍
   - **LangChain:** El corazón que orquesta todo el proceso, uniendo el lector de PDFs, el fragmentador de textos y la inteligencia artificial.
   - **FAISS (Facebook AI Similarity Search):** Actúa como nuestra "base de datos vectorial". Permite buscar increíblemente rápido entre los miles de párrafos del libro para encontrar los fragmentos más relevantes a cada pregunta.
   - **FastAPI:** El servidor web en Python responsable de exponer nuestra lógica a través de una API.

3. **Frontend: La Interfaz de Usuario (React + Vite)** ⚛️
   - Aplicación web ligera y moderna.
   - Proporciona una caja de texto para preguntar y muestra no solo la respuesta, sino **las fuentes de donde sacó la información** (número de página y fragmentos de texto).

---

## 🛠️ Requisitos Previos

Antes de empezar, necesitas tener instalados en tu equipo:
- **Python** (Y saber usar un entorno virtual `venv`)
- **Node.js y npm** (Para la web React)
- **Ollama**: Descárgalo desde [ollama.com](https://ollama.com/download)

Una vez tengas Ollama, descarga los modelos abriendo una terminal:
```bash
ollama pull llama3.1
ollama pull mxbai-embed-large
```

---

## 🚀 Guía de Instalación y Ejecución

Sigue estos pasos para arrancar el ecosistema en tu máquina local:

### 1. Preparación del Entorno (Backend)

Desde la raíz del proyecto, crea y activa un entorno virtual de Python, luego instala las dependencias:

```bash
# Crear entorno virtual llamado .venv
python -m venv .venv

# Activar el entorno (Linux/Mac)
source .venv/bin/activate
# En Windows usar: .venv\Scripts\activate

# Instalar los paquetes de Python requeridos
pip install -r backend/requirements.txt
```

### 2. Preparar el Documento (El 'Cerebro' del RAG)

Asegúrate de colocar tu documento con el nombre `libro.pdf` en el directorio principal del proyecto (`raglibro/libro.pdf`). 
> *Puedes usar cualquier PDF de tu elección, pero el proyecto base usa un PDF de conversaciones con CEOs.*

### 3. Indexar el Libro (Fragmentarlo y Crear Vectores)

Ejecuta el script indexador. Este leerá el PDF, lo dividirá en porciones pequeñas con solape, calculará sus vectores usando Ollama y guardará el resultado en FAISS.

```bash
python backend/app/indexer.py
```
*Si todo va bien, verás un mensaje indicando que el Vectorstore FAISS está listo y su tiempo de generación.*

### 4. Arrancar el Servidor Backend (API)

Necesitamos exponer la funcionalidad RAG para que la web pueda consumirlo. *(Asumiendo que el punto de entrada es `main.py` o usas uvicorn directamente)*:

```bash
uvicorn backend.app.main:app --reload
```
*El API ahora estará escuchando en `http://127.0.0.1:8000/`*

### 5. Arrancar la Interfaz Web (Frontend)

Abre otra pestaña en tu terminal y dirígete a la carpeta `frontend/web`:

```bash
cd frontend/web
npm install
npm run dev
```

### 🎉 ¡A Disfrutar!
Abre tu navegador en `http://localhost:5173/` (o el puerto que te indique Vite) y ¡ya puedes conversar con tu libro en privado!

---

*Desarrollado como proyecto de auto-aprendizaje en Inteligencia Artificial y arquitecturas Modernas.*
