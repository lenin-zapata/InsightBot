# 🚀 InsightBot: Chatbot Analítico con LLMs Open-Source

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Open_Source-7F52FF?style=for-the-badge)

**InsightBot** es una aplicación interactiva construida con Streamlit que permite a los usuarios consultar y analizar datos empresariales mediante un chatbot potenciado por modelos de lenguaje (LLMs) de código abierto.

## 🌟 Características Principales

- **🛠️ Selector de LLMs**: Elige entre diferentes modelos open-source disponibles localmente.
- **🌎 Interfaz bilingüe**: Soporte para español e inglés.
- **📊 Datos integrados**: Dataset de análisis empresarial listo para explorar.
- **💬 Chat intuitivo**: Interfaz conversacional estilo ChatGPT con Streamlit.

## 🛠️ Instalación y Ejecución

1. **Clona el repositorio**:
   git clone https://github.com/tu_usuario/InsightBot.git
   cd InsightBot


2. Crea un entorno virtual (recomendado):
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows

3. Instala dependencias:
   pip install -r requirements.txt

4. Ejecuta la app:
   streamlit run app.py

📂 Dataset Ficticio Incluido
📈 100 productos en 5 categorías.

📅 24 meses de datos históricos (2022-2023).

Métricas clave:

Ventas proyectadas vs. reales.

Compras reales.

Margen de ganancia por categoría.

Ejemplo de consultas posibles:
"¿Cuál fue el producto más vendido en Q1 2023?"
"Muestra una comparativa de ventas por categoría"

⚙️ Stack Tecnológico
Frontend: Streamlit

Backend: Python 3.9+

LLMs: Modelos open-source (ej. LLaMA, Falcon)

Procesamiento de datos: Pandas, NumPy

Visualización: Matplotlib, Plotly

🌳 Estructura del Proyecto

InsightBot/
├── data/                     # Dataset en CSV
│   └── retail_indicators.csv
├── app/                  
│   ├── main.py               # Aplicación principal
|   ├── vector_store.py
│   └── qa_chain.py
├── app.py                   
├── requirements.txt          # Dependencias
└── README.md                 # Este archivo

💼 Casos de Uso Recomendados
Análisis empresarial rápido: Obtén insights sin escribir código.

Prototipado de LLMs: Prueba modelos open-source en un caso real.

Educación: Enseña análisis de datos con ejemplos interactivos.

Desarrollo local: Ideal para entornos con restricciones de cloud.

✍️ Autor
Lenin Omar Zapata Esparza
📍 Quito, Ecuador
📧 lenin.zapata.1993@gmail.com
🔗 LinkedIn

📜 Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

🇪🇸 Español | 🇺🇸 English
El proyecto soporta ambos idiomas en la interfaz y documentación.
