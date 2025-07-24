# GMD-Copilot

## Overview

GMD-Copilot is a project that aims to provide an AI-powered copilot for various tasks, leveraging large language models (LLMs) and vector databases. It integrates with Flask for the web application and uses ChromaDB for vector storage. The project appears to be designed to assist with documentation, data definition language (DDL), and SQL-related queries.

## Demo

You can view a video demonstration of the application below.

[Watch the Demo](./demo/demo.mp4)

Alternatively, here is an embedded version:
<video src="https://github.com/AymanAitAhmed/GMD-Copilot/raw/main/demo/demo.mp4" controls="controls" style="max-width: 720px;">
</video>

## Features

- **AI-Powered Assistance**: Utilizes OpenAI models (or compatible alternatives like OpenRouter) for natural language understanding and generation.
- **Vector Database Integration**: Employs ChromaDB for efficient storage and retrieval of vectorized data, enabling semantic search and context-aware responses.
- **Flask Web Application**: Provides a web interface for user interaction, built with Flask and supporting real-time streaming of responses.
- **Modular Design**: The project is structured with clear separation of concerns, including modules for LLM integrations, ChromaDB handling, and web application components.
- **Database Interaction**: Capable of interacting with databases, including schema introspection and SQL query generation/execution.
- **Extensible**: Designed to be extensible, allowing for integration of different LLM providers and database types.

## Project Structure

The repository is organized into several key directories:

- `webApp/`: Contains the core web application logic, including:
    - `llm_integrations/`: Modules for integrating with various LLMs (e.g., OpenAI).
    - `my_chromadb/`: Custom ChromaDB integration and vector store management.
    - `base/`: Base classes and utilities for the application.
    - `auth/`: Authentication related modules.
- `FlaskApp.py`: The main Flask application entry point.
- `app.py`: Appears to be a core application file, possibly defining the main application logic and LLM/ChromaDB setup.
- `constants.py`: Defines various constants used throughout the project.
- `requeriments`: Lists project dependencies.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AymanAitAhmed/GMD-Copilot.git
    cd GMD-Copilot
    ```

2.  **Install dependencies:**
    It is recommended to create a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requeriments
    ```

3.  **Configuration:**
    Review `constants.py` and `app.py` for any necessary API keys (e.g., OpenAI) or database connection settings.

4.  **Run the application:**
    ```bash
    python FlaskApp.py
    ```
    The web application should then be accessible in your browser, typically at `http://127.0.0.1:8080`.

## Usage

Once the application is running, you can interact with the GMD-Copilot through its web interface. The specific functionalities will depend on the implemented features in `app.py` and the LLM integrations. It is designed to assist with:

- **Documentation Generation**: Automatically generate documentation based on provided context.
- **DDL Generation**: Create Data Definition Language (DDL) statements for database schemas.
- **SQL Query Assistance**: Help in generating and understanding SQL queries.





