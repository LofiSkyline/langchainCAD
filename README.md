# langchainCAD

CAD drawing analysis through VLM.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python run.py
   ```

## Project Structure

- `app/` - Flask application package
  - `api/` - API layer using blueprints
  - `services/` - Business logic and algorithm deployment
 - `relay/` - Interfaces for calling VLM/LLM models
- `run.py` - Entry point for running the Flask server

Set the `OPENAI_API_KEY` environment variable before running to allow LangChain
to access your model provider.

This is a minimal skeleton to get started. Replace stub functions with real
implementations.

### API Usage

`/api/analyze` expects a JSON body or a `multipart/form-data` request. When
using form data, send a `pdf` file along with a `json` field containing the CAD
metadata. Both pieces of information will be forwarded to the LangChain
pipeline for processing.
