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

This is a minimal skeleton to get started. Replace stub functions with real implementations.
