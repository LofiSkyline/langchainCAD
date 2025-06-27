from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    CORS(app)

    from .api import api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app
