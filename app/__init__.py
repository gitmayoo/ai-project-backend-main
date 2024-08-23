import os
from flask import Flask, g
from app.routes import api_blueprint
from flask_cors import CORS
from config import Config
from app.services.csv_helper import CSVHelper

def create_app():
    app = Flask(__name__)
    # app.config.from_object('config.Config')
    CORS(app, origins=["http://121.0.0.1:3000/"])
    # app.config.from_object('config.Config')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')  # Set upload folder
    app.config['CSV_FILE'] = os.path.join(os.getcwd(), 'data.csv')



    
    @app.before_request
    def setup():
        g.csv_helper = CSVHelper(app.config['CSV_FILE'])
        g.csv_helper.initialize_csv()

    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Register Blueprints
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app
