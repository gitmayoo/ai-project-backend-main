import os
from flask import Flask, g
from app.routes import api_blueprint
from flask_cors import CORS
from config import Config
from app.services.csv_helper import CSVHelper

def create_app():
    app = Flask(__name__)
    # app.config.from_object('config.Config')
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
    # app.config.from_object('config.Config')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')  # Set upload folder
    app.config['CSV_FILE'] = os.path.join(os.getcwd(), 'data.csv')
    app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
    # app.config['SEGMENTED_IMGS'] = os.path.join(os.getcwd(),'segmented_images',')
    app.config['SEGMENTED_IMGS'] = os.path.join(os.getcwd(),'segmented_images',)

    FACE_FOLDER = os.path.join(app.config['SEGMENTED_IMGS'], 'face')
    SKIN_FOLDER = os.path.join(app.config['SEGMENTED_IMGS'], 'skin')


    # @app.after_request
    # def add_cors_headers(response):
    #     response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins, or specify a specific one like 'http://localhost:3000'
    #     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')  # Allow specific headers
    #     response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')  # Allow specific methods
    #     return response
    
    @app.before_request
    def setup():
        g.csv_helper = CSVHelper(app.config['CSV_FILE'])
        g.csv_helper.initialize_csv()

    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    if not os.path.exists(app.config['SEGMENTED_IMGS']):
        os.makedirs(app.config['SEGMENTED_IMGS'])
        os.makedirs(FACE_FOLDER, exist_ok=True)
        os.makedirs(SKIN_FOLDER, exist_ok=True)

    # Register Blueprints
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)