import os
from flask import Flask, g
from app.routes import api_blueprint
from flask_cors import CORS
from config import Config
from app.services.csv_helper import CSVHelper
import firebase_admin
from firebase_admin import credentials

# firebase_credentials = f"C:\Users\hp\OneDrive\Desktop\final project program\ai-project-backend-main\fashion-ai-cc420-firebase-adminsdk-c8f4s-fd131ade1f.json"

# cred = credentials.Certificate(firebase_credentials)
# firebase_admin.initialize_app(credp)

firebase_cred = os.path.join(os.getcwd(),"app",'firebase_auth.json')
cred = credentials.Certificate(firebase_cred)
firebase_admin.initialize_app(cred)


def create_app():
    app = Flask(__name__)
    # app.config.from_object('config.Config')
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})




    # firebase_credentials = os.getenv('$GOOGLE_APPLICATION_CREDENTIALS')
    
    

   





    # app.config.from_object('config.Config')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')  # Set upload folder
    app.config['CSV_FILE'] = os.path.join(os.getcwd(), 'data.csv')
    app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
    # app.config['SEGMENTED_IMGS'] = os.path.join(os.getcwd(),'segmented_images',')
    app.config['SEGMENTED_IMGS'] = os.path.join(os.getcwd(),'segmented_images',)

    app.config['DRESS_IMGS'] = os.path.join(app.config['UPLOAD_FOLDER'],"Dresses")

    TOP_DRESSES_MEN = os.path.join(app.config['DRESS_IMGS'],"MEN",'top')
    TOP_DRESSES_WOMEN = os.path.join(app.config['DRESS_IMGS'],"WOMEN",'top')
    BOTTOM_DRESSES_WOMEN = os.path.join(app.config['DRESS_IMGS'],"WOMEN",'bottom')
    BOTTOM_DRESSES_MEN = os.path.join(app.config['DRESS_IMGS'],"MEN",'bottom')

    FACE_FOLDER = os.path.join(app.config['SEGMENTED_IMGS'], 'face')
    DRESS_PATH = os.path.join(app.config['SEGMENTED_IMGS'],'dress')
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
        os.makedirs(DRESS_PATH, exist_ok=True)


    if not os.path.exists(app.config['DRESS_IMGS']):
        os.makedirs(app.config['DRESS_IMGS'])
        os.makedirs(TOP_DRESSES_MEN, exist_ok=True)
        os.makedirs(BOTTOM_DRESSES_MEN, exist_ok=True)
        os.makedirs(TOP_DRESSES_WOMEN, exist_ok=True)
        os.makedirs(BOTTOM_DRESSES_WOMEN, exist_ok=True)


    # Register Blueprints
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)