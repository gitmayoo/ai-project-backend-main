from flask import Blueprint, request, jsonify,g, send_file, send_from_directory
from app.services.color_extraction import color_extractor
from app.services.image_segmentation import clothe_segmenter, segmenter
from app.services.upload_image import dress_image_upload, handle_upload
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt

from firebase_admin import firestore


api_blueprint = Blueprint('api', __name__)




@api_blueprint.route('/name_verification', methods=['POST'])
def checkName():
    data = request.get_json()
    user_name = data['name']
    print(user_name)
    
    response = g.csv_helper.read_row(user_name)
    return response
    # g.csv_helper()


@api_blueprint.route('/upload', methods=['POST'])
def upload():
    if request.method == 'OPTIONS':
        return '', 204
     
    user_name = request.form.get("name")
    user_image = request.files.get("file")
    rensponse = handle_upload(name=user_name,image=user_image)
    return rensponse



@api_blueprint.route('/segment',methods=['POST'])
def segment():
    # print(request.json)
    data = request.get_json()
    filename = data['filename']
    segmenter(filename)

    
    


    # segmenter()
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    # Process the filename as needed
    return jsonify({"message": f"Processing done for {filename}"}), 200



@api_blueprint.route('/image/<name>',methods=['GET'])
def get_image(name):
    from app import create_app
    app = create_app()
    # Send the file from the IMAGE_FOLDER directory
    # param1 = request.args.get('param1')
    # return send_from_directory( app.config['UPLOAD_FOLDER'], filename)

    img_response = g.csv_helper.image_url(name=name)
    print(img_response)
    return send_file(img_response[0], mimetype='image/jpeg')


@api_blueprint.route('/color',methods=['POST'])
def color_extract():
    data = request.get_json()
    filename = data['filename']
    user_name = data['name']
    print(filename,user_name)
    color = color_extractor(filename=filename)
    print(color)
    response = g.csv_helper.edit_row(user_name,face_tone=color)
    print(response)
    
    read_response = g.csv_helper.read_row(user_name)
    return read_response

    # response = color_extractor()
    # return jsonify(response)


@api_blueprint.route('/clotheUpload',methods=['POST'])
def clotheUpload():
    dress_image = request.files.get("file")
    dress_type = request.form.get("type",None)
    gender = request.form.get("gender")
    clothe_segmenter(filename=dress_image.filename,dress_type=dress_type,dress_gender=gender.upper())

    response = dress_image_upload(image=dress_image,dress_type=dress_type,gender=gender)
    return response


@api_blueprint.route("/db",methods=["POST"])
def dbTest():
    try:
        db = firestore.client()
        data = {
            'name':"karthee",
            "age":17
        }
        
        # Create a new document in the 'users' collection with an auto-generated ID
        doc_ref = db.collection('users').add(data)
        
        return jsonify({"message": "User added successfully", "doc_id": doc_ref[1].id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    




@api_blueprint.route('/signup', methods=['POST'])
def sign_up():
    db=firestore.client()
    data = request.get_json()

    name = data.get('name')
    phone = data.get('phone')
    password = data.get('password')

    if not name or not phone or not password:
        return jsonify({'error': 'Name, phone, and password are required'}), 400

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        # Store user details in Firestore
        user_data = {
            'name': name,
            'phone': phone,
            'password': hashed_password.decode('utf-8')
        }
        db.collection('users').document().set(user_data)

        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # return jsonify("data")

@api_blueprint.route('/login', methods=['POST'])
def login():
    db = firestore.client()
    data = request.get_json()
    phone = data.get('phone')
    password = data.get('password')

    if not phone or not password:
        return jsonify({'error': 'Phone and password are required'}), 400

    try:
        # Find the user in Firestore by phone number
        user_ref = db.collection('users').where('phone' ,"==", phone).get()
        
        if len(user_ref) == 0:
            return jsonify({'error': 'User not found'}), 404

        user_data = [doc.to_dict() for doc in user_ref]
        print(user_data)

        # Verify the password
        stored_password_hash = user_data[0]['password']
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            return jsonify({'error': 'Invalid credentials'}), 401

        return jsonify({'message': 'Login successful'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
