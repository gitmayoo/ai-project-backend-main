import datetime
import os
from flask import Blueprint, request, jsonify,g, send_file, send_from_directory
import joblib
from app.services.color_extraction import color_extractor
from app.services.image_segmentation import clothe_segmenter, segmenter
from app.services.upload_image import dress_image_upload, handle_upload
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import pandas as pd

from firebase_admin import firestore


api_blueprint = Blueprint('api', __name__)




@api_blueprint.route('/name_verification', methods=['POST'])
def checkName():
    data = request.get_json()
    user_name = data['name']
    print(user_name)
    response = g.csv_helper.read_row(user_name)
    return response


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
    data = request.get_json()
    filename = data['filename']
    segmenter(filename)
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    # Process the filename as needed
    return jsonify({"message": f"Processing done for {filename}"}), 200



@api_blueprint.route('/image/<name>',methods=['GET'])
def get_image(name):
    from app import create_app
    app = create_app()
    img_response = g.csv_helper.image_url(name=name)
    print(img_response)
    return send_file(img_response[0], mimetype='image/jpeg')




BASE_IMAGE_PATH= os.path.join(os.getcwd(),"app",'Clothes')
@api_blueprint.route('/get-cloth-image', methods=['GET'])
def get_cloth_image():
    try:
        # Parse JSON request data
        # info = request.get_json()
        # gender = info.get('gender')
        # image_name = info.get('image_name')
        # gender = image_gender
        # image_name = image_file
        gender = request.args.get('gender')
        image_name = request.args.get('image_name')
        # Validate input
        if not gender or not image_name:
            return jsonify({'error': 'Gender and image name are required'}), 400

        # Determine the folder based on gender
        gender_folder = gender.lower()  # Assuming gender folder names are 'male' and 'female'

        # Construct the full image path
        image_directory = os.path.join(BASE_IMAGE_PATH, gender_folder)
        print(image_directory)


        # Check if the file exists
        if not os.path.exists(os.path.join(image_directory, image_name)):
            return jsonify({'error': 'Image not found'}), 404

        # Send the image file as a response
        return send_from_directory(image_directory, image_name)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
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

    response = dress_image_upload(image=dress_image,dress_type=dress_type,gender=gender)
    # if data['is_dress']:
    #     path = segmenter(filename=filename)
    #     print(path)
    # return path
    path = clothe_segmenter(filename=dress_image.filename,dress_type=dress_type,dress_gender=gender.upper())
    return path


# @api_blueprint.route("/clothe_segment",methods=["POST"])
# def clothe_segmenter():
#     data = request.get_json()
#     filename = data['filename']
    
#     # dress = data['is_dress']
#     # print(dress)
    
   
#     if data['is_dress']:
#         path = segmenter(filename=filename)
#         print(path)
#         segmenter(filename,is_dress=True,dress_file=path) 
#     return path


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
    


def get_user_by_id(user_id):
    try:
        db = firestore.client()
        # Reference the user's document by ID
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if user_doc.exists:
            user =  user_doc.to_dict()
            user['user_id'] = user_id
            return user
        else:
            return None  # User document does not exist
    except Exception as e:
        print(f"Error retrieving user: {e}")
        return None



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
         # Create a document with an auto-generated ID
        user_ref = db.collection('users').add(user_data)

        # Extract the document ID
        user_id = user_ref[1].id
        user_info = get_user_by_id(user_id)
        # Return the document ID in the response
        return jsonify({'message': 'User created successfully', 'user_info': user_info}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # return jsonify("data")

@api_blueprint.route('/update_user', methods=['POST'])
def updateUser():
    db=firestore.client()
    data = request.get_json()
    user_id = data['user_id']
    user_gender = data['gender']
    user_tone = data['userTone']
    user_img_src = data['filename']
   
    user_ref = db.collection('users').document(user_id)
    user_ref.set({
        "gender":user_gender,
        "userTone":user_tone,
        'imageSrc':user_img_src
    }, merge=True)
    read = user_ref.get()
    print(read.to_dict())

   
    return jsonify(read.to_dict()),200
   




@api_blueprint.route('/login', methods=['POST'])
def login():
    db = firestore.client()
    data = request.get_json()
    name = data.get('name')
    password = data.get('password')

    if not name or not password:
        return jsonify({'error': 'Name and password are required'}), 400

    try:
        # Find the user in Firestore by name
        user_ref = db.collection('users').where('name', "==", name).get()
        
        if len(user_ref) == 0:
            return jsonify({'error': 'User not found'}), 404

        user_data = [doc.to_dict() for doc in user_ref]
        
        # Verify the password
        stored_password_hash = user_data[0]['password']
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            return jsonify({'error': 'Invalid credentials'}), 401

        # Include user_id in the response
        response_data = {
            "user_id": user_ref[0].id,  # Get the document ID (user_id)
            **user_data[0]               # Include the user data
        }
        print("user data is ", response_data)
        return jsonify({'message': 'User Logged in successfully', 'user_info': response_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_blueprint.route('/get-images', methods=['POST'])
def get_images():
    
    # Parse JSON request data

    try:
        info = request.get_json()
        color_category = info.get('color_category')
        gender = info.get('gender')
        dressType = info.get("type")
        if gender.lower() == "male":
            if dressType == "top":
                csv_file_path = os.path.join(os.getcwd(),"app",'dressData_male.csv')
            else:
                csv_file_path = os.path.join(os.getcwd(),"app",'dressDataBottom_male.csv')
        else:
            if dressType == "top":
                csv_file_path = os.path.join(os.getcwd(),"app",'dressData_female.csv')
                print("top female")
            else:
                csv_file_path = os.path.join(os.getcwd(),"app",'dressDataBottom_female.csv')
                print("bottom female")

            
        df = pd.read_csv(csv_file_path)
        if not color_category:
            return jsonify({'error': 'Color category is required'}), 400

        # Filter data based on the color category
        filtered_df = df[df['color_category'].str.lower() == color_category.lower()]

        if filtered_df.empty:
            return jsonify({'error': 'No images found for the specified color category'}), 404

        # Return image names and prominent colors
        result = filtered_df[['id','image_name', 'prominent_color']].to_dict(orient='records')



#  Helper function to convert RGB string to hex
        def rgb_to_hex(rgb_str):
            # Convert "(255, 0, 0)" -> "#FF0000"
            rgb = tuple(map(int, rgb_str.strip("()").split(",")))
            return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2]).upper()

        # Convert RGB values to hex and prepare the result
        result = []
        for _, row in filtered_df.iterrows():
            hex_color = rgb_to_hex(row['prominent_color'])
            result.append({
                'id':row['id'],
                'image_name': row['image_name'],
                'prominent_color': hex_color
            })

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@api_blueprint.route('/get-images-id', methods=['POST'])
def get_images_by_id():
    try:
        info = request.get_json()
        color_category = info.get('color_category')
        gender = info.get('gender')
        dressType = info.get("type")
        id = info.get("id")
        if gender == "male":
            if dressType == "top":
                csv_file_path = os.path.join(os.getcwd(),'dressData_male.csv')
            else:
                csv_file_path = os.path.join(os.getcwd(),'dressDataBottom_male.csv')
        else:
            if dressType == "top":
                csv_file_path = os.path.join(os.getcwd(),'dressData_female.csv')
                print("top female")
            else:
                csv_file_path = os.path.join(os.getcwd(),'dressDataBottom_female.csv')
                print("bottom female")

            
        df = pd.read_csv(csv_file_path)
        if not color_category:
            return jsonify({'error': 'Color category is required'}), 400

        # Filter data based on the color category
        filtered_df = df[df['id'] == id]

        if filtered_df.empty:
            return jsonify({'error': 'No images found for the specified color category'}), 404

        # Return image names and prominent colors
        result = filtered_df[['id','image_name', 'prominent_color']].to_dict(orient='records')



#  Helper function to convert RGB string to hex
        def rgb_to_hex(rgb_str):
            # Convert "(255, 0, 0)" -> "#FF0000"
            rgb = tuple(map(int, rgb_str.strip("()").split(",")))
            return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2]).upper()

        # Convert RGB values to hex and prepare the result
        result = []
        for _, row in filtered_df.iterrows():
            hex_color = rgb_to_hex(row['prominent_color'])
            result.append({
                'id':row['id'],
                'image_name': row['image_name'],
                'prominent_color': hex_color
            })

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


# Load the pre-trained models
modelPath = os.path.join(os.getcwd(),"app",'models')


# Load the pre-trained models and data
kmeans = joblib.load(os.path.join(modelPath, "kmeans_model.pkl"))
scaler = joblib.load(os.path.join(modelPath, "scaler.pkl"))
preprocessor = joblib.load(os.path.join(modelPath, "preprocessor.pkl"))



def hex_to_rgb(hex):
    """Convert hex color to RGB tuple."""
    hex_color = hex.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


@api_blueprint.route("/recommend", methods=["POST"])
def recommendation():
    try:
        # Parse request data
        info = request.get_json()
        gender = info.get('gender')
        colorTone = info.get('colorTone')
        dressType = info.get("dressType")



        if not gender or not colorTone:
            return jsonify({'error': 'Gender and colorTone are required'}), 400

        # Create DataFrame for new user
        user_df = pd.DataFrame([{
            'skin_tone_r': hex_to_rgb(colorTone)[0],
            'skin_tone_g': hex_to_rgb(colorTone)[1],
            'skin_tone_b': hex_to_rgb(colorTone)[2],
            'gender': gender,
            'purchased_top':12,
            "purchased_bottom":12
        }])

        # Apply transformations
        user_features = preprocessor.transform(user_df)
        user_features_scaled = scaler.transform(user_features)
        
        # Assign user to a cluster
        user_cluster = kmeans.predict(user_features_scaled)[0]


        # Load the actual data
        actual_data_path = 'app/datas/user_data_with_rgb.csv'
        df = pd.read_csv(actual_data_path)


        # Ensure the DataFrame contains necessary columns
        required_columns = ['skin_tone_r', 'skin_tone_g', 'skin_tone_b', 'gender', 'purchased_top', 'purchased_bottom', 'cluster']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {missing_columns}'}), 500
        
        # Apply preprocessing to the actual data
        df['gender'] = df['gender'].astype(str)  # Ensure gender is in the same format
        df_features = preprocessor.transform(df[['skin_tone_r', 'skin_tone_g', 'skin_tone_b', 'gender','purchased_top', 'purchased_bottom',]])
        df_scaled = scaler.transform(df_features)
        df['cluster'] = kmeans.predict(df_scaled)

       # Get recommendations based on the cluster
        if dressType == "top":
            recommended_items = df[df['cluster'] == user_cluster]['purchased_top'].unique()
        else:
            recommended_items = df[df['cluster'] == user_cluster]['purchased_bottom'].unique()

        # Limit to the first 6 recommendations
        recommended_items = recommended_items[:6]  # Get the first 6 items

        if gender.lower() == "male":
            if dressType == "top":
                csv_file_path = os.path.join(os.getcwd(),"app",'dressData_male.csv')
                print("top male")

            else:
                csv_file_path = os.path.join(os.getcwd(),"app",'dressDataBottom_male.csv')
                print("bottom female")

        else:
            if dressType == "top":
                csv_file_path = os.path.join(os.getcwd(),"app",'dressData_female.csv')
                print("top female")
            else:
                csv_file_path = os.path.join(os.getcwd(),"app",'dressDataBottom_female.csv')
                print("bottom female")

        df = pd.read_csv(csv_file_path)
        filtered_df = df[df['id'].isin(recommended_items)]

        

        if filtered_df.empty:
            return jsonify({'error': 'No images found for the specified color category'}), 404

                # Return image names and prominent colors
        result = filtered_df[['id','image_name', 'prominent_color']].to_dict(orient='records')



        #  Helper function to convert RGB string to hex
        def rgb_to_hex(rgb_str):
            # Convert "(255, 0, 0)" -> "#FF0000"
            rgb = tuple(map(int, rgb_str.strip("()").split(",")))
            return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2]).upper()

        # Convert RGB values to hex and prepare the result
        result = []
        for _, row in filtered_df.iterrows():
            hex_color = rgb_to_hex(row['prominent_color'])
            result.append({
                'id':row['id'],
                'image_name': row['image_name'],
                'prominent_color': hex_color
            })

        return jsonify(result), 200

    except Exception as e:
        print("error:",e)
        return jsonify({'error': str(e)}), 500



 
@api_blueprint.route("/save_purchase", methods=['POST'])
def purchase():
    try:
        info = request.get_json()
        db = firestore.client()
        purchased_top = info.get('purchasedTop')
        purchased_bottom = info.get('purchasedBottom')
        user_id = info.get('user_id')


        if purchased_top is None or purchased_bottom  is None or user_id is None:
            return jsonify({'error': 'Missing required fields'}), 400
        
        user_ref = db.collection('users').document(user_id)
        # Reference to the purchase sub-collection in the user's document

        #  Add purchase data with current timestamp
        purchase_data = {
            'purchased_top': purchased_top,
            'purchased_bottom': purchased_bottom,
            'user_id':user_id
        }
        purchase_data['date'] = firestore.SERVER_TIMESTAMP

        # Create a unique document ID for each purchase entry
        purchase_ref = user_ref.collection('purchases').add(
            purchase_data
        )

        return jsonify({"messgage":"Purchase is added to the user"}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500




    
