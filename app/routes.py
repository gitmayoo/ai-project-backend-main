import os
from flask import Blueprint, request, jsonify,g, send_file, send_from_directory
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
    # dress = data['is_dress']
    # print(dress)
    
   
    segmenter(filename)
    # if data['is_dress']:
    #     path = segmenter(filename=filename)
    #     print(path)
    # return path

    
    


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




BASE_IMAGE_PATH= '/Users/karthi/Development/mayoo-project/ai-project-backend-main/app/Clothes'
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
        user = db.collection('users').document().set(user_data)

        return jsonify({'message': 'User created successfully '}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # return jsonify("data")

@api_blueprint.route('/login', methods=['POST'])
def login():
    db = firestore.client()
    data = request.get_json()
    name = data.get('name')
    password = data.get('password')

    if not name or not password:
        return jsonify({'error': 'name and password are required'}), 400

    try:
        # Find the user in Firestore by phone number
        user_ref = db.collection('users').where('name' ,"==", name).get()
        
        if len(user_ref) == 0:
            return jsonify({'error': 'User not found'}), 404

        user_data = [doc.to_dict() for doc in user_ref]
        # print(user_data)

        # Verify the password
        stored_password_hash = user_data[0]['password']
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            return jsonify({'error': 'Invalid credentials'}), 401

        return jsonify(user_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# Initialize data
female_data = {
    'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'skin_tone_r': [255, 204, 139, 255, 180, 120, 230, 160, 100, 220],
    'skin_tone_g': [224, 153, 69, 185, 150, 90, 200, 130, 70, 190],
    'skin_tone_b': [189, 102, 19, 140, 120, 60, 170, 100, 40, 160],
    'skin_tone_category': ['Light', 'Medium', 'Dark', 'Light', 'Medium', 'Dark', 'Light', 'Medium', 'Dark', 'Light'],
    'purchased_item_id': [3, 2, 1, 4, 5, 5, 7, 8, 6, 10]
}

male_data = {
    'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'skin_tone_r': [255, 204, 139, 255, 180, 120, 230, 160, 100, 220],
    'skin_tone_g': [224, 153, 69, 185, 150, 90, 200, 130, 70, 190],
    'skin_tone_b': [189, 102, 19, 140, 120, 60, 170, 100, 40, 160],
    'skin_tone_category': ['Light', 'Medium', 'Dark', 'Light', 'Medium', 'Dark', 'Light', 'Medium', 'Dark', 'Light'],
    'purchased_item_id': [3, 2, 2, 4, 5, 11, 7, 8, 14, 10]
}



# Flask endpoint to get recommendations for a new user
@api_blueprint.route('/recommend', methods=['POST'])
def recommend():
    info = request.get_json()
    data = info['gender']
    colorTone = info['colorTone']
        # Create DataFrame

    df = pd.DataFrame(male_data)

    # Convert categorical features to numerical
    label_encoder = LabelEncoder()
    df['skin_tone_category'] = label_encoder.fit_transform(df['skin_tone_category'])

    # Extract features for clustering
    features = df[['skin_tone_r', 'skin_tone_g', 'skin_tone_b', 'skin_tone_category', 'purchased_item_id']]

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Apply K-means
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(features_normalized)

    # Function to assign a new user to a cluster
    def assign_user_to_cluster(user_features, kmeans_model, scaler):
        user_features_scaled = scaler.transform([user_features])
        cluster = kmeans_model.predict(user_features_scaled)
        return cluster[0]

    # Function to recommend items based on the cluster
    def recommend_items(user_cluster, data):
        recommended_items = data[data['cluster'] == user_cluster]['purchased_item_id']
        return recommended_items.unique()
    
    def hex_to_rgb(hex):
        hex_color = hex.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
        # Function to calculate luminance
    def calculate_luminance(rgb):
        r, g, b = rgb
        return 0.299*r + 0.587*g + 0.114*b

    # Classify skin tones based on luminance
    def classify_skin_tone(luminance):
        if luminance >= 200:
            return 'Light'
        elif luminance >= 100:
            return 'Medium'
        else:
            return 'Dark'

    color = hex_to_rgb(colorTone)
    print(color[0])
    # Parse JSON request data

    skin_tone_r = color[0]
    skin_tone_g = color[1]
    skin_tone_b = color[2]
    skin_tone_category = classify_skin_tone(calculate_luminance(color))
    purchased_item_id = male_data['purchased_item_id'][0]

    print(
    classify_skin_tone(calculate_luminance(color))

    )
    # # Convert skin tone category to numerical value
    skin_tone_category_num = label_encoder.transform([skin_tone_category])[0]

    # # Create user feature vector
    user_features = [skin_tone_r, skin_tone_g, skin_tone_b, skin_tone_category_num, purchased_item_id]

    # # Assign user to a cluster
    user_cluster = assign_user_to_cluster(user_features, kmeans, scaler)

    # # Get recommendations
    recommended_items = recommend_items(user_cluster, df)
    print(recommended_items)

    # Return recommendations as JSON response
    # return jsonify({
    #     'user_cluster': user_cluster,
    #     'recommended_items': recommended_items.tolist()
    # })
    # print(jsonify({"user_cluster":user_cluster}))

    return "succeed"


@api_blueprint.route('/get-images', methods=['POST'])
def get_images():

    

    
    # Parse JSON request data
    try:
        info = request.get_json()
        color_category = info.get('color_category')
        gender = info.get('gender')

        if gender == "male":
            csv_file_path = 'app/dressData_male.csv'
        else:
            csv_file_path = 'app/dressData_female.csv'
        df = pd.read_csv(csv_file_path)

        if not color_category:
            return jsonify({'error': 'Color category is required'}), 400

        # Filter data based on the color category
        filtered_df = df[df['color_category'].str.lower() == color_category.lower()]

        if filtered_df.empty:
            return jsonify({'error': 'No images found for the specified color category'}), 404

        # Return image names and prominent colors
        result = filtered_df[['image_name', 'prominent_color']].to_dict(orient='records')



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
                'image_name': row['image_name'],
                'prominent_color': hex_color
            })

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500