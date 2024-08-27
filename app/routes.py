from flask import Blueprint, request, jsonify,g, send_file, send_from_directory
from app.services.color_extraction import color_extractor
from app.services.image_segmentation import segmenter
from app.services.upload_image import handle_upload


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


