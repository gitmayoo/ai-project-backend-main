from flask import Blueprint, request, jsonify
from app.services.color_extraction import color_extractor
from app.services.image_segmentation import segmenter
from app.services.upload_image import handle_upload


api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/upload', methods=['POST'])
def upload():
    
    user_name = request.form.get("name")
    user_image = request.files.get("file")
    print(user_image)
    handle_upload(name=user_name,image=user_image)
    return jsonify({
        "name":user_name,
        
    })
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image uploaded'}), 400
    
    # image = request.files['image']
    # image_path = f'/path_to_save/{image.filename}'
    # image.save(image_path)
    
    # segmented_image_path = segment_image(image_path)
    # return jsonify({'segmented_image_path': segmented_image_path})

# @api_blueprint.route('/store_user', methods=['POST'])
# def store_user():
#     user_info = request.json
#     store_user_info(user_info)
#     return jsonify({'status': 'User info stored successfully'}), 201
