from flask import Flask, jsonify, request, redirect, url_for, render_template ,Response
import os

from flask_cors import CORS


import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


import colorgram
from PIL import Image as im
import torch
# import Response

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit the upload size to 16MB
CORS(app, origins=["http://121.0.0.1:3000/"])



torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
camModel = torch.hub.load('ultralytics/yolov5', 'custom', path='bestModel.pt', force_reload=True)

# faceLimit = 2
# faceCount = 0
# #video stream frames
# def gen_frames():
#     global faceLimit
#     global faceCount  
#     while True:
        
#         success, frame = camera.read()  
#         if not success:
#             break
#         else:
#             faceCount = 0
#             frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = camModel(frameRGB)
#             for box in results.xyxy[0]:
#                 if faceCount < faceLimit:
#                     faceCount += 1 
#                     if box[5] == 1:
#                         className = "Male:"
#                         bgr =(230, 216, 173)
#                     elif box[5] == 0:
#                         className = "Female:"
#                         bgr =(203, 192, 255)
                    
#                     conf = math.floor(box[4] * 100)
#                     xB = int(box[2])
#                     xA = int(box[0])
#                     yB = int(box[3])
#                     yA = int(box[1])
                        
#                     cv2.rectangle(frame, (xA, yA), (xB, yB), (bgr), 4)
#                     cv2.rectangle(frame, (xA, yA-50), (xA+180, yA), (bgr), -1)
#                     cv2.putText(frame, str(conf), (xA + 130, yA-13), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))
#                     cv2.putText(frame, className, (xA, yA-15), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))
#                 else:
#                     break

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  




# @app.route('/')
# def index():
#     return render_template('index.html')

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return "Hello world";

@app.route("/gender")
def gender_detection():
    image = cv2.imread("uploads\captured-image.jpg")
    # image= cv2.imread("WhatsApp Image 2024-08-18 at 00.35.23_352dc498.jpg")
    frameRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = camModel(frameRGB)
    labels = results.names
    detected_classes = [labels[int(box[5])] for box in results.xyxy[0]]
    
    # Assume the first detected class is the relevant one
    if detected_classes:
        detected_class = detected_classes[0]
        return jsonify({'gender': detected_class})
    else:
        return jsonify({'error': 'No face detected'})

   


@app.route('/upload', methods=['POST']) # type: ignore
def handle_upload():
    # Image's desired dimension
    DESIRED_HEIGHT = 256
    DESIRED_WIDTH = 256

    
    def resize_and_preview(image):
    # To get the width and size of the image
        h, w = image.shape[:2]
        # Checking the image's orientation (Landscape or Portrait)
        if h < w:
            new_img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
        else:
            new_img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

        return new_img
   

    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # type: ignore
        file.save(file_path)
        img_file = cv2.imread(file_path)
        resized_img = resize_and_preview(img_file)
        os.remove(file_path)
        cv2.imwrite(f"uploads/{file.filename}",resized_img,)
        return f"File successfully resized and uploaded: {file_path}"

@app.route("/color",methods=["GET"])
def color_extractor():
    # Load the extracted image
    image = im.open("face_extracted.jpg")
    colors = colorgram.extract(image, 10)


    sorted_colors = sorted(colors, key=lambda color: color.proportion, reverse=False)

    # The first color in the sorted list is the most prominent by proportion
    most_prominent_color = sorted_colors[0]
    print("Prominent")
    print(most_prominent_color.rgb)
    print("Prominent")

    extracted_colors = []
    for color in colors:
        if color.rgb.r >= 10 :  # Check if all color components are non-zero
            print(color.rgb)
            extracted_colors.append(color.rgb)


    mean_color = np.mean([np.array(color) for color in extracted_colors], axis=0)
    mean_color = mean_color.astype(int)  # Convert to integers for display

   

    # Add mean color patch
    normalized_mean_color = (mean_color[0] / 255, mean_color[1] / 255, mean_color[2] / 255)




    mean_color = most_prominent_color.rgb


    # Normalize color values to be between 0 and 1
    normalized_mean_color = [c / 255 for c in mean_color]

   
    return str(mean_color)




@app.route("/segment",methods=["POST"])
def segmenter():

    # Specify the path of segmenter model
    model_path = 'selfie_segmenter.tflite'
    base_options = python.BaseOptions(model_asset_path=model_path)


    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode


    options = ImageSegmenterOptions(
        base_options,
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True
    )

    image = mp.Image.create_from_file('WhatsApp Image 2024-08-18 at 00.35.23_352dc498.jpg')

    # with ImageSegmenter.create_from_options(options) as segmenter:
    segmenter = ImageSegmenter.create_from_options(options)
    #   #Image segmenting function
    def img_segmenter(image,to_be_segmented_part):
    # Ensuring no values except string is given
        if not isinstance(to_be_segmented_part, str):
            raise ValueError("Wrong input: Expected an string.")
    
        
    # Segmenting the image
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.confidence_masks
        confidence_mask = segmentation_result.confidence_masks
        segmented_part = None

        #Handling the segmented part
        if(to_be_segmented_part.lower() == "skin"):
            segmented_part = confidence_mask[2]
        elif(to_be_segmented_part.lower() == "face"):
            segmented_part = confidence_mask[3]
        else:
            return "to_be_segmented_part is not specified"
        return segmented_part

        
    confidence_mask = img_segmenter(image,"skin")
    data = im.fromarray((confidence_mask.numpy_view() * 255).astype(np.uint8))
    data.save("segmented.jpg")

    # Choosing between the value of 0.5 and 0.7
    # .7 for more accuracy
    threshold = 0.5

    binary_mask = (confidence_mask.numpy_view() > threshold).astype(np.uint8) * 255


    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    mask = np.zeros_like(image.numpy_view().copy())
    image_with_contours = image.numpy_view().copy()

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour on the image
    # cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    # resized_image = confidence_mask.numpy_view().copy().astype(np.uint8) * 255
    new_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGBA2RGB)

    image_size = image.numpy_view().shape
    image_type = image.numpy_view().dtype

    mask_size = binary_mask.shape
    mask_type = binary_mask.dtype

    face_extracted = cv2.bitwise_and(new_image, binary_mask)
    # plt.imshow(face_extracted)
    # plt.show()


    data = im.fromarray(face_extracted)
    data.save("face_extracted.jpg")


    # return str(confidence_mask.numpy_view())
    return str((image_size,image_type,mask_size,mask_type))

        #     #Ensuring no values except string is given
        #     if not isinstance(to_be_segmented_part, str):
        #         raise ValueError("Wrong input: Expected an string.")

        #     #Segmenting the image
        #     segmentation_result = segmenter.segment(image)
        #     category_mask = segmentation_result.confidence_masks
        #     confidence_mask = segmentation_result.confidence_masks
        #     segmented_part = None

        #     #Handling the segmented part
        #     if(to_be_segmented_part.lower() == "skin"):
        #         segmented_part = confidence_mask[2]
        #     elif(to_be_segmented_part.lower() == "face"):
        #         segmented_part = confidence_mask[4]
        #     else:
        #         return "to_be_segmented_part is not specified"

        #     # Previewing the segmented part
        #     if(segmentation_result != None):
        #         plt.imshow(segmented_part.numpy_view())
        #         plt.title(f"Segmentated {to_be_segmented_part.capitalize()} Region")
        #         plt.show()


        #     threshold = 0.5
        #     binary_mask = (category_mask[2].numpy_view() > threshold).astype(np.uint8) * 255


        # # Find contours in the binary mask
        #     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Determine circle parameters
        # # largest_contour = max(contours, key=cv2.contourArea)
        # # (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        # # center = (int(x), int(y))
        # # radius = int(radius)

        # # Draw the circle on the original image
        # # image_with_circle = image.numpy_view().copy()
        # # cv2.circle(image_with_circle, center, radius, (0, 255, 0), 2)

        #     mask = np.zeros_like(image.numpy_view().copy())
        #     image_with_contours = image.numpy_view().copy()

        #     cv2.drawContours(image_with_contours, contours, -1, (255, 255, 255), 2)
        #     plt.imshow(image_with_contours)
        #     plt.title(f"Segmentated {to_be_segmented_part.capitalize()} Region")
        #     plt.show()




        #     return segmented_part



        #Calling the segmenter function and create a confidence mask
        # Essential for defining the contours / outline
        

        # data = im.fromarray(image.numpy_view())
        # data.save("image.jpg")


    

# @app.route('/file',methods=["GET"])
# def handleFileUpload():
    
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], "WhatsApp Image 2024-05-22 at 19.29.23.jpeg")
#     file_exists = os.path.isfile(file_path)

#     confidence_mask = None
#      # Specify the path of segmenter model
#     image = mp.Image.create_from_file('uploads/WhatsApp Image 2024-05-22 at 19.29.23.jpeg')
#     with ImageSegmenter.create_from_options(options) as segmenter:

# #   #Image segmenting function
#         def img_segmenter(image,to_be_segmented_part):

#             #Ensuring no values except string is given
#             if not isinstance(to_be_segmented_part, str):
#                 raise ValueError("Wrong input: Expected an string.")

#             #Segmenting the image
#             segmentation_result = segmenter.segment(image)
#             category_mask = segmentation_result.confidence_masks
#             confidence_mask = segmentation_result.confidence_masks
#             segmented_part = None

#             #Handling the segmented part
#             if(to_be_segmented_part.lower() == "skin"):
#                 segmented_part = confidence_mask[2]
#             elif(to_be_segmented_part.lower() == "face"):
#                 segmented_part = confidence_mask[4]
#             else:
#                 return "to_be_segmented_part is not specified"

#             # Previewing the segmented part
#             if(segmentation_result != None):
#             # plt.imshow(segmented_part.numpy_view())
#             # plt.title(f"Segmentated {to_be_segmented_part.capitalize()} Region")
#             # plt.show()
#                 pass

#             threshold = 0.5
#             binary_mask = (category_mask[2].numpy_view() > threshold).astype(np.uint8) * 255


#         # Find contours in the binary mask
#             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Determine circle parameters
#         # largest_contour = max(contours, key=cv2.contourArea)
#         # (x, y), radius = cv2.minEnclosingCircle(largest_contour)
#         # center = (int(x), int(y))
#         # radius = int(radius)

#         # Draw the circle on the original image
#         # image_with_circle = image.numpy_view().copy()
#         # cv2.circle(image_with_circle, center, radius, (0, 255, 0), 2)

#             mask = np.zeros_like(image.numpy_view().copy())
#             image_with_contours = image.numpy_view().copy()

#             cv2.drawContours(image_with_contours, contours, -1, (255, 255, 255), 2)
#             # plt.imshow(image_with_contours)
#             # plt.title(f"Segmentated {to_be_segmented_part.capitalize()} Region")
#             # plt.show()




#             return segmented_part



#         #Calling the segmenter function and create a confidence mask
#         # Essential for defining the contours / outline
#         confidence_mask = img_segmenter(image,"face")

#         # data = im.frosmarray(image.numpy_view())
#         # data.save("image.jpg")
#         print(confidence_mask)
#       # Choosing between the value of 0.5 and 0.7
#     # .7 for more accuracy
#         threshold = 0.5

#         binary_mask = (confidence_mask.numpy_view() > threshold).astype(np.uint8) * 255 # type: ignore


#         # Find contours in the binary mask
#         contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


#         mask = np.zeros_like(image.numpy_view().copy())
#         image_with_contours = image.numpy_view().copy()

#         # Find the largest contour
#         largest_contour = max(contours, key=cv2.contourArea)

#         # Draw the largest contour on the image
#         # cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)
#         cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

#         binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
#         face_extracted = cv2.bitwise_and(image.numpy_view().copy(), mask)
#         # plt.imshow(face_extracted)
#         # plt.show()


#         # data = im.fromarray(face_extracted)
#         # data.save("face_extracted.jpg")
#         print("hiiii")

#     return "hiii"

# if __name__ == '__main__':
#     app.run(debug=True)






# model_path = 'selfie_segmenter.tflite'
# base_options = python.BaseOptions(model_asset_path=model_path)


# BaseOptions = mp.tasks.BaseOptions
# ImageSegmenter = mp.tasks.vision.ImageSegmenter
# ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
# VisionRunningMode = mp.tasks.vision.RunningMode


# options = ImageSegmenterOptions(
#     base_options,
#     running_mode=VisionRunningMode.IMAGE,
#     output_category_mask=True
# )

# image = mp.Image.create_from_file('uploads/WhatsApp Image 2024-05-22 at 19.29.23.jpeg')
# def options_giver(new_options,new_image,new_to_be_segmented_part):
#     with ImageSegmenter.create_from_options(new_options) as segmenter:

#     #   #Image segmenting function
#         def img_segmenter(new_image,new_to_be_segmented_part):

#     #Ensuring no values except string is given
#             if not isinstance(new_to_be_segmented_part, str):
#                 raise ValueError("Wrong input: Expected an string.")

#             #Segmenting the image
#             segmentation_result = segmenter.segment(image)
#             category_mask = segmentation_result.confidence_masks
#             confidence_mask = segmentation_result.confidence_masks
#             segmented_part = None

#             #Handling the segmented part
#             if(new_to_be_segmented_part.lower() == "skin"):
#                 segmented_part = confidence_mask[2]
#             elif(new_to_be_segmented_part.lower() == "face"):
#                 segmented_part = confidence_mask[4]
#             else:
#                 return "to_be_segmented_part is not specified"

#             # Previewing the segmented part
#             if(segmentation_result != None):
#                 pass
#             # plt.imshow(segmented_part.numpy_view())
#             # plt.title(f"Segmentated {to_be_segmented_part.capitalize()} Region")
#             # plt.show()


#             threshold = 0.5
#             binary_mask = (category_mask[2].numpy_view() > threshold).astype(np.uint8) * 255


#         # Find contours in the binary mask
#             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Determine circle parameters
#         # largest_contour = max(contours, key=cv2.contourArea)
#         # (x, y), radius = cv2.minEnclosingCircle(largest_contour)
#         # center = (int(x), int(y))
#         # radius = int(radius)

#         # Draw the circle on the original image
#         # image_with_circle = image.numpy_view().copy()
#         # cv2.circle(image_with_circle, center, radius, (0, 255, 0), 2)

#             mask = np.zeros_like(image.numpy_view().copy())
#             image_with_contours = image.numpy_view().copy()

#             cv2.drawContours(image_with_contours, contours, -1, (255, 255, 255), 2)
#             # plt.imshow(image_with_contours)
#             # plt.title(f"Segmentated {to_be_segmented_part.capitalize()} Region")
#             # plt.show()




#             return segmented_part
    
#     return img_segmenter(new_image=new_image,new_to_be_segmented_part=new_to_be_segmented_part)



#     #Calling the segmenter function and create a confidence mask
#     # Essential for defining the contours / outline


# # data = im.fromarray(image.numpy_view())
# # data.save("image.jpg")


if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4444)))