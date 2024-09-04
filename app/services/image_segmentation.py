import os
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from PIL import Image as im
import numpy as np





def segmenter(filename,to_be_segmented_part = "face"):
    from app import create_app
    app = create_app()


    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



    file_path = os.path.join(os.path.dirname(__file__), '..', "models","selfie_segmenter.tflite")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    segmented_image_path = os.path.join(app.config['SEGMENTED_IMGS'],to_be_segmented_part,filename)
    # image_path = os.path.join(app.config['UPLOAD_FOLDER'],"53a013b7b03234d99cb20cf346f77b88.jpg")
    # segmented_image_path = os.path.join(app.config['SEGMENTED_IMGS'],to_be_segmented_part,"53a013b7b03234d99cb20cf346f77b88.jpg")



    # Specify the path of segmenter model
    model_path =file_path
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

    image = mp.Image.create_from_file(image_path)

    # with ImageSegmenter.create_from_options(options) as segmenter:
    segmenter = ImageSegmenter.create_from_options(options)
    #   #Image segmenting function
    def img_segmenter(image,to_be_segmented_part):
    # Ensuring no values except string is given
        if not isinstance(to_be_segmented_part, str):
            raise ValueError("Wrong input: Expected an string.")
    
        
    # Segmenting the image
        segmentation_result = segmenter.segment(image)

        confidence_mask = segmentation_result.confidence_masks
        segmented_part = None

        #Handling the segmented part
        if(to_be_segmented_part.lower() == "skin"):
            segmented_part = confidence_mask[2]
        elif(to_be_segmented_part.lower() == "face"):
            segmented_part = confidence_mask[3]
        elif(to_be_segmented_part.lower() == 'dress'):
             segmented_part = confidence_mask[4]
        else:
            return "to_be_segmented_part is not specified"
        return segmented_part

        
    confidence_mask = img_segmenter(image,to_be_segmented_part)
    # data = im.fromarray((confidence_mask.numpy_view() * 255).astype(np.uint8))
    # data.save("segmented.jpg")

    # Choosing between the value of 0.5 and 0.7
    # .7 for more accuracy
    threshold = 0.5

    binary_mask = (confidence_mask.numpy_view() > threshold).astype(np.uint8) * 255

    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    # resized_image = confidence_mask.numpy_view().copy().astype(np.uint8) * 255
    new_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGBA2RGB)

    image_size = image.numpy_view().shape
    image_type = image.numpy_view().dtype

    mask_size = binary_mask.shape
    mask_type = binary_mask.dtype

    face_extracted = cv2.bitwise_and(new_image, binary_mask)
    


    data = im.fromarray(face_extracted)
    data.save(segmented_image_path)


    # return str(confidence_mask.numpy_view())
    return str((image_size,image_type,mask_size,mask_type))





def clothe_segmenter(filename,dress_type,dress_gender):
        from app import create_app
        app = create_app()
        file_path = os.path.join(app.config['DRESS_IMGS'],f'{dress_gender}',f'{dress_type}',filename)
        segemented = segmenter(filename=file_path,to_be_segmented_part='dress')
        print(segemented)
        


        
        

