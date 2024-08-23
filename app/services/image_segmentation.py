import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from PIL import Image as im
import numpy as np


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

    image = mp.Image.create_from_file('uploads\WhatsApp Image 2024-08-10 at 14.22.17_36837c06.jpg')

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