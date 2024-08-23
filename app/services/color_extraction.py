import numpy as np
import colorgram
from PIL import Image as im



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