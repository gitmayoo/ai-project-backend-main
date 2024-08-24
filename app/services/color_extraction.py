import os
from typing import Counter
from flask import jsonify
import numpy as np
import colorgram
from PIL import Image as im


# class Rgb:
#     def __init__(self, r, g, b):
#         self.r = r
#         self.g = g
#         self.b = b

#     def to_tuple(self):
#         return (self.r, self.g, self.b)


def color_extractor(filename):
    # Load the extracted image
    # image = im.open("/Users/karthi/Development/mayoo-project/ai-project-backend-main/app/segmented_images/face/Larthi_20240824_125901.jpg")
    from app import create_app
    app = create_app()
    image_path = os.path.join(app.config['SEGMENTED_IMGS'],"face",filename)
    print(image_path)
    image = im.open(image_path)
    colors = colorgram.extract(image,10)

    color_tuples = [(color.rgb.r, color.rgb.g, color.rgb.b) for color in colors]

   
        # Define base skin tone colors (these are just examples, you may need to adjust)
    skin_tones = [
        (246,737, 228) , 
        (243, 731, 219),
        (247, 134, 208),
        (234, 218, 186),
        (215, 189, 150),
        (160,126, 86),    
        (130, 92, 67),
        (96, 65, 52),
        (58, 49, 42),
        (41, 36, 32),
        # Add more tones as needed
    ]

    # Function to calculate Euclidean distance between two RGB colors
    def color_distance(c1, c2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    # Function to determine if a color is within the skin tone range
    def is_skin_tone(color, skin_tones, threshold=40):
        for tone in skin_tones:
            if color_distance(color, tone) < threshold:
                return True
        return False


    def get_most_common_skin_tone(extracted_colors, skin_tones):
        # Filter colors to include only those within the skin tone range
        skin_tone_colors = [color for color in extracted_colors if is_skin_tone(color, skin_tones)]

        # Count occurrences of each color
        color_counts = Counter(skin_tone_colors)
        
        # Find the most common color
        most_common_color, count = color_counts.most_common(1)[0] if color_counts else (None, 0)
        
        return most_common_color


   
    

        # Function to calculate Euclidean distance from black
    def distance_from_black(color):
        black = (0, 0, 0)
        return np.sqrt(sum((c - b) ** 2 for c, b in zip(color, black)))

    # Function to filter out colors close to black
    def filter_colors_near_black(colors, threshold=50):
        return [color for color in colors if distance_from_black(color) > threshold]
    
    print(filter_colors_near_black(color_tuples))

    print(get_most_common_skin_tone(filter_colors_near_black(color_tuples),skin_tones))

    def find_closest_skin_tone(most_common_color, skin_tones):
        if most_common_color is None:
            return None
        closest_tone = min(skin_tones, key=lambda tone: color_distance(most_common_color, tone))
        return closest_tone
    print(find_closest_skin_tone(get_most_common_skin_tone(filter_colors_near_black(color_tuples),skin_tones),skin_tones))

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    hex  =  rgb_to_hex(find_closest_skin_tone(get_most_common_skin_tone(filter_colors_near_black(color_tuples),skin_tones),skin_tones))
    return hex

    # # Function to extract colors from an image
    # def extract_colors_from_image(image_path, num_colors=10):
    #     colors = colorgram.extract(image_path, num_colors)
    #     return [color.rgb for color in colors]

    # # Function to append colors from colorgram to existing colors, excluding those close to black
    # def append_and_filter_colors(existing_colors, image_path, num_colors=10, black_threshold=50):
    #     new_colors = extract_colors_from_image(image_path, num_colors)
        
    #     # Filter new colors to include only those within the skin tone range
    #     filtered_colors = [color for color in new_colors if is_skin_tone(color, skin_tones)]
        
    #     # Further filter out colors close to black
    #     filtered_colors = filter_colors_near_black(filtered_colors, black_threshold)
        
    #     # Append filtered colors to the existing list
    #     existing_colors.extend(filtered_colors)
    #     return existing_colors



    # # # Function to calculate Euclidean distance between two RGB colors
    # # def color_distance(c1, c2):


    # #     # return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    # #     return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    # # # Function to determine if a color is within the skin tone range
    # # def is_skin_tone(color, skin_tones, threshold=40):
    # #     for tone in skin_tones:
    # #         if color_distance(color, tone) < threshold:
    # #             return True
    # #     return False
    # def color_distance(c1, c2):
    #     # Convert Rgb objects to tuples if necessary
    #     if isinstance(c1, Rgb):
    #         c1 = c1.to_tuple()
    #         print("true")
    #     if isinstance(c2, Rgb):
    #         print("true")

    #         c2 = c2.to_tuple()
        
    #     # Ensure c1 and c2 are tuples or lists of numeric values
    #     if not (isinstance(c1, (tuple, list)) and isinstance(c2, (tuple, list))):
    #         raise ValueError("c1 and c2 must be tuples or lists of numeric values")
        
    #     # Compute Euclidean distance
    #     return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    # def is_skin_tone(color, skin_tones, threshold=40):
    #     for tone in skin_tones:
    #         if color_distance(color, tone) < threshold:
    #             return True
    #     return False
    # # Extract colors from the image
    
    # colors = colorgram.extract(image, 50)
    # # Filter the colors to keep only those that are close to a skin tone
    # skin_colors = [color.rgb for color in colors if is_skin_tone(color.rgb, skin_tones)]

    
    # # Find the most common color
    # most_common_color = Counter(skin_colors).most_common(1)[0][0]
    # print("Most Frequent Color:", most_common_color)

    # def closest_skin_tone(color, skin_tones):
    #     closest_tone = min(skin_tones, key=lambda tone: color_distance(color, tone))
    #     return closest_tone

    # color = closest_skin_tone(skin_colors,skin_tones)
    # return str(color)

   


    # sorted_colors = sorted(colors, key=lambda color: color.proportion, reverse=False)

    # # The first color in the sorted list is the most prominent by proportion
    # most_prominent_color = sorted_colors[0]
    # print("Prominent")
    # print(most_prominent_color.rgb)
    # print("Prominent")

    # extracted_colors = []
    # for color in colors:
    #     if color.rgb.r >= 10 :  # Check if all color components are non-zero
    #         print(color.rgb)
    #         extracted_colors.append(color.rgb)


    # mean_color = np.mean([np.array(color) for color in extracted_colors], axis=0)
    # mean_color = mean_color.astype(int)  # Convert to integers for display

   

    # # Add mean color patch
    # normalized_mean_color = (mean_color[0] / 255, mean_color[1] / 255, mean_color[2] / 255)






  