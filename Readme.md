# Shape Detection and Processing

Team Name: Azmuth Alliance </br>
Members: Preyanshu Mishra

## Overview

This project focuses on three core components designed to analyze and process shapes in images or polyline data:

1. **Regularization**
2. **Symmetry Detection**
3. **Occlusion Completion**

Each component ensures that shapes are accurately represented, symmetrical, and fully restored even when occluded.

## Part 1: Regularization

### Description

The regularization process converts hand-drawn or irregular shapes from images or polyline data into more geometrically precise forms. This transformation adheres to strict mathematical, geometric, and symmetrical principles. The process involves detecting contours, classifying them into regular shapes based on their mathematical properties, and then overlaying them with their geometrically accurate counterparts.

### Examples

![Regularize](https://github.com/user-attachments/assets/b041e67e-244d-4e91-9848-5b50d0938bfa)
![image](https://github.com/user-attachments/assets/38df1c5a-bdc2-49db-9340-3a7a92221f7e)






### Usage

To regularize shapes:

1. **Input**: Provide the path to the image or CSV file containing polyline data, or directly provide the image in PNG or JPG format.
2. **Command**: Run the code to identify irregular shapes that can be converted into regular shapes and then transform them into more geometrically accurate forms.

## Part 2: Symmetry Detection

### Description

Symmetry detection analyzes shapes within an image to identify axes of symmetry. This process includes detecting symmetries at various angles (0-360 degrees) to understand the inherent balance in the shapes. The shapes are first converted into polylines, then the contours are mapped, and their reflections are analyzed to detect symmetrical properties.

### Example Output

![Arrow Symmetry](https://github.com/user-attachments/assets/4f900261-9c56-49ef-9975-baf0ae2bb2a9)
![image](https://github.com/user-attachments/assets/ae1cec01-66d1-40ba-8cf4-14120f6369b0)
![image](https://github.com/user-attachments/assets/f3c98897-0850-4bd3-93e5-d09cb2ba30fd)



### Usage

To perform symmetry detection:

1. **Input**: Provide the path to the processed image with regularized shapes.
2. **Command**: Execute the code to detect and analyze symmetries in the 2D shapes and doodles.

## Part 3: Occlusion Completion

### Description

The occlusion completion process identifies and reconstructs occluded or missing parts of shapes within an image. This ensures that shapes are completed while maintaining their original structure and integrity. The process involves converting the image into polylines, smoothing and fitting curves, and then identifying contours that can be completed into regular curves. The best-fitting regular shapes are selected based on how well they match the original image contours and other criteria.

### Example Output

![Pin Occlusion](https://github.com/user-attachments/assets/1fd20906-2df3-4c8e-a75e-bd2a6f2b9000)
![image](https://github.com/user-attachments/assets/bf6ae19e-fc07-4b69-8d12-1c8d763283bb)
![image](https://github.com/user-attachments/assets/ffb87cb2-3e18-4996-97cf-02fb58443554)
![image](https://github.com/user-attachments/assets/9dc89eb4-0f7f-4b94-8fd7-579f591d5512)
![image](https://github.com/user-attachments/assets/a56eaf94-5a98-42d1-8293-491ea8a5968a)





### Usage

To perform occlusion completion:

1. **Input**: Provide the path to the processed image that needs occlusion completion.
2. **Command**: Execute the code to detect and complete occluded parts of shapes.

## Additional Resources

- **Demo**: [View the demo of the shape detection and processing system](https://drive.google.com/drive/folders/1VFv4FwevESxdGaDDQXtsQz8vFNlBcWib?usp=sharing)
- **Google Colab**: [Open the Google Colab notebook for interactive execution](https://colab.research.google.com/drive/17ZL4JkqX7e9FCS6UppqX0U2l0X9fhZt-?usp=sharing)
