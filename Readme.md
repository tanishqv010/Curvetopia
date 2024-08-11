# Shape Detection and Processing

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

![Regularization Example](examples/regularization_output.png)

### Usage

To regularize shapes:

1. **Input**: Provide the path to the image or CSV file containing polyline data, or directly provide the image in PNG or JPG format.
2. **Command**: Run the code to identify irregular shapes that can be converted into regular shapes and then transform them into more geometrically accurate forms.

## Part 2: Symmetry Detection

### Description

Symmetry detection analyzes shapes within an image to identify axes of symmetry. This process includes detecting symmetries at various angles (0-360 degrees) to understand the inherent balance in the shapes. The shapes are first converted into polylines, then the contours are mapped, and their reflections are analyzed to detect symmetrical properties.

### Example Output

![Symmetry Detection Example](examples/symmetry_output.png)

### Usage

To perform symmetry detection:

1. **Input**: Provide the path to the processed image with regularized shapes.
2. **Command**: Execute the code to detect and analyze symmetries in the 2D shapes and doodles.

## Part 3: Occlusion Completion

### Description

The occlusion completion process identifies and reconstructs occluded or missing parts of shapes within an image. This ensures that shapes are completed while maintaining their original structure and integrity. The process involves converting the image into polylines, smoothing and fitting curves, and then identifying contours that can be completed into regular curves. The best-fitting regular shapes are selected based on how well they match the original image contours and other criteria.

### Example Output

![Occlusion Completion Example](examples/occlusion_output.png)

### Usage

To perform occlusion completion:

1. **Input**: Provide the path to the processed image that needs occlusion completion.
2. **Command**: Execute the code to detect and complete occluded parts of shapes.

## Additional Resources

- **Demo**: [View the demo of the shape detection and processing system](https://your-demo-link.com)
- **Google Colab**: [Open the Google Colab notebook for interactive execution](https://colab.research.google.com/your-notebook-link)
