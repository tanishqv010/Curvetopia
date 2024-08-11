import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_ellipse_points(ellipse, num_points=100):
    if not isinstance(ellipse, tuple) or len(ellipse) != 3:
        raise ValueError("Ellipse is not in the expected format.")
    (cx, cy), (MA, ma), angle = ellipse
    t = np.linspace(0, 2 * np.pi, num_points)
    x = cx + (MA / 2) * np.cos(t) * np.cos(np.deg2rad(angle)) - (ma / 2) * np.sin(t) * np.sin(np.deg2rad(angle))
    y = cy + (MA / 2) * np.cos(t) * np.sin(np.deg2rad(angle)) + (ma / 2) * np.sin(t) * np.cos(np.deg2rad(angle))
    return np.column_stack((x, y))

def generate_circle_points(circle, num_points=100):
    (cx, cy), radius = circle
    t = np.linspace(0, 2 * np.pi, num_points)
    x = cx + radius * np.cos(t)
    y = cy + radius * np.sin(t)
    return np.column_stack((x, y))

def generate_rectangle_points(rectangle, num_points=100):
    if not isinstance(rectangle, tuple) or len(rectangle) != 3:
        raise ValueError("Rectangle is not in the expected format.")
    (cx, cy), (w, h), angle = rectangle
    t = np.linspace(0, 2 * np.pi, num_points)
    theta = np.deg2rad(angle)
    corners = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2)
    ]
    x = [corner[0] for corner in corners] + [corners[0][0]]
    y = [corner[1] for corner in corners] + [corners[0][1]]
    return np.column_stack((x, y))

def find_closest_point_on_contour(point, contour):
    min_dist = float('inf')
    closest_point = None
    for pt in contour:
        dist = np.linalg.norm(np.array(point) - np.array(pt[0]))
        if dist < min_dist:
            min_dist = dist
            closest_point = pt[0]
    return closest_point, min_dist

def find_all_distances_to_contour(shape_points, contour):
    distances = []
    for pt in shape_points:
        closest_point, distance = find_closest_point_on_contour(pt, contour)
        distances.append(distance)
    return distances

def calculate_global_distance_threshold(contours, percentage=0.3, shape_type='ellipse'):
    all_distances = []
    for contour in contours:
        if len(contour) >= 5:
            try:
                if shape_type == 'ellipse':
                    shape = cv2.fitEllipse(contour)
                    shape_points = generate_ellipse_points(shape)
                elif shape_type == 'circle':
                    (cx, cy), radius = cv2.minEnclosingCircle(contour)
                    shape_points = generate_circle_points(((cx, cy), radius))
                elif shape_type == 'rectangle':
                    rect = cv2.minAreaRect(contour)
                    shape_points = generate_rectangle_points(rect)
                distances = find_all_distances_to_contour(shape_points, contour)
                all_distances.extend(distances)
            except cv2.error as e:
                print(f"Error fitting shape: {e}")
    all_distances.sort()
    num_distances = len(all_distances)
    min_percentage_count = int(num_distances * percentage)
    min_distances = all_distances[:min_percentage_count]
    threshold = np.mean(min_distances) if min_distances else 0
    print(f"Global {shape_type} distance threshold: {1.2*threshold}")
    return 1.1*threshold

def count_continuous_points_within_threshold(shape_points, contour, distance_threshold):
    """Count the number of continuous points within the distance threshold."""
    continuous_count = 0
    max_continuous = 0
    for pt in shape_points:
        closest_point, distance = find_closest_point_on_contour(pt, contour)
        if distance <= distance_threshold:
            continuous_count += 1
        else:
            if continuous_count > max_continuous:
                max_continuous = continuous_count
            continuous_count = 0
    if continuous_count > max_continuous:
        max_continuous = continuous_count
    return max_continuous

def calculate_ellipse_area(ellipse):
    _, (MA, ma), _ = ellipse
    return np.pi * (MA / 2) * (ma / 2)


def detect_shapes(image_path, min_major_axis=30, max_major_axis=1000, min_aspect_ratio=0.5, max_aspect_ratio=2,
                  ellipse_percentage=0.3, circle_percentage=0.3, rectangle_percentage=0.3, min_continuous_points=4,
                  min_circle_radius=20, min_side_length=30, area_difference_threshold=0.2):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=False)
    ellipse_distance_threshold = calculate_global_distance_threshold(contours, ellipse_percentage, shape_type='ellipse')
    circle_distance_threshold = calculate_global_distance_threshold(contours, circle_percentage, shape_type='circle')
    rectangle_distance_threshold = calculate_global_distance_threshold(contours, rectangle_percentage, shape_type='rectangle')
    valid_ellipses = []
    valid_circles = []
    valid_rectangles = []
    for contour in contours:
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse
                aspect_ratio = MA / ma
                if min_major_axis <= MA <= max_major_axis and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                    ellipse_points = generate_ellipse_points(ellipse)
                    continuous_points_count = count_continuous_points_within_threshold(ellipse_points, contour, ellipse_distance_threshold)
                    if continuous_points_count >= min_continuous_points:
                        is_overlapping = any(cv2.pointPolygonTest(cv2.ellipse2Poly((int(ve[0][0]), int(ve[0][1])), 
                                        (int(ve[1][0] / 2), int(ve[1][1] / 2)), int(ve[2]), 0, 360, 5), 
                                        (int(x), int(y)), False) >= 0 for ve in valid_ellipses)
                        if not is_overlapping:
                            valid_ellipses.append(ellipse)
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                if radius >= min_circle_radius:
                    circle_points = generate_circle_points(((cx, cy), radius))
                    continuous_points_count = count_continuous_points_within_threshold(circle_points, contour, circle_distance_threshold)
                    if continuous_points_count >= min_continuous_points:
                        is_overlapping = any(
                            cv2.pointPolygonTest(
                                cv2.ellipse2Poly(
                                    (int(vc[0][0]), int(vc[0][1])),
                                    (int(radius), int(radius)),
                                    0,
                                    0,
                                    360,
                                    5
                                ),
                                (int(cx), int(cy)),
                                False
                            ) >= 0
                            for vc in valid_circles
                        )
                        if not is_overlapping:
                            valid_circles.append(((cx, cy), radius))
            except cv2.error as e:
                print(f"Error fitting ellipse or circle: {e}")
        if len(contour) >= 4:
            try:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                rect_points = generate_rectangle_points(rect)
                continuous_points_count = count_continuous_points_within_threshold(rect_points, contour, rectangle_distance_threshold)
                if continuous_points_count >= min_continuous_points:
                    is_overlapping = any(
                        cv2.pointPolygonTest(
                            np.array([box], dtype=np.int32),
                            (int(rect[0][0]), int(rect[0][1])),
                            False
                        ) >= 0
                        for vrect in valid_rectangles
                    )
                    if not is_overlapping:
                        valid_rectangles.append(rect)
            except cv2.error as e:
                print(f"Error fitting rectangle: {e}")
    if len(valid_ellipses) > 1:
        largest_ellipse = max(valid_ellipses, key=lambda e: calculate_ellipse_area(e))
        valid_ellipses.remove(largest_ellipse)
        second_largest_ellipse = max(valid_ellipses, key=lambda e: calculate_ellipse_area(e), default=None)
        largest_area = calculate_ellipse_area(largest_ellipse)
        if second_largest_ellipse:
            second_largest_area = calculate_ellipse_area(second_largest_ellipse)
            relative_difference = abs(largest_area - second_largest_area) / max(largest_area, second_largest_area)
            if relative_difference > area_difference_threshold:
                cv2.ellipse(image, largest_ellipse, (0, 255, 0), 4)
                cx, cy = largest_ellipse[0]
                cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            else:
                for ellipse in [largest_ellipse, second_largest_ellipse]:
                    cv2.ellipse(image, ellipse, (0, 0, 0), 2)
                    cx, cy = ellipse[0]
                    cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        else:
            cv2.ellipse(image, largest_ellipse, (0, 0, 0), 2)
            cx, cy = largest_ellipse[0]
            cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    elif len(valid_ellipses) == 1:
        cv2.ellipse(image, valid_ellipses[0], (0, 255, 0), 2)
        cx, cy = valid_ellipses[0][0]
        cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    for circle in valid_circles:
        (cx, cy), radius = circle
        cv2.circle(image, (int(cx), int(cy)), int(radius), (0, 255, 0), 4)
        cv2.circle(image, (int(cx), int(cy)), 5, (255, 0, 0), -1)
    for rect in valid_rectangles:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 4)
        cx, cy = rect[0]
        cv2.circle(image, (int(cx), int(cy)), 5, (255, 0, 0), -1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Shapes with Centroids")
    plt.axis("off")
    plt.show()
    output_image_path = 'output_image_with_shapes.png'
    cv2.imwrite(output_image_path, image)
    white_background = np.ones_like(edges) * 255
    cv2.drawContours(white_background, contours, -1, (0, 0, 0), 3)
    plt.figure()
    plt.imshow(white_background, cmap='gray')
    plt.title("Contours on White Background")
    plt.axis("off")
    plt.show() 

def read_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    paths = []
    for path_id in np.unique(data[:, 0]):
        path_data = data[data[:, 0] == path_id][:, 1:]
        path_segments = []
        for segment_id in np.unique(path_data[:, 0]):
            segment_data = path_data[path_data[:, 0] == segment_id][:, 1:]
            path_segments.append(segment_data)
        paths.append(path_segments)
    return paths

def image_from_paths(paths, image_size=(500, 500)):
    img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255  # Create a white background
    for path in paths:
        for segment in path:
            for i in range(len(segment) - 1):
                cv2.line(img, tuple(segment[i].astype(int)), tuple(segment[i+1].astype(int)), (0, 0, 0), 2)
    return img

def main(path, iscsv):
    min_major_axis = 70
    max_major_axis = 1000
    min_aspect_ratio = 0.2
    max_aspect_ratio = 2
    ellipse_percentage = 0.47
    circle_percentage = 0.16
    rectangle_percentage = 0.3
    min_continuous_points = 5
    min_circle_radius = 50
    min_side_length = 30
    area_difference_threshold = 0.2
    if(iscsv):
        path_XYs = read_csv(path)
        image_size = (500, 500)
        image = image_from_paths(path_XYs, image_size=image_size)
        if image is not None:
            save_status = cv2.imwrite('shapes_image.png', image)
            if save_status:
                print("Image saved successfully.")
            else:
                print("Failed to save the image.")
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Generated Image from CSV Data")
            plt.axis('off')
            plt.show()
            detect_shapes('shapes_image.png', min_major_axis, max_major_axis, min_aspect_ratio, max_aspect_ratio, 
                                    ellipse_percentage, circle_percentage, rectangle_percentage, min_continuous_points,
                                    min_circle_radius, min_side_length, area_difference_threshold)
    else:
        detect_shapes(path, min_major_axis, max_major_axis, min_aspect_ratio, max_aspect_ratio, 
                                    ellipse_percentage, circle_percentage, rectangle_percentage, min_continuous_points,
                                    min_circle_radius, min_side_length, area_difference_threshold)