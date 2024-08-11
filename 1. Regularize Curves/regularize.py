import cv2
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import os
from figures import main

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def bezier_curve(points, n_points=100):
    def bernstein_poly(i, n, t):
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def bezier(t, points):
        n = len(points) - 1
        return sum(bernstein_poly(i, n, t) * np.array(pt) for i, pt in enumerate(points))

    t = np.linspace(0.0, 1.0, n_points)
    curve = np.array([bezier(ti, points) for ti in t])
    return curve.astype(np.int32)

def smooth_contours(paths, n_points=100):
    smoothed_paths = []
    for path in paths:
        smoothed_path = []
        for contour in path:
            curve = bezier_curve(contour, n_points=n_points)
            smoothed_path.append(curve)
        smoothed_paths.append(smoothed_path)
    return smoothed_paths

def draw_paths_on_image(image, paths, color=(0, 0, 0), thickness=2):
    for path in paths:
        for contour in path:
            cv2.polylines(image, [contour], isClosed=False, color=color, thickness=thickness)

def calculate_angle(pt1, pt2, pt3):
    vec1 = np.array(pt1) - np.array(pt2)
    vec2 = np.array(pt3) - np.array(pt2)
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def is_valid_rectangle(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if 9 >= len(approx) >= 4:
        angles = []
        for i in range(len(approx)):
            pt1 = approx[i - 2][0]
            pt2 = approx[i - 1][0]
            pt3 = approx[i][0]
            angle = calculate_angle(pt1, pt2, pt3)
            angles.append(angle)
        right_angle_count = sum(80 <= angle <= 100 for angle in angles)
        if right_angle_count >= 3:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.9 < aspect_ratio < 1.1:
                return 'Square'
            else:
                return 'Rectangle'
    return None

def is_valid_triangle(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 3:
        return True
    return False

def is_circle(contour, circularity_threshold=0.7):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    area = cv2.contourArea(contour)
    if area == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity > circularity_threshold

def is_valid_star(contour, min_points=11, angle_threshold=10, point_variation_threshold=2):
    if len(contour) < min_points:
        return False
    points = contour[:, 0, :]
    num_points = len(points)
    angles = []
    distances = []
    for i in range(num_points):
        pt1 = points[i]
        pt2 = points[(i + 1) % num_points]
        pt3 = points[(i + 2) % num_points]
        angle = calculate_angle(pt1, pt2, pt3)
        angles.append(angle)
        distances.append(np.linalg.norm(pt1 - pt2))
    if is_circle(contour):
        return False
    angle_variation = np.std(angles)
    print("angle_variation", angle_variation)
    if 14>angle_variation > angle_threshold:
        print("angle_variation must be greater than angle_threshold")
        distance_variation = np.std(distances)
        print("distance_variation", distance_variation)
        if distance_variation > point_variation_threshold:
            return True
    return False

def draw_regular_shape(image, contour, shape, scale_triangle=1.0, scale_square=1.0, scale_rectangle=1.0, scale_star=1.0):
    x, y, w, h = cv2.boundingRect(contour)
    if shape == 'Triangle':
        pts = np.array([[x + w // 2, y], 
                        [x + int(w * (1 - scale_triangle) / 2), y + h], 
                        [x + int(w * (1 + scale_triangle) / 2), y + h]], 
                        np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
    elif shape == 'Square':
        scaled_w = int(w * scale_square)
        scaled_h = int(h * scale_square)
        top_left = (x + (w - scaled_w) // 2, y + (h - scaled_h) // 2)
        bottom_right = (top_left[0] + scaled_w, top_left[1] + scaled_h)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    elif shape == 'Rectangle':
        scaled_w = int(w * scale_rectangle)
        scaled_h = int(h * scale_rectangle)
        top_left = (x + (w - scaled_w) // 2, y + (h - scaled_h) // 2)
        bottom_right = (top_left[0] + scaled_w, top_left[1] + scaled_h)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    elif shape == 'Star':
        center = (x + w // 2, y + h // 2)
        radius = int(w // 2 * scale_star)
        draw_star_outline(image, center, radius, (0, 255, 0), 2)

def draw_star_outline(image, center, radius, color=(0, 255, 0), thickness=2):
    points = []
    for i in range(10):
        angle = i * np.pi / 5
        r = radius if i % 2 == 0 else radius / 2
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] - r * np.sin(angle))
        points.append((x, y))
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

def print_contour_properties(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    print(f'Contour Properties:')
    print(f'  Number of vertices: {num_vertices}')
    print(f'  Bounding rectangle: x={x}, y={y}, width={w}, height={h}')
    print(f'  Area: {area}')
    print(f'  Perimeter: {perimeter}')

def main1(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.csv':
        paths = read_csv(file_path)
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        smoothed_paths = smooth_contours(paths, n_points=100)
        draw_paths_on_image(image, smoothed_paths)
    else:
        image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        shape = None
        if is_valid_triangle(contour):
            shape = 'Triangle'
        elif (shape := is_valid_rectangle(contour)):
            pass
        elif is_valid_star(contour):
            shape = 'Star'
        if shape:
            draw_regular_shape(image, contour, shape)
    cv2.imwrite('final_image_with_shapes.png', image)
    main('final_image_with_shapes.png', 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main1('isolated.csv')