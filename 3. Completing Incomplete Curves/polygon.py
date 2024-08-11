import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_angle_close(angle, target_angle, tolerance=10):
    return abs(angle - target_angle) < tolerance

def draw_rectangle(image, corners, color=(0, 0, 255), thickness=2):
    if len(corners) < 4:
        return
    for i in range(4):
        pt1 = tuple(map(int, corners[i]))
        pt2 = tuple(map(int, corners[(i + 1) % 4]))
        cv2.line(image, pt1, pt2, color, thickness)

def is_valid_rectangle(corners, tolerance=10):
    angles = []
    if len(corners) < 4:
        return
    for i in range(4):
        pt1 = corners[i]
        pt2 = corners[(i + 1) % 4]
        pt3 = corners[(i + 2) % 4]
        vec1 = pt2 - pt1
        vec2 = pt3 - pt2
        angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
        angle = np.degrees(angle)
        angle = abs(angle) % 180
        angles.append(angle)
    close_to_90_count = sum(is_angle_close(angle, 90, tolerance) for angle in angles) 
    return close_to_90_count >= 2

def is_valid_triangle(corners, tolerance=10):
    if len(corners) < 3:
        return False
    angles = []
    for i in range( 3):
        pt1 = corners[i]
        pt2 = corners[(i + 1) % 3]
        pt3 = corners[(i + 2) % 3]
        vec1 = pt2 - pt1
        vec2 = pt3 - pt2
        angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
        angle = np.degrees(angle)
        angle = abs(angle) % 180
        angles.append(angle)
        print(angles,"triangle")
    close_to_90_count = sum(is_angle_close(angle, 90, 20) for angle in angles)
    close_to_60_count = sum(is_angle_close(angle, 60, 30) for angle in angles)
    return close_to_60_count >= 2 and close_to_90_count == 0

def perpendicular_distance(point, line_start, line_end):
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    point = np.array(point)
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point - line_start)
    line_vec_normalized = line_vec / line_len
    projection = np.dot(point_vec, line_vec_normalized)
    if projection < 0:
        nearest_point = line_start
    elif projection > line_len:
        nearest_point = line_end
    else:
        nearest_point = line_start + projection * line_vec_normalized
    return np.linalg.norm(point - nearest_point)

def find_closest_point_on_contour(point, contour):
    min_dist = float('inf')
    closest_point = None
    for i in range(len(contour)):
        line_start = contour[i][0]
        line_end = contour[(i + 1) % len(contour)][0]
        dist = perpendicular_distance(point, line_start, line_end)
        if dist < min_dist:
            min_dist = dist
            closest_point = line_start
    return closest_point, min_dist

def find_all_distances_to_contour(shape_points, contour):
    distances = []
    contour = [tuple(pt[0]) for pt in contour]
    num_contour_points = len(contour)
    for pt in shape_points:
        pt = tuple(pt)
        min_distance = float('inf')
        for i in range(num_contour_points):
            line_start = contour[i]
            line_end = contour[(i + 1) % num_contour_points]
            distance = perpendicular_distance(pt, line_start, line_end)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
    return distances

def count_continuous_points_within_threshold(shape_points, contour, distance_threshold=8):
    continuous_count = 0
    max_continuous = 0
    for pt in shape_points:
        pt = tuple(pt)
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

def generate_rectangle_points(rectangle, num_points=300):
    if not isinstance(rectangle, tuple) or len(rectangle) != 3:
        raise ValueError("Rectangle is not in the expected format.")
    (cx, cy), (w, h), angle = rectangle
    corners = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2)
    ]
    side_lengths = [
        np.linalg.norm(np.array(corners[0]) - np.array(corners[1])),
        np.linalg.norm(np.array(corners[1]) - np.array(corners[2])),
        np.linalg.norm(np.array(corners[2]) - np.array(corners[3])),
        np.linalg.norm(np.array(corners[3]) - np.array(corners[0]))
    ]
    perimeter = sum(side_lengths)
    points_per_side = num_points // 4
    remainder_points = num_points % 4
    if len(corners) < 4:
        return
    points = []
    for i in range(4):
        start_point = np.array(corners[i])
        end_point = np.array(corners[(i + 1) % 4])
        side_length = side_lengths[i]
        for j in range(points_per_side + (1 if i < remainder_points else 0)):
            alpha = j / (points_per_side + 1)
            point = start_point * (1 - alpha) + end_point * alpha
            points.append(point)
    return np.array(points)

def calculate_global_distance_threshold(contours, percentage=1):
    all_distances = []
    for contour in contours:
        if len(contour) >= 4:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) <= 10:
                rect = cv2.minAreaRect(approx)
                shape_points = generate_rectangle_points(rect)
                distances = find_all_distances_to_contour(shape_points, approx)
                all_distances.extend(distances)
    if all_distances:
        all_distances.sort()
        num_distances = len(all_distances)
        min_percentage_count = int(num_distances * percentage)
        min_distances = all_distances[:min_percentage_count]
        mean_distance = np.mean(min_distances)
        return mean_distance - 0.1 * mean_distance, mean_distance + 0.1 * mean_distance
    return 0, 0

def calculate_centroid(points):
    return np.mean(points, axis=0)

def calculate_angle(vector):
    return np.arctan2(vector[1], vector[0])

def create_equilateral_triangle(centroid, vertex, scale_factor=1.0):
    side_length = np.linalg.norm(centroid - vertex) * 2 * scale_factor
    angle_offset = np.arctan2(vertex[1] - centroid[1], vertex[0] - centroid[0])
    vertices = []
    for i in range(3):
        angle = angle_offset + i * 2 * np.pi / 3
        vertex = centroid + side_length * np.array([np.cos(angle), np.sin(angle)]) / np.sqrt(3)
        vertices.append(vertex)
    return np.array(vertices, dtype=int)

def calculate_angle_between_vectors(vec1, vec2):
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def generate_triangle_points(triangle, num_points=100):
    points_per_side = num_points // 3
    remainder_points = num_points % 3
    if len(triangle) < 3:
        return
    points = []
    for i in range( 3):
        start_point = triangle[i]
        end_point = triangle[(i + 1) % 3]
        for j in range(points_per_side + (1 if i < remainder_points else 0)):
            alpha = j / (points_per_side + 1)
            point = start_point * (1 - alpha) + end_point * alpha
            points.append(point)
    return np.array(points)


def align_triangle(detected_triangle_points, centroid):
    side_length = cv2.norm(detected_triangle_points[0] - detected_triangle_points[1])
    height = (np.sqrt(3) / 2) * side_length
    perfect_triangle = np.array([
        [0, -2 * height / 3],
        [-side_length / 2, height / 3],
        [side_length / 2, height / 3]
    ])
    angle = calculate_orientation(detected_triangle_points)
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle-2.4, 2.3)
    rotated_triangle = cv2.transform(np.array([perfect_triangle]), rotation_matrix)[0]
    aligned_triangle = rotated_triangle + centroid
    return np.int0(aligned_triangle)

def calculate_orientation(points):
    vector = points[1] - points[0]
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi 
    return angle

def detect_triangles_rectangles_squares(image, min_continuous_points_rect=0, min_continuous_points_tri=0, triangle_area_percentage=0.02):
    firstRect = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = image.copy()
    min_threshold, max_threshold = calculate_global_distance_threshold(contours)
    image_area = image.shape[0] * image.shape[1]
    triangle_area_threshold = image_area * triangle_area_percentage
    valid_rectangles = []
    valid_triangles = []
    firstRect = []
    rectangle_count = 0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) <= 6 and is_valid_triangle(np.array([pt[0] for pt in approx])):
            area = cv2.contourArea(approx)
            if area > triangle_area_threshold:
                continuous_points_count = count_continuous_points_within_threshold(np.array([pt[0] for pt in approx]), approx, min_threshold)
                centroid = calculate_centroid(np.array([pt[0] for pt in approx]))
                aligned_triangle = align_triangle(np.array([pt[0] for pt in approx]), centroid)
                triangle_points = generate_triangle_points(aligned_triangle, num_points=100)
                continuous_points_count = count_continuous_points_within_threshold(triangle_points, approx, min_threshold)
                print(continuous_points_count,"trianlge")
                if continuous_points_count >= min_continuous_points_tri:
                    valid_triangles.append(aligned_triangle)
        elif len(approx) <= 10 and is_valid_rectangle(np.array([pt[0] for pt in approx])):
            rectangle_count += 1
            rect = cv2.minAreaRect(approx)
            rect_points = generate_rectangle_points(rect)
            continuous_points_count = count_continuous_points_within_threshold(rect_points, approx, min_threshold)
            if rectangle_count == 1:
                firstRect.append(rect)
            if continuous_points_count >= min_continuous_points_rect:
                valid_rectangles.append(rect)
    for triangle in valid_triangles:
        cv2.polylines(detected_contours, [triangle], True, (0, 255, 0), 2)
    if len(valid_rectangles) == 0 and len(firstRect) != 0:
        for rect in firstRect:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(detected_contours, [box], 0, (255, 0, 0), 2)
            return detected_contours
    for rect in valid_rectangles:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(detected_contours, [box], 0, (255, 0, 0), 2)
    return detected_contours

def detect_phase1(image_path):
    image = cv2.imread(image_path)
    min_continuous_points_rect = 160
    min_continuous_points_tri = 50
    detected_image = detect_triangles_rectangles_squares(image, min_continuous_points_rect, min_continuous_points_tri)
    detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    return detected_image_rgb
