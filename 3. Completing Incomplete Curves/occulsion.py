import cv2
import numpy as np
import svgwrite
import matplotlib.pyplot as plt
from skimage.measure import approximate_polygon, find_contours
from polygon import detect_phase1

def imgae_to_polyline(image_path):
    img = cv2.imread(image_path, 0)
    contours = find_contours(img, 0)
    paths_XYs = []
    for contour in contours:
        polygon1 = approximate_polygon(contour, tolerance=2.5)
        polygon2 = approximate_polygon(contour, tolerance=15)
        paths_XYs.append([contour])
    result_contour = np.zeros(img.shape + (3, ), np.uint8)
    result_polygon1 = np.zeros(img.shape + (3, ), np.uint8)
    result_polygon2 = np.zeros(img.shape + (3, ), np.uint8)
    for contour in contours:
        polygon1 = approximate_polygon(contour, tolerance=2.5)
        polygon2 = approximate_polygon(contour, tolerance=15)
        contour = contour.astype(np.int64).tolist()
        polygon1 = polygon1.astype(np.int64).tolist()
        polygon2 = polygon2.astype(np.int64).tolist()
        for idx, coords in enumerate(contour[:-1]):
            y1, x1, y2, x2 = coords + contour[idx + 1]
            result_contour = cv2.line(result_contour, (x1, y1), (x2, y2),
                                    (0, 255, 0), 2)
        for idx, coords in enumerate(polygon1[:-1]):
            y1, x1, y2, x2 = coords + polygon1[idx + 1]
            result_polygon1 = cv2.line(result_polygon1, (x1, y1), (x2, y2),
                                    (0, 255, 0), 2)
        for idx, coords in enumerate(polygon2[:-1]):
            y1, x1, y2, x2 = coords + polygon2[idx + 1]
            result_polygon2 = cv2.line(result_polygon2, (x1, y1), (x2, y2),
                                    (0, 255, 0), 2)
    svg_path = 'output_image.svg'
    polylines2svg(paths_XYs, svg_path)
    cv2.imwrite('contour_lines.png', result_contour)
    cv2.imwrite('polygon1_lines.png', result_polygon1)
    cv2.imwrite('polygon2_lines.png', result_polygon2)

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            if XY.ndim == 2:
                W, H = max(W, np.max(XY[:, 1])), max(H, np.max(XY[:, 0]))
    padding = 0.2
    W, H = int(W + padding * W), int(H + padding * H)
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges', size=(W, H))
    group = dwg.g()
    color = "black"
    for path in paths_XYs:
        path_data = []
        for XY in path:
            path_data.append(f"M {XY[0, 1]},{XY[0, 0]}")
            for j in range(1, len(XY)):
                path_data.append(f"L {XY[j, 1]},{XY[j, 0]}")
            if not np.allclose(XY[0], XY[-1]):
                path_data.append("Z")
        group.add(dwg.path(d=" ".join(path_data), fill='none', stroke=color, stroke_width=2))
    dwg.add(group)
    dwg.save()

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
    return 1.1*threshold

def count_continuous_points_within_threshold(shape_points, contour, distance_threshold):
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
    """Calculate the area of an ellipse."""
    _, (MA, ma), _ = ellipse
    return np.pi * (MA / 2) * (ma / 2)

def detect_shapes(image_path, min_major_axis=70, max_major_axis=1000, min_aspect_ratio=0.2, max_aspect_ratio=2,
                  ellipse_percentage=0.47, circle_percentage=0.16, rectangle_percentage=0.3, min_continuous_points=5,
                  min_circle_radius=50, min_side_length=30, area_difference_threshold=0.2):
    image = cv2.imread(image_path)
    processed_image = detect_phase1(image_path)
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 50, 150)
    white_background = np.ones_like(gray) * 255
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=False)
    cv2.drawContours(white_background, contours, -1, (0, 0, 0), 3)
    output = cv2.cvtColor(white_background, cv2.COLOR_GRAY2BGR)
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
                print(MA, ma, aspect_ratio)
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
                cv2.ellipse(output, largest_ellipse, (0, 0, 0), 4)
                cx, cy = largest_ellipse[0]
                cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            else:
                for ellipse in [largest_ellipse, second_largest_ellipse]:
                    cv2.ellipse(output, ellipse, (0, 0, 0), 2)
                    cx, cy = ellipse[0]
                    cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        else:
            cv2.ellipse(output, largest_ellipse, (0, 0, 0), 2)
            cx, cy = largest_ellipse[0]
            cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    elif len(valid_ellipses) == 1:
        cv2.ellipse(output, valid_ellipses[0], (0, 0, 0), 2)
        cx, cy = valid_ellipses[0][0]
        cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    for circle in valid_circles:
        (cx, cy), radius = circle
        cv2.circle(output, (int(cx), int(cy)), int(radius), (0, 0, 0), 4)
        cv2.circle(output, (int(cx), int(cy)), 5, (255, 0, 0), -1)
    for rect in valid_rectangles:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output, [box], 0, (0, 0, 0), 4)
        cx, cy = rect[0]
        cv2.circle(output, (int(cx), int(cy)), 5, (255, 0, 0), -1)
    output_image_path = 'output_image_with_shapes.png'
    cv2.imwrite(output_image_path, output)
    fig, axs = plt.subplots(1, 2, figsize=(6, 6))
    axs[0].imshow(output)
    axs[0].set_title("Output Image")
    axs[0].axis("off")
    axs[1].imshow(image, cmap='gray')
    axs[1].set_title("Original Image")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()
    imgae_to_polyline(output_image_path)

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
    img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255
    for path in paths:
        for segment in path:
            for i in range(len(segment) - 1):
                cv2.line(img, tuple(segment[i].astype(int)), tuple(segment[i+1].astype(int)), (0, 0, 0), 2)
    return img

if __name__ == '__main__':
    custom_choice = input("Enter '1' to input a CSV file or '2' to input an image file: ").strip()
    if custom_choice == '1':
        csv_path = input("Enter the path of the CSV file: ").strip()
        try:
            path_XYs = read_csv(csv_path)
            image_size = (500, 500)
            image = image_from_paths(path_XYs, image_size=image_size)
            if image is not None:
                save_status = cv2.imwrite('shapes_image.png', image)
                if save_status:
                    print("Image saved successfully.")
                else:
                    print("Failed to save the image.")
                detect_shapes('shapes_image.png')
            else:
                print("Error: Failed to create image from paths.")
        except Exception as e:
            print(f"Error processing CSV file: {e}")
    elif custom_choice == '2':
        custom_image_path = input("Enter the path of the custom image: ").strip()
        detect_shapes(custom_image_path)
    else:
        print("Invalid choice. Please enter '1' or '2'.")
