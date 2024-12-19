import cv2
from cv2.typing import MatLike
import numpy as np
from standard_court import coordinates, dimensions, court_lines

'''
Steps:
- find the key corners on real img
- find homography matrix
- draw court lines on real img
'''

def get_key_corners(frame):
    key_corners = []
    areas = get_red_circle_areas(frame)
    if areas is not None:
        for i, a in enumerate(areas):
            corners = detect_corners(a.area)
            x, y = corners[0].ravel()
            x, y = x + a.start[0], y + a.start[1]
            key_corners.append((x, y))
    return key_corners


class ExtractedArea:
    area: MatLike
    start: tuple[int, int]
    end: tuple[int, int]

    def __init__(self, area, start, end) -> None:
        self.area = area
        self.start = start
        self.end = end


def get_red_circle_areas(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, binary_img = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    
    # Step 2: Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Step 3: Create masks for red color
    # Red color can appear in two ranges in HSV
    lower_red1 = np.array([0, 100, 100])  # Adjust values as needed
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Combine both ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Step 3: Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        red_mask, 
        cv2.HOUGH_GRADIENT, 
        dp=2, 
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10, 
        maxRadius=30
    )

    # If circles are detected, process them
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_list = []

        # Store circle information (x, y, radius)
        for i in circles[0, :]: # type: ignore
            x, y, r = i
            circle_list.append((x, y, r))

        # Sort the circles: top-to-bottom, then left-to-right
        def sort_circles(circle_list):
            # Define a row threshold (e.g., based on the circle radius or height difference)
            row_threshold = 40  # Adjust based on your circle size and image scaling

            # Sort circles by y-coordinate, then group into rows
            sorted_by_y = sorted(circle_list, key=lambda c: c[1])  # Sort by y-coordinate
            rows = []
            current_row = []

            for circle in sorted_by_y:
                if not current_row:
                    current_row.append(circle)
                else:
                    # Compare y-coordinates to group circles into the same row
                    if abs(circle[1] - current_row[-1][1]) < row_threshold:
                        current_row.append(circle)
                    else:
                        # Add the completed row and start a new one
                        rows.append(current_row)
                        current_row = [circle]

            # Add the last row
            if current_row:
                rows.append(current_row)

            # Sort each row by x-coordinate (left-to-right)
            for row in rows:
                row.sort(key=lambda c: c[0])

            # Flatten rows back into a sorted list
            sorted_circles = [circle for row in rows for circle in row]
            return sorted_circles

        circle_list = sort_circles(circle_list)

        # Step 5: Extract ROIs
        extracted_areas: list[ExtractedArea] = []
        border = 8
        for x, y, r in circle_list:
            # Define bounding box around each circle
            x1, y1 = max(0, x - r + border), max(0, y - r + border)
            x2, y2 = min(frame.shape[1], x + r - (border * 2)), min(frame.shape[0], y + r - (border * 2))

            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            area = ExtractedArea(roi, (x1,y1), (x2,y2))
            extracted_areas.append(area)

            # Optional: Draw circles for visualization
            # cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the processed frame
        # cv2.imshow('Detected Circles', frame)

        return extracted_areas



def detect_corners(image) -> MatLike:
    # Preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    # cv2.imshow(f"binary-{rand()}", binary_img)
    corners = cv2.goodFeaturesToTrack(binary_img, maxCorners=1, qualityLevel=0.4, minDistance=10)
    corners = corners.astype(np.int64)
    return corners


def debug_corners(frame: MatLike, corners: list[tuple[int,int]]):
    for i, c in enumerate(corners):
        x, y = c
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.putText(frame, f"{i+1}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)  # Draw a filled green circle
    cv2.imshow("debug corners", frame)


def find_homography(corners, scale=100):
    # Detected points in the original image
    original_points = np.array([
        corners[0], # service box bottom left
        corners[1], # service box bottom middle
        corners[2], # service box bottom right
        corners[3], # baseline bottom left
        # corners[4], # baseline middle
        corners[5], # baseline bottom right
    ], dtype=np.float32)


    standard_points = np.array(flip_y_axis([
        coordinates["service_boxes"]["left"][0],
        coordinates["service_boxes"]["left"][1],
        coordinates["service_boxes"]["right"][1],
        coordinates["singles_inner"][0],
        coordinates["singles_inner"][1],
    ]), dtype=np.float32)

    # Scaling factor for the bird's-eye view (optional: convert meters to pixels)
    # scale = 100  # Example: 1 meter = 100 pixels
    standard_points *= scale

    # Compute the homography matrix
    H, _ = cv2.findHomography(standard_points, original_points)
    print("H", H)
    return H


def normalize_coordinates(point, offset):
    """
    Normalize the transformed coordinates by adding the offset of the cropped area.
    
    Args:
        point (np.ndarray): The transformed point as [x, y].
        offset (tuple): The (x, y) offset of the cropped area within the original image.
        
    Returns:
        tuple: Normalized point as (x, y) integer values.
    """
    x, y = point[:2]
    return int(x + offset[0]), int(y + offset[1])


def draw_court_lines(img, homography, court_lines, offset, scale):
    """
    Draw tennis court lines on the image using the homography matrix.
    """
    for i, line in enumerate(court_lines):
        # Scale and transform start and end points
        start_point = np.array([line[0][0] * scale, line[0][1] * scale, 1], dtype=np.float32)
        end_point = np.array([line[1][0] * scale, line[1][1] * scale, 1], dtype=np.float32)

        transformed_start = np.dot(homography, start_point)
        transformed_start /= transformed_start[2]

        transformed_end = np.dot(homography, end_point)
        transformed_end /= transformed_end[2]

        # Debugging: Print transformed points
        print(f"Line {i + 1}:")
        print("  Start (scaled):", start_point)
        print("  End (scaled):", end_point)
        print("  Transformed Start:", transformed_start)
        print("  Transformed End:", transformed_end)

        # Normalize coordinates to the original image
        normalized_start = normalize_coordinates(transformed_start, offset)
        normalized_end = normalize_coordinates(transformed_end, offset)

        # Debugging: Check normalized coordinates
        print("  Normalized Start:", normalized_start)
        print("  Normalized End:", normalized_end)

        # Draw the line
        cv2.line(img, normalized_start, normalized_end, (0, 255, 0), 2)


def real_to_image_coords(real_coords, image_width=dimensions["width"]*100, image_height=dimensions["height"]*100):
    # Court dimensions in meters
    court_width = dimensions["width"]
    court_height = dimensions["height"]

    # Scaling factors
    scale_x = image_width / court_width
    scale_y = image_height / court_height

    # Convert real-world coordinates to image coordinates
    image_coords = []
    for x, y in real_coords:
        x_img = x * scale_x
        y_img = image_height - (y * scale_y)  # Flip y-axis
        image_coords.append((int(x_img), int(y_img)))

    return image_coords


def flip_y_axis(real_world_coords, court_height=23.77):
    """
    Adjust real-world coordinates to match image coordinate system by flipping the Y-axis.

    Args:
        real_world_coords (list of tuple): List of (x, y) real-world coordinates in meters.
        court_height (float): Height of the tennis court in meters (default: 23.77).

    Returns:
        list of tuple: Adjusted coordinates with Y-axis flipped.
    """
    return [(x, court_height - y) for x, y in real_world_coords]


def main():
    frame_path = "./data/tennis-court-circles.png"
    frame = cv2.imread(frame_path)

    corners = get_key_corners(frame)
    debug_corners(frame, corners)

    scale = 100

    H = find_homography(corners, scale)
    draw_court_lines(frame, H, court_lines, (0,0), scale)

    cv2.imshow("image", frame)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
