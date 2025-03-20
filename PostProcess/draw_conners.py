import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the image and points list from param
        img, points = param
        if len(points) < 4:
            # Draw circle at clicked point
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            # Store coordinates
            points.append((x, y))
            # Print coordinates
            print(f"Point {len(points)}: ({x}, {y})")
            # Update display
            cv2.imshow('image', img)

def mark_points():
    # Load image
    img = cv2.imread("EyeTrackData/Calibration/Left/2025-01-07_16-50-40_526.png")
    if img is None:
        print("Error: Could not load image")
        return
    
    # Resize to 1000x1000
    img = cv2.resize(img, (1000, 1000))
    
    # Create a copy for drawing
    img_copy = img.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow('image')
    points = []
    cv2.setMouseCallback('image', mouse_callback, (img_copy, points))
    
    print("Click to mark 25 points. Press 'q' to quit.")
    
    while True:
        cv2.imshow('image', img_copy)
        
        # Break if 25 points marked or 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or len(points) >= 25:
            break
    
    cv2.destroyAllWindows()
    return points


def show_click_points():
    # Load image
    img = cv2.imread("D:\EyeTrack\PostProcess\Blink\open_man.jpg")
    if img is None:
        print("Error: Could not load image")
        return
    
    # Create window
    cv2.namedWindow('image')
    
    # Mouse callback function
    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked coordinates: ({x}, {y})")
            # Draw circle at clicked point
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow('image', img)
    
    # Set mouse callback
    cv2.setMouseCallback('image', click_callback)
    
    # Show image and wait for clicks
    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()



if __name__ == "__main__":
    show_click_points()