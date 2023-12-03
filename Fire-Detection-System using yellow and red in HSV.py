import cv2
import numpy as np

def count_non_zero_bits(binary_image):
    return np.count_nonzero(binary_image)

def main():
 
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([20, 100, 100])
        upper_color = np.array([30, 255, 255])
        color_mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        filtered_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
        filtered_gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(filtered_gray, 1, 255, cv2.THRESH_BINARY)
        non_zero_bits = count_non_zero_bits(binary_image)
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Filtered Result', filtered_frame)
        cv2.imshow('Binary Image', binary_image)
        print("Non-zero bit count:", non_zero_bits)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()