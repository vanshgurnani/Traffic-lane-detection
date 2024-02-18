        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)