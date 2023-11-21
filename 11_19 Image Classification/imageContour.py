    import cv2
    import numpy as np

    def apply_closing(image, kernel_size):
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        closing_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return closing_result
    low_H = 0
    low_S = 99
    low_V = 119
    high_H = 10
    high_S = 255
    high_V = 236
    # to actually visualize the effect of `CHAIN_APPROX_SIMPLE`, we need a proper image
    image_filepath = '/home/michaelangelo/Documents/Team19_image_classifier-master/2023Simgs/S2023_imgs/111.png'
    #data, label = load_data(images_filepath, labels_filepath)
    image1 = cv2.imread(image_filepath)
    #height, width, channels = image1.shape
    #x1,y1,x2,y2 = width/4, height/4, width-(width/4), height-(height/4)
    #image1 = image1[int(y1):int(y2), int(x1):int(x2)]
    blur = cv2.GaussianBlur(image1,(15,15),0)
    frame_HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV,(low_H, low_S, low_V),(high_H, high_S, high_V))
    frame_threshold = apply_closing(frame_threshold, 35)
    #ret, thresh1 = cv2.threshold(img_gray1, 100, 200, cv2.THRESH_BINARY)
    #ret, thresh1 = cv2.adaptiveThreshold(img_gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours2, hierarchy2 = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #filtered_contours = [contour for i, contour in enumerate(contours2) if hierarchy2[0,i,3]==-1]
    image_copy2 = image1.copy()
    cv2.drawContours(frame_threshold, contours2, -1, (100, 200, 0), 2, cv2.LINE_AA)
    cv2.imshow('SIMPLE Approximation contours', frame_threshold)
    cv2.waitKey(0)
    image_copy3 = image1.copy()
    for i, contour in enumerate(contours2): # loop over one contour area
    for j, contour_point in enumerate(contour): # loop over the points
        # draw a circle on the current contour coordinate
        cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 2, cv2.LINE_AA)
    # see the results
    print(contours2)
    #print(hierarchy2)
    cv2.imshow('CHAIN_APPROX_SIMPLE Point only', image_copy3)
    cv2.waitKey(0)
    cv2.imwrite('contour_point_simple.jpg', image_copy3)
    cv2.destroyAllWindows()