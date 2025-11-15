import cv2
import numpy as np

path = "lut_orange2_generated.npy"   
kernel_size = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

lut = np.load(path)
last_center = None
cap = cv2.VideoCapture(2) #"Resources/video2.mp4"

while True:
    ret, frame = cap.read()
    if not ret:
        print("break")
        break

    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Bajar brillo
    v = np.clip(v - 40, 0, 255)

    hsv_darker = cv2.merge([h, s, v])
    frame = cv2.cvtColor(hsv_darker, cv2.COLOR_HSV2BGR)
    

    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    U = yuv[:, :, 1]
    V = yuv[:, :, 2]

    mask = lut[V, U]

    # Filtrado
    # quita puntitos de ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # rellena agujeros pequeños dentro de la pelota
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # suavizado pequeño para bordes más lisos
    mask = cv2.medianBlur(mask, 5)

    # encontrar la pelota
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        print(len(contours))

    
    possible_ellipses = []

    for cnt in contours:
        if len(cnt) >= 4: 
            area = cv2.contourArea(cnt)
            print(area)
            if area > 400 and area < 3000:  # se ajusta dependiendo del tamaño esperado
                ellipse = cv2.fitEllipse(cnt)
                possible_ellipses.append(ellipse)
                

    chosen_ellipse = None

    if possible_ellipses:
        if last_center is None:
            chosen_ellipse = max(possible_ellipses, key=lambda e: np.pi * (e[1][0] / 2) * (e[1][1] / 2))
        else:
            def distance(e):
                center = e[0]
                return np.linalg.norm(np.array(center) - np.array(last_center))
            
            chosen_ellipse = min(possible_ellipses, key=distance)
        
    if chosen_ellipse is not None:
        cv2.ellipse(frame, chosen_ellipse, (0, 255, 0), 2)
        last_center = chosen_ellipse[0] 
        (x, y) = last_center
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        print("Ball center: " + str(last_center))
    else:
        last_center = None
        print("Ball not detected")


    # Visualización
    result = cv2.bitwise_and(frame, frame, mask=mask)

    scale = 0.4 
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    result = cv2.resize(result, None, fx=scale, fy=scale)

    cv2.imshow("Original", frame)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()