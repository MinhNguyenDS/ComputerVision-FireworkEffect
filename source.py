import cv2
import imageio
import mediapipe as mp
import numpy as np

# Khởi tạo các biến
i = 0
i_flare = 0
count_energy = 0
count_time = 0
phao_no = False
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=2)
mpDraw=mp.solutions.drawing_utils

# Đọc file effect
img_rocket = cv2.imread('material/rocket.png', -1)
lighter_gif = imageio.mimread("material/lighter.gif")
flare_gif = imageio.mimread("material/flare.gif")
vid = cv2.VideoCapture('material/firework.mp4')

# Tiền xử lý các effect
img_rocket = cv2.cvtColor(img_rocket, cv2.COLOR_BGRA2BGR)
nums_lighter = len(lighter_gif)
lighter_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in lighter_gif]
nums_flare = len(flare_gif)
flare_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in flare_gif]

# Hàm Cộng 2 ảnh 
def sum2img(img1, img2):
    img = np.int64(img1) + np.int64(img2)
    img[img>255] = 255
    img = np.uint8(img)
    return img

# Hàm lấy tọa độ điểm
def position_data(lmlist):
    global wrist, index_tip, midle_tip, ring_tip, pinky_tip, thumb_tip, index_mcp, midle_mcp, ring_mcp, pinky_mcp
    
    wrist = (lmlist[0][0], lmlist[0][1])
    thumb_tip = (lmlist[4][0], lmlist[4][1])

    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_tip = (lmlist[12][0], lmlist[12][1])
    ring_tip  = (lmlist[16][0], lmlist[16][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])

    index_mcp = (lmlist[5][0], lmlist[5][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    ring_mcp = (lmlist[13][0], lmlist[13][1])
    pinky_mcp  = (lmlist[17][0], lmlist[17][1])

# Hàm tính khoảng cách giữa hai điểm
def calculate_distance(p1,p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght

# Hàm ghép BG vào FG
def transparent(targetImg, x, y, size=None):
    # Resize ảnh
    if size is not None:
        targetImg = cv2.resize(targetImg, size)

    # Tạo ảnh buffer có tỉ lệ tương tự như ảnh gốc
    newFrame = img.copy()

    b, g, r = cv2.split(targetImg)
    overlay_color = cv2.merge((b, g, r))

    h, w, _ = overlay_color.shape

    # Cắt vùng của ảnh gốc có ảnh cần chèn 
    roi = newFrame[y:y + h, x:x + w]

    img2gray = cv2.cvtColor(overlay_color,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)

    # Hình BG, Hình FG (pháo, quet)
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask = cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color,mask = mask)
    
    # Kết hợp background and foreground
    newFrame[y:y + h, x:x + w] =  cv2.add(img1_bg, img2_fg)
    return newFrame

video=cv2.VideoCapture(0)
while True:
    # Khởi tạo các biến
    x_thumb_tip_quet = x_index_mcp_quet = x_pinky_mcp_phao = x_wrist_phao = 0
    y_thumb_tip_quet = y_index_mcp_quet = y_pinky_mcp_phao = y_wrist_phao = 0
    tip_index_dis_phao =  tip_midle_dis_phao = tip_ring_dis_phao = tip_pingky_dis_phao = 0
    x_phao = y_phao = palm_phao = 0
    diameter_phao = diameter_quet = 0
    count_hand = 0

    # Đọc ảnh và dùng mediapine detect tay
    ret, img=video.read()
    img=cv2.flip(img, 1)
    rgbimg=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(rgbimg)
      
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            count_hand += 1
            
            # Lấy điểm của ngón tay
            lmList=[]
            for id, lm in enumerate(hand.landmark):
                h,w,c=img.shape
                coorx, coory=int(lm.x*w), int(lm.y*h)
                lmList.append([coorx, coory])
                # # vẽ các điểm của ngón tay
                # cv2.circle(img, (coorx, coory), 6, (50,50,255), -1)
            
            # # Vẽ xương tay
            # mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

            # Lấy tọa độ các điểm
            position_data(lmList)

            # Tính khoảng cách các điểm 
            tip_index_dis = calculate_distance(wrist, index_tip)
            tip_midle_dis = calculate_distance(wrist, midle_tip)
            tip_ring_dis = calculate_distance(wrist, ring_tip)
            tip_pingky_dis = calculate_distance(wrist, pinky_tip)
            
            mcp_index_dis = calculate_distance(wrist, index_mcp)
            mcp_midle_dis = calculate_distance(wrist, midle_mcp)
            mcp_ring_dis = calculate_distance(wrist, ring_mcp)
            mcp_pingky_dis = calculate_distance(wrist, pinky_mcp)

            # Tính tỉ lệ 
            dis_4 = ((mcp_index_dis + mcp_midle_dis + mcp_ring_dis + mcp_pingky_dis) / 4)
            tip_thumb_index_dis = calculate_distance(index_mcp, thumb_tip)
            ratio_dis = tip_thumb_index_dis / dis_4

            tip_index_dis_phao =  tip_index_dis
            tip_midle_dis_phao = tip_midle_dis
            tip_ring_dis_phao = tip_ring_dis
            tip_pingky_dis_phao = tip_pingky_dis

            # Các giá trị and giải quyết các ngon tay không gấp vào
            if (tip_index_dis < 150 and tip_midle_dis < 150 and tip_ring_dis < 150 and tip_pingky_dis < 150):
                # Quẹt
                if (ratio_dis >= 0.75):
                    x_index_mcp_quet = index_mcp[0]
                    y_index_mcp_quet = index_mcp[1]
                    x_thumb_tip_quet = thumb_tip[0]
                    y_thumb_tip_quet = thumb_tip[1]
                    
                    palm = (mcp_index_dis + mcp_midle_dis + mcp_ring_dis + mcp_pingky_dis) / 4
                    
                    centerx = ((wrist[0] + index_mcp[0])/2) * 1
                    centery = ((wrist[1] + index_mcp[1])/5) * 2
                    shield_size = 3.2

                    # Tính đường kính của cái hình FG
                    diameter = round(palm * shield_size)

                    # Tính tọa độ điểm thêm hình
                    x = round(centerx - (diameter / 2))
                    y = round(centery - (diameter / 2))

                    # Xử lý biên
                    h, w, c = img.shape
                    if x < 0:
                        x = 0
                    elif x > w:
                        x = w
                    if y < 0:
                        y = 0
                    elif y > h:
                        y = h
                    # Nếu ra hai biên phải và dưới thì update lại đường kính 
                    if x + diameter > w:
                        diameter = w - x
                    if y + diameter > h:
                        diameter = h - y
                    
                    diameter_quet = diameter

                    shield_size = diameter, diameter

                    if (diameter != 0):
                        img = transparent(lighter_imgs[i], x, y, shield_size)
                    i = (i+1)%nums_lighter

                # Pháo
                elif (ratio_dis < 0.5): 
                    x_pinky_mcp_phao = pinky_mcp[0]
                    y_pinky_mcp_phao = pinky_mcp[1]
                    x_wrist_phao = wrist[0]
                    y_wrist_phao = wrist[1]

                    palm = (mcp_index_dis + mcp_midle_dis + mcp_ring_dis + mcp_pingky_dis) / 4
                    palm_phao = palm
                    centerx = (midle_mcp[0] + wrist[0]) /2
                    centery = (midle_mcp[1] + wrist[1]) /2
                    shield_size = 3.2

                    # Tính đường kính của cái hình FG
                    diameter = round(palm * shield_size)

                    # Tính tọa độ điểm thêm hình
                    x = round(centerx - (diameter / 2))
                    y = round(centery - (diameter / 2))

                    # Xử lý biên
                    h, w, c = img.shape
                    if x < 0:
                        x = 0
                    elif x > w:
                        x = w
                    if y < 0:
                        y = 0
                    elif y > h:
                        y = h
                    # Nếu ra hai biên phải và dưới thì update lại đường kính 
                    if x + diameter > w:
                        diameter = w - x
                    if y + diameter > h:
                        diameter = h - y

                    shield_size = diameter, diameter
                    hei, wid, col = img_rocket.shape
                    cen = (wid // 2, hei // 2)

                    diameter_phao = diameter

                    x_phao = x
                    y_phao = y

                    # Xoay hình
                    M1 = cv2.getRotationMatrix2D(cen, 50, 1.0)
                    img_rocket_rotate = cv2.warpAffine(img_rocket, M1, (wid, hei))
                    if (diameter != 0):
                        img = transparent(img_rocket_rotate, x, y, shield_size)
            
            # Xử lý tích tụ năng lượng
            # Điều kiện thõa 2 tay và xuất hiện cả pháo lẫn quẹt
            if count_hand == 2 and x_thumb_tip_quet != 0 and x_index_mcp_quet != 0 and  x_pinky_mcp_phao != 0 and x_wrist_phao != 0:
                # Điều kiện pháo nằm trên quẹt
                if y_thumb_tip_quet > y_pinky_mcp_phao and y_index_mcp_quet > y_wrist_phao :
                    if ((x_thumb_tip_quet + x_index_mcp_quet)/2 > x_pinky_mcp_phao and (x_thumb_tip_quet + x_index_mcp_quet)/2 < x_wrist_phao)\
                        or ((x_thumb_tip_quet + x_index_mcp_quet)/2 < x_pinky_mcp_phao and (x_thumb_tip_quet + x_index_mcp_quet)/2 > x_wrist_phao):
                        # Xử lý quẹt để sát pháo 
                        X = diameter_phao + diameter_quet
                        Y = abs(y_thumb_tip_quet - y_pinky_mcp_phao) + abs(y_index_mcp_quet - y_wrist_phao)
                        if (0.25 < Y/X < 0.85):
                            count_energy += 2

                            # Vẽ flare    
                            thumb_tip_quet = (x_thumb_tip_quet, y_thumb_tip_quet)
                            index_mcp_quet = (x_index_mcp_quet, y_index_mcp_quet)
                            pinky_mcp_phao = (x_pinky_mcp_phao, y_pinky_mcp_phao)
                            wrist_phao = (x_wrist_phao, x_wrist_phao)

                            wrist_thumb_dis_flare = calculate_distance(wrist_phao, thumb_tip_quet)
                            pinky_index_dis_flare = calculate_distance(pinky_mcp_phao, index_mcp_quet)
                            palm = (wrist_thumb_dis_flare + pinky_index_dis_flare)/2
                            
                            centerx = ((x_wrist_phao + x_thumb_tip_quet + x_pinky_mcp_phao + x_index_mcp_quet)/4) * 1
                            centery = ((y_wrist_phao + y_thumb_tip_quet + y_pinky_mcp_phao + y_index_mcp_quet)/4) * 1
                            shield_size = 2

                            # Tính đường kính của cái hình FG
                            diameter = round(palm * shield_size)

                            # Tính tọa độ điểm thêm hình
                            x = round(centerx - (diameter / 2))
                            y = round(centery - (diameter / 2))

                            # Xử lý biên
                            h, w, c = img.shape
                            if x < 0:
                                x = 0
                            elif x > w:
                                x = w
                            if y < 0:
                                y = 0
                            elif y > h:
                                y = h
                            # Nếu ra hai biên phải và dưới thì update lại đường kính 
                            if x + diameter > w:
                                diameter = w - x
                            if y + diameter > h:
                                diameter = h - y
                            
                            shield_size = diameter, diameter

                            if (diameter != 0):
                                img = transparent(flare_imgs[i_flare], x, y, shield_size)
                            i_flare = (i_flare+1)%nums_flare

                    elif count_energy > 2 : count_energy -= 1
                    
            elif count_energy > 2 : count_energy -= 1
            print('Energy (full:40):',count_energy)
            if count_energy >= 40 : phao_no = True
        
    # Xử lý pháo nổ
    if (phao_no == True) & (count_time < 40) & (tip_index_dis_phao > 150):
        count_energy = 0

        count_time += 1
        h,w,c = img.shape
        ret_, firework = vid.read()
        firework = cv2.resize(firework, (w,h))
        firework = cv2.cvtColor(firework, cv2.COLOR_BGR2RGB)
        img = sum2img(img, firework)
                
    if count_time == 38:
        count_time = 0
        phao_no = False
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.imshow("Fire work effect", img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()