import matplotlib.pylab as plt
import cv2
import numpy as np

img=cv2.imread('1.jpg')
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)
ht=img.shape[0]
wd=img.shape[1]

region= [
    (0,ht),
    (wd/2, ht/2),
    (wd,ht)
]
def region_interest(img,vertice):
    mask=np.zeros_like(img)
   # channel_count=img.shape[2]
    match_mask_c=255
    cv2.fillPoly(mask,vertice,match_mask_c)
    masked_image= cv2.bitwise_and(img, mask)
    return masked_image

def draw_li(img, li):
    img=np.copy(img)
    line_img=np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)

    for line in li:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1,y1),(x2,y2),(255,0,0),thickness=20)

    img= cv2.addWeighted(img,0.8,line_img,1,0.0)
    return img

gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
can_img=cv2.Canny(gray_img,100,200)
crop_img= region_interest(can_img,
                        np.array([region], np.int32))

li=cv2.HoughLinesP(crop_img,
                   rho=6,
                   theta=np.pi/60,
                   threshold=160,lines=np.array([]),
                   minLineLength=40,
                   maxLineGap=25)

image_lines=draw_li(img,li)
plt.imshow(image_lines)
plt.show()

#img=cv2.imread('1.jpg')
#img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process(img):
    print(img.shape)
    ht = img.shape[0]
    wd = img.shape[1]

    region = [
        (0, ht),
        (wd / 2, ht / 2),
        (wd, ht)
    ]

    def region_interest(img, vertice):
        mask = np.zeros_like(img)
        # channel_count=img.shape[2]
        match_mask_c = 255
        cv2.fillPoly(mask, vertice, match_mask_c)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_li(img, li):
        img = np.copy(img)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for line in li:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

        img = cv2.addWeighted(img, 0.8, line_img, 1, 0.0)
        return img

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    can_img = cv2.Canny(gray_img, 100, 200)
    crop_img = region_interest(can_img,
                               np.array([region], np.int32))

    li = cv2.HoughLinesP(crop_img,
                         rho=6,
                         theta=np.pi / 60,
                         threshold=160, lines=np.array([]),
                         minLineLength=40,
                         maxLineGap=25)

    image_lines = draw_li(img, li)
    return image_lines

cap=cv2.VideoCapture('bi.mp4')

while(cap.isOpened()):
    ret, frame =cap.read()
    frame= process(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
