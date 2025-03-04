import cv2

label_dir = 'data/animals_data/labels/dog_1.txt'
img_dir = 'data/animals_data/JPEGImages/dog_1.jpg'

img = cv2.imread(img_dir)
img_w,img_h = img.shape[1],img.shape[0]

idmap = ['cat','dog']

id,cx,cy,aw,ah = 0,0,0,0,0

anchor_info = []

with open(label_dir,'r',encoding = 'utf-8') as file:
    for line in file:
        anchor_info = line.split()
        id = idmap[int(anchor_info[0])]
        cx = int(float(anchor_info[1]) * img_w)
        cy = int(float(anchor_info[2]) * img_h)
        aw = int(float(anchor_info[3]) * img_w)
        ah = int(float(anchor_info[4]) * img_h)

x1 = cx - aw // 2
x2 = cx + aw // 2
y1 = cy - ah // 2
y2 = cy + ah // 2

print(x1,x2,y1,y2)
print(img_w,img_h)

img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),thickness = 1)
cv2.imshow("Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()