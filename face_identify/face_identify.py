#    __author__ = '武广'
#    __date__ = '2017/5/5'
#    __Desc__ = 人脸检测小例子，圈出人脸
import cv2
def image_size_transform(ResizeImg, w2, h2):
    if h2 > 700:
        ResizeImg = cv2.resize(src=ResizeImg, dsize=(int(w2*(670/h2)), 670), interpolation=cv2.INTER_LINEAR)
    elif h2 < 400:
        ResizeImg = cv2.resize(src=ResizeImg, dsize=(int(w2*(500/h2)), 500), interpolation=cv2.INTER_LINEAR)
    else:
        pass
    return ResizeImg
# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier("E:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")

    # 读取图片
    image = cv2.imread("face10.jpg")
    image = image_size_transform(image, image.shape[1], image.shape[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # w = image.shape[1]
    # h = image.shape[0]
    # for xi in range(0, w):
    #     for xj in range(0,h):
    #         image[xj, xi, 0] = int(image[xj, xi, 0] * 1.05)
    #         image[xj, xi, 1] = int(image[xj, xi, 1] * 1.05)
    #         image[xj, xi, 2] = int(image[xj, xi, 1] * 1.05)
    # 探测图片中的人脸
    faces = face_cascade.detectMultiScale(
                                            gray,
                                            scaleFactor=1.14,
                                            minNeighbors=5,
                                            minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE
                                            )

    print("发现{0}个人脸!".format(len(faces)))

    for(x, y, w, h) in faces:
        # cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detected faces", image)
    #cv2.imwrite("classmates.jpg", image)
    cv2.waitKey()
    cv2.destroyAllWindows()