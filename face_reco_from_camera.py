# 摄像头实时人脸识别

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera

# Created at 2018-05-11
# Updated at 2018-10-29

import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import pandas as pd # 数据处理的库 Pandas
import time
import os

import redis
import pickle

class Redis:
    @staticmethod
    def connect():
        r = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)
        return r

    #将内存数据二进制通过序列号转为文本流，再存入redis
    @staticmethod
    def set_data(r,key,data,ex=None):
        r.set(key,pickle.dumps(data),ex)

    # 将文本流从redis中读取并反序列化，返回返回
    @staticmethod
    def get_data(r,key):
        data = r.get(key)
        if data is None:
            return None

        return pickle.loads(data)


# 人脸识别模型，提取 128D 的特征矢量
# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("static/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 计算两个向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("e_distance: ", dist)

    if dist > 0.4:
        return "diff"
    else:
        return "same"


# 处理存放所有人脸特征的 CSV
path_features_known_csv = "static/features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)

# 存储的特征人脸个数
# print(csv_rd.shape[0])

# 用来存放所有录入人脸特征的数组
features_known_arr = []
features_known_name = []

# 读取已知人脸数据
# known faces
for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.loc[i, :])):
    # for j in range(0, len(csv_rd.ix[i, :])):
    #     print(csv_rd.loc[i, :][j])
        features_someone_arr.append(csv_rd.loc[i, :][j])
        # features_someone_arr.append(csv_rd.ix[i, :][j])
    #    print(features_someone_arr)
    name = features_someone_arr.pop()
    features_known_name.append(name)
    features_known_arr.append(features_someone_arr)
print("Faces in Database：", len(features_known_arr))

# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('static/data_dlib/shape_predictor_68_face_landmarks.dat')

# 创建 cv2 摄像头对象
cap = cv2.VideoCapture(1)
# cap.open("rtsp://admin:Aa123456@192.180.0.180/Streaming/Channels/103")

# cap.set(propId, value)
# 设置视频参数，propId 设置的视频参数，value 设置的参数值
cap.set(3, 100)

# 返回一张图像多张人脸的 128D 特征
def get_128d_features(img_gray):
    faces = detector(img_gray, 1)
    if len(faces) != 0:
        face_des = []
        for i in range(len(faces)):
            shape = predictor(img_gray, faces[i])
            face_des.append(facerec.compute_face_descriptor(img_gray, shape))
    else:
        face_des = []
    return face_des


# cap.isOpened() 返回 true/false 检查初始化是否成功
while cap.isOpened():

    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    # print(img_gray)
    # 人脸数 faces
    faces = detector(img_gray, 0)

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)

    # 存储人脸名字和位置的两个 list
    # list 1 (faces): store the name of faces               Jack    unknown unknown Mary
    # list 2 (pos_namelist): store the positions of faces   12,1    1,21    1,13    31,1

    # 存储所有人脸的名字
    pos_namelist = []
    name_namelist = []
    features_known_arr2 = []

    other = os.listdir('static/data_faces_from_camera/other')
    others=[]
    for i in range(len(other)):
        if(other[i] != '.DS_Store'):
            others.append(other[i])
    now = int(round(time.time(), 2) * 1000)
    if len(others)>0:
        last = max(others)[:-4]
    else:
        last = 0
    code = 800
    # print(last)

    # print(int(last)+code)
    # 按下 q 键退出
    if kk == ord('q'):
        break
    else:
        # 检测到人脸
        if len(faces) != 0:
            # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(img_rd, faces[i])
                features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

            # 遍历捕获到的图像中所有的人脸
            for k in range(len(faces)):
                # 让人名跟随在矩形框的下方
                # 确定人名的位置坐标
                # 先默认所有人不认识，是 unknown
                name_namelist.append("unknown")

                # 每个捕获人脸的名字坐标
                pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                print(features_known_arr)
                # 对于某张人脸，遍历所有存储的人脸特征
                for i in range(len(features_known_arr)):
                    # features_known_arr2 = features_known_arr
                    print("with person_", str(i+1), "the ", end='')
                    # name = features_known_arr2[i].pop()

                    # print(features_known_arr2[i])

                    # 将某张人脸与存储的所有人脸数据进行比对
                    compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])

                    if compare == "same":  # 找到了相似脸
                        name_namelist[k] = features_known_name[i]
                        # name_namelist[k] = "person_" + str(i+1)
                    #else 不相似的脸 截图保存 等待后续操作
                    else:
                        print(now)
                        print(last)
                        #
                        if((now) > int(last)+code or (int(last) == 0)):
                            # 将人脸计数器清零
                            cnt_ss = 0
                            path_make_dir = "static/data_faces_from_camera/"
                            for kd, d in enumerate(faces):
                                # 计算矩形框大小
                                height = (d.bottom() - d.top())
                                width = (d.right() - d.left())
                                hh = int(height / 2)
                                ww = int(width / 2)
                                color_rectangle = (255, 255, 255)
                                if (d.right() + ww) > 640 or (d.bottom() + hh > 480) or (d.left() - ww < 0) or (
                                        d.top() - hh < 0):
                                    cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                                    save_flag = 1
                                    color_rectangle = (0, 0, 255)
                                else:
                                    save_flag = 1
                                    color_rectangle = (0, 255, 255)
                                # 根据人脸大小生成空的图像
                                im_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)
                                if save_flag:
                                    cnt_ss += 1
                                    # print(cnt_ss)
                                    if(height * 2<720):
                                        for ii in range(height * 2):
                                            if(width * 2<720):
                                                for jj in range(width * 2):
                                                    if(d.top() - hh + ii<720):
                                                      im_blank[ii][jj] = img_rd[d.top() - hh + ii][d.left() - ww + jj]
                                        cv2.imwrite(path_make_dir + "/other/" + str(now) + ".jpg", im_blank)
                                print("写入本地：", path_make_dir + "/other/" + str(now) + ".jpg")

                # 矩形框
                for kk, d in enumerate(faces):
                    # print(d.left(), d.top())
                    # print(d.right(), d.bottom())
                    # 绘制矩形框
                    # if(name_namelist[kk]!='unknown'):
                    #     cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                    # else:
                        cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 0, 255),
                                      2)
                    # cv2.rectangle(img_rd,
                    #               tuple([d.left() - ww, d.top() - hh]),
                    #               tuple([d.right() + ww, d.bottom() + hh]),
                    #               color_rectangle, 2)

            # 在人脸框下面写人脸名字
            for i in range(len(faces)):
                cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

        # 將識別出的人臉存入Redis
        # r = Redis.connect()
        # if(len(name_namelist)>0):
        #     Redis.set_data(r, 'name', name_namelist)
        print("Name list now:", name_namelist, "\n")

    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("camera", img_rd)
# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()
