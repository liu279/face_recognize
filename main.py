from flask import Flask, render_template_string,render_template, Response, jsonify,request
import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCv
import os           # 读写文件
import shutil       # 读写文件
import sys
import json
# app = Flask(__name__)
import redis
import pickle
import leancloud

from skimage import io
import csv
import pandas as pd
import base64
import time
from numba import jit
# from mpi4py import MPI


leancloud.init("rMFNrioxVUtWdFdcVSlPSX6T-gzGzoHsz", "0fbQtBDhRgptE0DLhpLruQyr")
app = Flask(__name__, static_folder='static/data_faces_from_camera/other')


class Redis:
    @staticmethod
    def connect():
        r = redis.StrictRedis(host='localhost', port=6379, db=0)
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


class VideoCamera(object):

    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        # Dlib 正向人脸检测器
        detector = dlib.get_frontal_face_detector()
        # Dlib 68 点特征预测器
        predictor = dlib.shape_predictor('static/data_dlib/shape_predictor_68_face_landmarks.dat')

        # 设置视频参数
        self.video.set(3, 480)

        # while self.video.isOpened():
        # 480 height * 640 width
        flag, img_rd = self.video.read()

        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
        # 人脸数 faces
        faces = detector(img_gray, 0)
        # print(faces)
        # 矩形框
        # for k, d in enumerate(faces):
            # print(d)
            # print(d.left())
            # print(d.top())
            # print(d.right())
            # print(d.bottom())
        #     # 计算矩形大小
        #     # (x,y), (宽度width, 高度height)
        #     pos_start = tuple([d.left(), d.top()])
        #     pos_end = tuple([d.right(), d.bottom()])
        #
        #     # 计算矩形框大小
        #     height = (d.bottom() - d.top())
        #     width = (d.right() - d.left())
        #
        #     hh = int(height / 2)
        #     ww = int(width / 2)
        #
        #     # 设置颜色 / The color of rectangle of faces detected
        #     color_rectangle = (255, 255, 255)
        #     # if (d.right() + ww) > 640 or (d.bottom() + hh > 480) or (d.left() - ww < 0) or (d.top() - hh < 0):
        #     #     cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        #     #     color_rectangle = (0, 0, 255)
        #     # else:
        #     #     color_rectangle = (255, 255, 255)
        #
        #     cv2.rectangle(img_rd,
        #                   tuple([d.left() - ww, d.top() - hh]),
        #                   tuple([d.right() + ww, d.bottom() + hh]),
        #                   color_rectangle, 2)
        #     # print(cv2)


        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return {'img': jpeg.tobytes(), 'faces': faces, }
        # return {'img':jpeg.tobytes(),'left':d.left() - ww, 'top':d.top() - hh,'right':d.right() + ww, 'bottom':d.bottom() + hh }


@app.route('/')  # 主页
def index():
    # frame = VideoCamera().get_frame()
    # if(len(frame['faces'])):
    #     top = frame['faces'][0].top()
    # else:
    #     top = 0
    #
    # data = {'top': top}
    # jinja2模板，具体格式保存在index.html文件中
    # dir = './static/data_faces_from_camera/other/'
    # arr=[]
    # for file in os.listdir(dir):
    #     if(file != '.DS_Store'):
    #         arr.append(file)
    # print(arr)
    # arrs={
    #     "data":arr, data=json.dumps(arr)
    # }

    return render_template('index.html')


@app.route('/line', methods=['get'])  # 这个地址返回未识别到的
def line():
    dir = './static/data_faces_from_camera/other/'
    arr=[]
    for file in os.listdir(dir):
        if(file != '.DS_Store'):
            arr.append(file)
    # print(arr)
    arr.sort(reverse=True)
    return json.dumps(arr) #jsonify()


@app.route('/line2', methods=['get'])  # 这个地址返回识别到的
def line2():
    r = Redis.connect()
    goods = Redis.get_data(r, 'name')
    das=[]
    Todo = leancloud.Object.extend('Test')
    query = Todo.query
    for i in goods:
        # print(i)
        query.equal_to('image_id', i)
        query_list = query.find()
        for ii in query_list:
            # print(ii.get('image_id'))
            da = {
                'image_id': ii.get('image_id'),
                'name': ii.get('name'),
                'info': ii.get('info'),
                'img': ii.get('img'),
            }
            das.append(da)

    # print(goods)
    # print(das)
    return json.dumps(das) #jsonify()


@app.route('/line3', methods=['post'])  # 这个地址是提交个人资料
def line3():
    path_faces_rd = "static/data_faces_from_camera/other"
    path_csv = "static/data_csvs_from_camera/"
    # return jsonify(image_list)
    if request.method == 'POST':
        image_list = request.values.get('imgs').strip(',').split(',')
        dirname = request.values.get('dirname')
        name = request.values.get('name')
        info = request.values.get('info')
        imgs = image_list
        # 写入单人的csv
        with open(path_csv + dirname + ".csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(imgs)):
                # 调用return_128d_features()得到128d特征
                print("正在读的人脸图像：", path_faces_rd + "/" + imgs[i])
                features_128d = return_128d_features(path_faces_rd + "/" + imgs[i])
                #  print(features_128d)
                # 遇到没有检测出人脸的图片跳过
                if features_128d == 0:
                    i += 1
                else:
                    writer.writerow(features_128d)

        # 存放所有特征均值的 CSV 的路径
        path_csv_feature_all = "static/features_all.csv"

        # 存放人脸特征的 CSV 的路径
        path_csv_rd = "static/data_csvs_from_camera/"

        # 写入全部的csv
        with open(path_csv_feature_all, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            csv_rd = os.listdir(path_csv_rd)
            print("##### 得到的特征均值 / The generated average values of features stored in: #####")

            for i in range(len(csv_rd)):
                feature_mean = compute_the_mean(path_csv_rd + csv_rd[i])

                feature_mean.append(csv_rd[i][:-4])
                # print(csv_rd[i][:-4])
                print(feature_mean)
                print(path_csv_rd + csv_rd[i])
                writer.writerow(feature_mean)

        Todo = leancloud.Object.extend('Test')
        todo = Todo()
        todo.set('image_id',dirname)
        todo.set('name',name)
        todo.set('info',info)
        todo.set('img',image_list[0])
        try:
            todo.save()
            return jsonify({'code':200, 'msg':'成功'})
        except leancloud.LeanCloudError as e:
            return jsonify({'code': 201, 'msg': '失败'})

@app.route('/line4', methods=['post'])  # 这个地址是提交个人资料
def line4():
    path_faces_rd = "static/data_faces_from_camera/other"
    path_csv = "static/data_csvs_from_camera/"
    if request.method == 'POST':
        image_list = request.values.get('imgs').strip(',').split(',')
        name = request.values.get('name')
        image_id = request.values.get('image_id')
        sex = request.values.get('sex')
        age = request.values.get('age')
        rank = request.values.get('rank')
        department = request.values.get('department')
        slogan = request.values.get('slogan')
        workon_t = request.values.get('workon_t')
        workout_t = request.values.get('workout_t')
        avatar = image_list[0]

        f = open('/Users/cityfruit-lf/Desktop/face_recognition-master2/api',encoding='UTF-8')
        user_info = json.loads(f.read())
        # f.close()

        ii=[]
        for i in range(len(user_info['result'])):
            ii.append(user_info['result'][i]['id'])
        id = max(ii)

        data = {
            'id': id+1,
            'uid': '',
            'avatar': '/Users/cityfruit-lf/Desktop/flask/static/data_faces_from_camera/other/'+avatar,
            'name': name,
            'sex': sex,
            'age': age,
            'rank': rank,
            'department': department,
            'workon_t': workon_t,
            'workout_t': workout_t,
            "work_night": "白班",
            "slogan": slogan,
            "add_time": 1531468860,
            "update_time": 1534215330,
            "coid": 0,
            "avatar_name": '/Users/cityfruit-lf/Desktop/flask/static/data_faces_from_camera/other/'+avatar
        }
        user_info['result'].append(data)
        print(user_info)
        with open('/Users/cityfruit-lf/Desktop/face_recognition-master2/api', 'w') as f:
            json.dump(user_info, f)
        # f.write(user_info)

        # 写入单人的csv
        with open(path_csv + image_id + ".csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(image_list)):
                # 调用return_128d_features()得到128d特征
                print("正在读的人脸图像：", path_faces_rd + "/" + image_list[i])
                features_128d = return_128d_features(path_faces_rd + "/" + image_list[i])
                #  print(features_128d)
                # 遇到没有检测出人脸的图片跳过
                if features_128d == 0:
                    i += 1
                else:
                    writer.writerow(features_128d)

        # 存放所有特征均值的 CSV 的路径
        path_csv_feature_all = "static/features_all.csv"

        # 存放人脸特征的 CSV 的路径
        path_csv_rd = "static/data_csvs_from_camera/"

        # 写入全部的csv
        with open(path_csv_feature_all, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            csv_rd = os.listdir(path_csv_rd)
            print("##### 得到的特征均值 / The generated average values of features stored in: #####")

            for i in range(len(csv_rd)):
                feature_mean = compute_the_mean(path_csv_rd + csv_rd[i])

                feature_mean.append(csv_rd[i][:-4])
                # print(csv_rd[i][:-4])
                print(feature_mean)
                print(path_csv_rd + csv_rd[i])
                writer.writerow(feature_mean)

        Todo = leancloud.Object.extend('Test')
        todo = Todo()
        todo.set('image_id', image_id)
        todo.set('name', name)
        todo.set('info', slogan)
        todo.set('img', image_list[0])
        # return jsonify({'code': 200, 'msg': '成功'})

        try:
            todo.save()
            return jsonify({'code':200, 'msg':'成功'})
        except leancloud.LeanCloudError as e:
            return jsonify({'code': 201, 'msg': '失败'})





@app.route('/line5', methods=['post'])  # 这个地址返回识别到的
def line5():
    # filepath = "static/data_faces_from_camera/other/1544510366240.jpg"
    # img = cv2.imread(filepath)  # 读取图片
    # print(img)
    if request.method == 'POST':
        file = request.values.get('url').split(',')[1]
        imD = base64.b64decode(file)
        nparr = np.fromstring(imD, np.uint8)
        # print(nparr)
        # return jsonify({'code': 201, 'msg': file})

        # cv2.IMREAD_COLOR 以彩色模式读入 1
        # cv2.IMREAD_GRAYSCALE 以灰色模式读入 0
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色

        # OpenCV人脸识别分类器
        classifier = cv2.CascadeClassifier(
            "static/haarcascades/haarcascade_frontalface_default.xml"
        )
        color = (0, 255, 0)  # 定义绘制颜色
        # 调用识别人脸
        faceRects = classifier.detectMultiScale(
            image, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        data=[]
        if len(faceRects):  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                print(faceRect)
                print(x)
                print(y)
                print(x + h)
                print(y + w)
                da = {
                    'xx':x,
                    'yy':y,
                    'xh':x + h,
                    'yw':y + w,
                }
                data.append(da)

            print(data)

        return json.dumps(data,cls=MyEncoder) #jsonify()






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



@jit
@app.route('/line6', methods=['post'])  # 这个地址返回识别到的
def line6():
    start = time.time()
    # 人脸识别模型，提取 128D 的特征矢量
    # face recognition model, the object maps human faces into 128D vectors
    facerec = dlib.face_recognition_model_v1("static/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

    # Dlib 检测器和预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('static/data_dlib/shape_predictor_68_face_landmarks.dat')
    # print(predictor)
    # print(time.strftime("%H:%M:%S", time.localtime()))
    if request.method == 'POST':
        try:

            file = request.values.get('url').split(',')[1]
            imD = base64.b64decode(file)
            nparr = np.frombuffer(imD, np.uint8)

            # image = cv2.imread('static/data_faces_from_camera/other/1544510366240.jpg')
            # res = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
            # cv2.IMREAD_COLOR 以彩色模式读入 1
            # cv2.IMREAD_GRAYSCALE 以灰色模式读入 0
            # img_gray = cv2.cvtColor(nparr, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rd = img_gray
            # 人脸数 faces
            faces = detector(img_gray, 0)


            # # OpenCV人脸识别分类器
            # classifier = cv2.CascadeClassifier(
            #     "static/haarcascades/haarcascade_frontalface_default.xml"
            # )
            # # 调用识别人脸
            # faces = classifier.detectMultiScale(
            #     img_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

            # print(faces)
            # 存储所有人脸的名字
            name_namelist = []
            # return json.dumps({'code': 200, 'msg': 'ss'}, cls=MyEncoder)
            other = os.listdir('static/data_faces_from_camera/other')
            others = []
            for i in range(len(other)):
                if (other[i] != '.DS_Store'):
                    others.append(other[i])
            now = int(round(time.time(), 2) * 1000)
            if len(others) > 0:
                last = max(others)[:-4]
            else:
                last = 0
            code = 800

            # 处理存放所有人脸特征的 CSV
            path_features_known_csv = "static/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)

            # 用来存放所有录入人脸特征的数组
            features_known_arr = []
            features_known_name = []

            # 读取已知人脸数据
            # known faces
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, len(csv_rd.loc[i, :])):
                    features_someone_arr.append(csv_rd.loc[i, :][j])
                name = features_someone_arr.pop()
                features_known_name.append(name)
                features_known_arr.append(features_someone_arr)

            data = []
            # 检测到人脸
            if len(faces) != 0:
                # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
                features_cap_arr = []
                for i in range(len(faces)):
                    shape = predictor(img_rd, faces[i])
                    features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))
                    # print(shape)
                    # print(features_cap_arr)
                # return json.dumps({'code': 200, 'msg': 'ss'}, cls=MyEncoder)

                # 遍历捕获到的图像中所有的人脸
                for k in range(len(faces)):
                    # 让人名跟随在矩形框的下方
                    # 确定人名的位置坐标
                    # 先默认所有人不认识，是 unknown
                    name_namelist.append("unknown")

                    # 每个捕获人脸的名字坐标
                    # pos_namelist.append(
                    #     tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                    # print(features_known_arr)
                    # 对于某张人脸，遍历所有存储的人脸特征
                    for i in range(len(features_known_arr)):
                        # da={}
                        # 将某张人脸与存储的所有人脸数据进行比对
                        compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                        print(compare)
                        if compare == "same":  # 找到了相似脸
                            name_namelist[k] = features_known_name[i]
                            print(111)
                            # print(faces[k])
                            da = {'name':name_namelist[k],'left':faces[k].left(),'top':faces[k].top(),'bottom':faces[k].bottom(),'right':faces[k].right(),'is_known':1}
                            data.append(da)
                        # else 不相似的脸 截图保存 等待后续操作
                        else:
                            print(222)
                            da = {'name':name_namelist[k],'left':faces[k].left(),'top':faces[k].top(),'bottom':faces[k].bottom(),'right':faces[k].right(),'is_known':0}
                            data.append(da)
                            path_make_dir = "static/data_faces_from_camera/"
                            # for kd, d in enumerate(faces):
                            #     # 计算矩形框大小
                            #     height = (d.bottom() - d.top())
                            #     width = (d.right() - d.left())
                            #     hh = int(height / 2)
                            #     ww = int(width / 2)
                            #     # 根据人脸大小生成空的图像
                            #     im_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)
                            #
                            #     for ii in range(height * 2):
                            #         for jj in range(width * 2):
                            #             im_blank[ii][jj] = img_rd[d.top() - hh + ii][d.left() - ww + jj]
                            #     cv2.imwrite(path_make_dir + "/other/" + str(now) + ".jpg", im_blank)

                            # 计算矩形框大小
                            height = (faces[k].bottom() - faces[k].top())
                            width = (faces[k].right() - faces[k].left())
                            hh = int(height / 2)
                            ww = int(width / 2)
                            # 根据人脸大小生成空的图像
                            im_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)
                            # print(im_blank)
                            # if (height * 2 < 720):
                            for ii in range(height * 2):
                                    # if (width * 2 < 720):
                                for jj in range(width * 2):
                                            # if (faces[k].top() - hh + ii < 720):
                                    im_blank[ii][jj] = img_rd[faces[k].top() - hh + ii][faces[k].left() - ww + jj]
                            cv2.imwrite(path_make_dir + "/other/" + str(now) + ".jpg", im_blank)

                print(data)
            end = time.time()
            tt = end - start
            print('mm:',tt)
            # print(time.strftime("%H:%M:%S", time.localtime()))
            return json.dumps(data, cls=MyEncoder)
        except:
            return json.dumps([], cls=MyEncoder)




class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        else:
            return super(MyEncoder, self).default(obj)

# 返回单张图像的 128D 特征
def return_128d_features(path_img):
    # Dlib 正向人脸检测器
    detector = dlib.get_frontal_face_detector()
    # Dlib 人脸预测器
    predictor = dlib.shape_predictor("static/data_dlib/shape_predictor_5_face_landmarks.dat")

    # Dlib 人脸识别模型
    # Face recognition model, the object maps human faces into 128D vectors
    facerec = dlib.face_recognition_model_v1("static/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)

    print("检测到人脸的图像：", path_img, "\n")

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    # print(face_descriptor)
    return face_descriptor


# 从 CSV 中读取数据，计算 128D 特征的均值
def compute_the_mean(path_csv_rd):
    column_names = []

    # 128列特征
    for feature_num in range(128):
        column_names.append("features_" + str(feature_num + 1))

    # 利用pandas读取csv
    rd = pd.read_csv(path_csv_rd, names=column_names)

    # 存放128维特征的均值
    feature_mean = []

    for feature_num in range(128):
        tmp_arr = rd["features_" + str(feature_num + 1)]
        tmp_arr = np.array(tmp_arr)

        # 计算某一个特征的均值
        tmp_mean = np.mean(tmp_arr)
        feature_mean.append(tmp_mean)
    return feature_mean







def gen(camera):
    while True:
        frame = camera.get_frame()
        # print(frame['faces'][0].top())
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame['img'] + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port = 5000)