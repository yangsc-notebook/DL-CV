import face_recognition
import cv2
from cv2 import cv2
import os
#1、准备工作、
#存图像的数据库
face_databases_dir='/home/ysc/dataFX/deeplearn/opencv_start/face_databases' 
#存用户名(标签)
user_names=[]
#存用户面部特征
user_faces_encodings=[]
#2、正式工作
#得到图像数据库中所有文件名
files=os.listdir(face_databases_dir)
#循环读取文件名进行进一步的处理
for image_shot_name in files:
    #截图文件名的.前部分作为用户名，作为用户名存入user_names的列表中
    user_name,_=os.path.splitext(image_shot_name)
    user_names.append(user_name)
    #读取图片文件中面部特征信息存入user_face_enconding中
     #拼接存储地址
    image_file_name = os.path.join(face_databases_dir,image_shot_name)
    #导入图片
    image_file = face_recognition.load_image_file(image_file_name)
    #读取导入图片的特征信息
    # [0]的作用(一张图图中只有一个人的面部特征，所以用0)
    face_encoding = face_recognition.face_encodings(image_file)[0]
    user_faces_encodings.append(face_encoding)




#打开摄像头，读取摄像头拍摄到的图画
#定位到图画中人的脸部，并用绿色的框框把人的脸部框起来,并用姓名做标注，未知用户使用unknow
#定位和锁定目标人物，改使用红色的框把目标人物的脸框起来
red_name=['yangsc','yang']

#1、打开摄像头，获取摄像头对象
video_capture=cv2.VideoCapture(0)
#2、不停地循环取获取摄像头拍摄到的图画，并作进一步的处理
while True:
  #1、获取摄像图拍摄到的图画
  ret,frame= video_capture.read() #frame 摄像头所拍摄的画面
  #2、从拍摄画面中提取出人的脸部所在区域，(可能有多个)
  #['第一个人脸所在区域'，‘第二个人脸所在区域’]
  face_locations=face_recognition.face_locations(frame)
  #从所有人的头像所在区域提取脸部特征（可能有多个）
  #[“第一个人脸对应面部特征‘，’第二个人脸对应面部特征‘]
  face_encodings=face_recognition.face_encodings(frame,face_locations)

  #定义用于存储拍摄到的用户的姓名的列表
  #['第一个人的姓名'，’第二个人的姓名‘]
  #如果特征匹配不上数据库中特征，则unknown
  names=[]
  for face_one in face_encodings:
    matchs=face_recognition.compare_faces(user_faces_encodings,face_one)
    #comapre_faces(数据库面部特征列表，视频中的面部特征)
    #返回结果布尔值
    name="UnKnown"
    for index,is_match in enumerate(matchs):
      if is_match:
        name=user_names[index]
        break
    names.append(name)


  #3、循环遍历人的脸部所在区域，并画框，在框上标注姓名
  #zip
  #zip(['第一个人脸所在区域'，‘第二个人脸所在区域’],['第一个人的姓名'，’第二个人的姓名‘])
  #'第一个人脸所在区域','第一个人的姓名'
  #‘第二个人脸所在区域’,’第二个人的姓名‘

  for (top,right,bottom,left),name in zip(face_locations,names):
    color=(0,255,0)
    if name in red_name:
      color=(0,0,255)
    #在人像所在区域画框
    cv2.rectangle(frame,(left,top),(right,bottom),color,2)
    font=cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame,name,(left,top-10),font,0.5,color,1)

  #4、通过opencv把画面显示出来
  cv2.imshow("video",frame)
  #5、设定按q退出while循环，退出程序机制
  if cv2.waitKey(1) & 0xFF==ord('q'):
    break;
#3、退出程序的时候，释放摄像头或者其他资源
video_capture.release()
cv2.destroyAllWindows()