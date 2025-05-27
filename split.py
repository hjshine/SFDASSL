import os
import random

# 이미지 파일들이 위치한 폴더
images_directory = '/data1/datasets/cityscapes/CityScapesFoggy/yolov5_format/images/train'  # 절대 경로

# 이미지 파일 확장자
image_extension = '.jpg'

# 이미지 파일 목록을 가져옴
image_files = [os.path.join(images_directory, file) for file in os.listdir(images_directory) if file.endswith(image_extension)]

# 파일 목록을 랜덤하게 섞음
random.shuffle(image_files)

# 비율을 설정
split_ratio = 0.98  # 예를 들어, 80%의 파일을 train.txt로, 나머지 20%를 train_unlabel.txt로 설정

# 비율에 따라 파일 목록을 분할
split_index = int(len(image_files) * split_ratio)
train_unlabel_files = image_files[:split_index]
train_files = image_files[split_index:]

# train_files = image_files[:split_index]
# train_unlabel_files = image_files[split_index:]

# train.txt 파일에 train 파일 경로를 적음
with open('train_foggy_2.txt', 'w') as f:
    for image_file in train_files:
        f.write(image_file + '\n')

# train_unlabel.txt 파일에 train_unlabel 파일 경로를 적음
with open('train_foggy_unlabel_98.txt', 'w') as f:
    for image_file in train_unlabel_files:
        f.write(image_file + '\n')

print("Files have been split and written to train.txt and train_unlabel.txt")
