# Image-Agumentation-for-ATM-Module-Assembly-Defect-Detection-Learning

1. 이미지 전처리

1.1 결함종류, rotation별로 직접 디렉토리 구분
dent, nick, scratch
rotation1 ~ rotation8

1.2 resize to 1024x1024
resize_images.ipynb

1.3 roboflow 라벨링
roboflow.com 결함 이미지만 업로드 후 annotate
데이터셋 다운로드 후 라벨링 파일들을 하나의 디렉토리에 저장
라벨 파일명을 변경(그림20.png에 해당하는 라벨링 파일이름 -> 그림20.txt)
rename_labels.ipynb




2. 이미지 생성
2.1 찍힘 이미지 생성

2.1.1 모듈 종류(front1, main), rotation별로 결함 위치에 대해 crop
정상이미지도 같은 방식으로 crop 진행
정상이미지는 trainA, 결함이미지는 trainB에 저장

crop&paste_images.ipynb

정상이미지 crop
input_dir
"/content/drive/MyDrive/dataset/atm_dataset_1024x1024/front1/rotation1” 
output_dir
"/content/drive/MyDrive/dataset/CUT_dataset/front1_nick_rotation1_CUT/trainA"

결함이미지 crop
input_dir
"/content/drive/MyDrive/dataset/atm_dataset_1024x1024/front1_nick/rotation1” 
output_dir
"/content/drive/MyDrive/dataset/CUT_dataset/front1_nick_rotation1_CUT/trainB”

crop좌표(결함 위치)
front1
rotation1(450, 300, 50)
rotation2(525, 375, 50)
rotation3(600, 500, 50)
rotation4(600, 450, 50)
rotation5(325, 350, 50)
rotation6(670, 370, 50)
rotation7(650, 620, 50)
rotation8(375, 400, 50)
main
rotation1(370, 570, 50)
rotation2(370, 520, 50)
rotation3(430, 430, 50)

2.1.2 trainA, trainB를 복사하여 testA, testB 디렉토리를 생성
%cd /content/drive/MyDrive/dataset/CUT_dataset/front1_nick_rotation1_CUT
%cp trainA testA -r
%cp trainB testB -r

2.1.3 생성된 trainA, trainB 디렉토리 파일들을 모듈별로 통합하여 train데이터셋 준비
%mkdir -p /content/drive/MyDrive/dataset/CUT_dataset/front1_nick_CUT/trainA
%mkdir -p /content/drive/MyDrive/dataset/CUT_dataset/front1_nick_CUT/trainB
%cd /content/drive/MyDrive/dataset/CUT_dataset/front1_nick_rotation1_CUT
%cp trainA/* ../front1_nick_CUT/trainA
%cp trainB/* ../front1_nick_CUT/trainB

2.1.4 준비된 데이터셋으로 CUT모델 학습
CUT.ipynb
!python train.py --dataroot /content/drive/MyDrive/dataset/CUT_dataset/front1_nick_CUT \
                 --name /content/drive/MyDrive/result/front1_nick_CUT_result \
                 --CUT_mode CUT

2.1.5 학습된 모델로 각 모듈, rotation에 대해 이미지 생성
!python test.py \
  --dataroot /content/drive/MyDrive/dataset/CUT_dataset/front1_nick_rotation1_CUT \
  --name /content/drive/MyDrive/result/front1_nick_CUT_reusult \
  --model cut \
  --num_test 100

2.1.6 생성된 이미지를 정상 이미지에 다시 붙여넣기
crop&paste_images.ipynb
background_dir(정상 이미지 디렉토리)
"/content/drive/MyDrive/dataset/atm_dataset_1024x1024/front1/rotation1” 
patch_dir(결함 이미지 디렉토리)
”/content/drive/MyDrive/result/front1_nick_CUT_reusult/rotation1/images/fake_B"
output_dir(저장할 디렉토리)
"/content/drive/MyDrive/result/atm_dataset_1024x1024_CUT_front1_nick_reusult"


2.2 스크래치 이미지 생성

2.2.1 라벨링 파일 기준으로 결함 부분만 crop
crop_boundingbox.ipynb
image_dir = "/content/drive/MyDrive/dataset/atm_dataset"
label_dir = "/content/drive/MyDrive/dataset/labels"
output_root = "/content/drive/MyDrive/dataset/cropped_defects"

2.2.2 resize to 128x128
resize_images.ipynb

2.2.3 StyleGAN3 모델 학습
SytleGAN3.ipynb
!python train.py --outdir=/content/drive/MyDrive/result/scratch_128x128_StyleGAN3_training-runs \
            --data=/content/drive/MyDrive/dataset/cropped_defects/scratch_128x128 \
            --cfg=stylegan3-t \
            --gpus=1 \
            --batch=32 \
            --gamma=0.5 \
            --batch-gpu=16 \
            --snap=10 \
            --kimg=3200 \
            --metrics=none \
            --resume=/content/drive/MyDrive/result/scratch_128x128_StyleGAN3_training-runs/00000-stylegan3-t-scratch_128x128-gpus1-batch32-gamma0.5/network-snapshot-001800.pkl


2.2.4 학습된 모델로 이미지 생성
SytleGAN3.ipynb
!python gen_images.py \
  --outdir=/content/drive/MyDrive/result/scratch_128x128_StyleGAN3_result \
  --trunc=1 \
  --seeds=0-99 \
  --noise-mode=random \
  --network=/content/drive/MyDrive/result/scratch_128x128_StyleGAN3_training-runs/00000-stylegan3-t-scratch_128x128-gpus1-batch32-gamma0.5/network-snapshot-001800.pkl

2.2.5 결함 이미지 boundingbox에 다시 붙여넣기
paste_to_boundingbox.ipynb


