# -*- coding: utf-8 -*-
"""rename_labels.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mqKd53WYE-vaXpLTiBp45jljHiMRJtYX
"""

#라벨 이름 변경 코드
import os
import re

# 파일들이 들어있는 폴더 경로
folder_path = "/content/drive/MyDrive/dataset/abc"  # ← 여기에 실제 경로 입력

# 변경된 파일 수 카운트
renamed_count = 0

# 디렉토리 내 모든 파일 순회
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        # 정규표현식으로 숫자 추출
        match = re.search(r"-(\d+)_png", filename)
        if match:
            number = match.group(1)
            new_name = f"그림{number}.txt"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # 이름 충돌 방지
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                renamed_count += 1
            else:
                print(f"⚠️ 파일명 충돌: {new_name} (건너뜀)")

print(f"\n✅ 총 {renamed_count}개의 파일명을 변경했습니다.")