# import os
# import shutil

# # Папка, где лежат исходные файлы
# source_folder = r"C:\Users\vkash\Desktop\FilterData\Test\Normal"

# # Папка, куда сохранять копии (может быть та же, что source_folder, или новая)
# destination_folder = r"C:\Users\vkash\Desktop\FilterData\videos"

# # Создаём папку назначения, если её нет
# os.makedirs(destination_folder, exist_ok=True)

# for filename in os.listdir(source_folder):
#     if filename.endswith("_converted.avi") and filename.startswith("t_"):
#         # Вырезаем середину (например, n001)
#         middle = filename[3:-14]  # 2 = len("t_"), -13 = len("_converted.avi")
        
#         # Формируем новое имя
#         new_name = f"n_{middle}.mp4"
        
#         # Полные пути
#         old_path = os.path.join(source_folder, filename)
#         new_path = os.path.join(destination_folder, new_name)
        
#         # Копируем файл с новым именем
#         shutil.copy2(old_path, new_path)
#         print(f"Copied: {filename} → {new_name}")

# print("Копирование завершено.")


import os
import shutil

# Папка, где лежат исходные файлы
source_folder = r"C:\Users\vkash\Desktop\FilterData\Train\Normal"

# Папка, куда сохранять копии
destination_folder = r"C:\Users\vkash\Desktop\FilterData\videos"

# Начинаем с номера 47
current_index = 47

# Создаём папку назначения, если её нет
os.makedirs(destination_folder, exist_ok=True)

# Получаем список файлов, сортируем по имени (чтобы шли по порядку)
files = sorted(f for f in os.listdir(source_folder) if f.endswith("_converted.avi") and f.startswith("n"))

for filename in files:
    # Формируем новое имя, например: n_047.mp4
    new_name = f"n_{current_index:03d}.mp4"
    
    # Полные пути
    old_path = os.path.join(source_folder, filename)
    new_path = os.path.join(destination_folder, new_name)
    
    # Копируем файл с новым именем
    shutil.copy2(old_path, new_path)
    print(f"Copied: {filename} → {new_name}")
    
    current_index += 1

print("Копирование завершено.")



# python -m weapon_gait.cli train --manifest data\manifest.csv `
#                                 --pose-backend mediapipe `
#                                 --gait-backend stats `
#                                 --model runs\rf.joblib