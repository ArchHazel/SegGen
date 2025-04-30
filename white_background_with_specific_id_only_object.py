import os
import cv2
import numpy as np
import natsort

original_image_path = "./keyframes/images/"
mask_data_dir = "./outputs/mask_data/"
output_dir = "./outputs/merged_images/"
output_video_path = "./outputs/merged_video.mp4" 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_num = len([f for f in os.listdir(original_image_path) if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))])
full_image_list = os.listdir(original_image_path)

full_image_list = natsort.natsorted(full_image_list, key=lambda x: int(x.split('.')[0]))
full_name_list = [os.path.splitext(x)[0] for x in full_image_list]  

print(f'full_image_list: {full_image_list}')
print(f"Total frames found: {frame_num}")

selected_id = full_image_list[0].split('.')[0]  
print(f'selected_id: {selected_id}')
selected_id = 46
object_list = []
images_for_video = []  

for i in full_name_list:
    original_file = [f for f in full_image_list if f.startswith(f"{i}.")][0]
    image_name = os.path.join(original_image_path, original_file)
    
    mask_name = os.path.join(mask_data_dir, f"mask_{i}.npy")
    print(f'image_name: {image_name}')
    print(f'mask_name: {mask_name}')
    
    mask_img = np.load(mask_name)
    mask_img = mask_img.astype(np.uint16)
    original_image = cv2.imread(image_name)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    merged_image = np.ones_like(original_image) * 255 
    merged_image[mask_img == selected_id] = original_image[mask_img == selected_id]


    output_path = os.path.join(f"./outputs/merged_images/", f"merged_{int(i):06d}.jpg")
    merged_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, merged_image)
    

    avg_pixel_value = np.mean(merged_image)
    print(f'Average pixel value for {i}: {avg_pixel_value}')
    

    if avg_pixel_value < 255:
        object_list.append(i)
        images_for_video.append(merged_image)  


for i in object_list:
    print(f'object_list: {i}')


if images_for_video:

    height, width, _ = images_for_video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height)) 

    for image in images_for_video:
        out.write(image) 

    out.release() 
    print(f"Video saved at {output_video_path}")
else:
    print("No images to create a video.")
