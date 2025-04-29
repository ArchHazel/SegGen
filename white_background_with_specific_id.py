import os
import cv2
import numpy as np


original_image_path = "./outputs_annex/rgb/"
mask_data_dir = "./outputs_annex/mask_data/"
output_dir = "./outputs/merged_images/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_num = 200
selected_id  =19
for idx in range(1, 20):
    for i in range(frame_num):
        image_name = os.path.join(original_image_path, f"{i}.png")
        mask_name = os.path.join(mask_data_dir, f"mask_{i}.npy")
        mask_img = np.load(mask_name)
        mask_img = mask_img.astype(np.uint16)

        original_image = cv2.imread(image_name)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Merge the images
        merged_image = np.ones_like(original_image) * 255  # Create a white background
        merged_image[mask_img == selected_id] = original_image[mask_img == selected_id]

        # Save the merged image
        if not os.path.exists(os.path.join(output_dir, f"{idx}")):
            os.makedirs(os.path.join(output_dir, f"{idx}"))
        output_path = os.path.join(f"./outputs/merged_images/{idx}", f"merged_{i:06d}.jpg")
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, merged_image)
