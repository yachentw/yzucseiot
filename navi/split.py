import cv2
import os

def split_and_rotate_image(image_path, grid_size, output_folder='output_images'):
    """
    將影像切割成格子狀，並針對每個小區塊進行0-360度的旋轉（每次10度）並存檔
    :param image_path: 原始影像的路徑
    :param grid_size: (行數, 列數) 表示將影像分成的行數和列數
    :param output_folder: 輸出檔案存放的資料夾
    """
    # 讀取原始影像
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法載入影像：{image_path}")
        return
    
    # 取得影像的高和寬
    img_height, img_width, _ = img.shape
    rows, cols = grid_size

    # 每個區塊的高和寬
    block_height = img_height // rows
    block_width = img_width // cols

    # 如果沒有輸出資料夾，則建立該資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 將影像切割成格子狀
    for i in range(rows):
        for j in range(cols):
            # 計算每個小區塊的起始和結束座標
            start_y = i * block_height
            end_y = (i + 1) * block_height
            start_x = j * block_width
            end_x = (j + 1) * block_width
            
            # 從原始影像中裁切出該小區塊
            block = img[start_y:end_y, start_x:end_x]

            # 產生小區塊的儲存路徑
            block_folder = os.path.join(output_folder, f"block_{i}_{j}")
            
            # 如果小區塊的儲存資料夾不存在，則建立
            if not os.path.exists(block_folder):
                os.makedirs(block_folder)
            
            # 旋轉影像0°到360°（每10°旋轉一次）
            for angle in range(0, 360, 10):
                rotated_block = rotate_image(block, angle)
                block_filename = os.path.join(block_folder, f"block_{i}_{j}_angle_{angle}.jpg")
                
                # 儲存旋轉後的影像
                cv2.imwrite(block_filename, rotated_block)
                print(f"已儲存旋轉小區塊：{block_filename}")

    print(f"完成影像分割與旋轉，所有小區塊已儲存在：{output_folder}")


def rotate_image(image, angle):
    """
    旋轉影像
    :param image: 要旋轉的影像
    :param angle: 旋轉的角度
    :return: 旋轉後的影像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 計算旋轉矩陣，然後執行仿射變換
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated


if __name__ == "__main__":
    # 設定參數
    image_path = 'parking.jpg'  # 替換為您想要切割的影像路徑
    grid_size = (2, 2)  # 切割成 2x2 的格子
    output_folder = 'output_images'  # 輸出的小區塊儲存的資料夾

    # 執行影像分割與旋轉
    split_and_rotate_image(image_path, grid_size, output_folder)
