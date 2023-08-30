import os
import shutil
import random

def data_assign(train_path, test_path, image_path, mask_path, rate=0.8):
    # train_path = 'dataset/train'
    # test_path = 'dataset/test'
    # image_path = "dataset/CXR_png"
    # mask_path = "dataset/mask"

    train_img = train_path + "/image"
    train_mask = train_path + "/label"

    test_img = test_path + "/image"
    test_mask = test_path + "/label"

    # delete the folder if it exists
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    os.mkdir(train_path)
    os.mkdir(train_img)
    os.mkdir(train_mask)
    os.mkdir(test_path)
    os.mkdir(test_img)
    os.mkdir(test_mask)

    # image_list = os.listdir(image_path) # CHNCXR_0001_0.png
    mask_list = os.listdir(mask_path)   # CHNCXR_0001_0_mask.png
    random.shuffle(mask_list)
    train_mask_list = mask_list[:int(len(mask_list)*rate)]
    test_mask_list = mask_list[int(len(mask_list)*rate):]

    # print(len(image_list))
    # print(len(mask_list))
    # print("training data:", len(train_mask_list))
    # print("testing data:", len(test_mask_list))

    for mask in train_mask_list:
        image = mask[:-9] + ".png"
        shutil.copy(image_path + "/" + image, train_img + "/" + image)
        # shutil.copy(mask_path + "/" + mask, train_mask + "/" + mask)
        shutil.copy(mask_path + "/" + mask, train_mask + "/" + image)

    for mask in test_mask_list:
        image = mask[:-9] + ".png"
        shutil.copy(image_path + "/" + image, test_img + "/" + image)
        shutil.copy(mask_path + "/" + mask, test_mask + "/" + image)

    return train_mask_list, test_mask_list

