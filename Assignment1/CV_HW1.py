import numpy as np
import cv2 as cv
import os
import evaluation as eval

###############################################################
##### This code has been tested in Python 3.6 environment #####
###############################################################

def main():

    ##### Set parameters
    threshold = 28

    ##### Set path
    input_path = './input_resize'    # input path
    gt_path = './groundtruth_resize'       # groundtruth path
    bg_result_path = './background'
    result_path = './result'        # result path

    ##### load input
    input = [img for img in sorted(os.listdir(input_path)) if img.endswith(".jpg")]

    ##### 이전 결과 지우기
    for img in os.scandir(result_path):
        os.remove(img.path)

    ##### first frame and first background
    frame_current = cv.imread(os.path.join(input_path, input[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    frame_bg_gray = frame_current_gray

    print(frame_current.shape)
    height = frame_current.shape[0]
    width = frame_current.shape[1]

    ##### 배경 modeling에 사용할 이전 frame들을 저장하는 np array
    bg_list = np.zeros((0,height,width))

    for image_idx in range(len(input)):
        print(image_idx)

        ##### calculate foreground region
        diff = frame_current_gray - frame_bg_gray
        diff_abs = np.abs(diff).astype(np.float64)

        ##### make mask by applying threshold
        ##### diff가 threshold 이상이면 1.0, 아니면 0.0을 저장한 mask(nparray)
        frame_diff = np.where(diff_abs > threshold, 1.0, 0.0)

        ##### apply mask to current frame
        # current_gray_masked       : frame_current_gray에서 움직이는 부분만 남긴거
        # current_gray_masked_mk2   : current_gray_masked를 binary(흰/검) 이미지로 변환
        current_gray_masked = np.multiply(frame_current_gray, frame_diff)
        current_gray_masked_mk2 = np.where(current_gray_masked > 0, 255.0, 0.0)

        ##### final result
        result = current_gray_masked_mk2.astype(np.uint8)

        ##### result에서 노이즈 제거 및 opening
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # result = cv.medianBlur(result,13)
        # result = cv.erode(result,kernel)
        # result = cv.medianBlur(result,11)
        # result = cv.dilate(result,kernel)
        # result = cv.dilate(result,kernel)

        ##### show background & result
        bg = frame_bg_gray.astype(np.uint8)
        cv.imshow('result', result)
        cv.imshow('background', bg)

        ##### renew background
        ############################################
        frame_reshape = np.reshape(frame_current_gray,(1,height,width))
        bg_list = np.append(bg_list,frame_reshape,axis=0)

        # 500 frame이 넘어가면 최근 500개의 frame만 사용해서 bg modeling
        if len(bg_list)>300:
            bg_list = bg_list[len(bg_list)-300:]

        # 이전 frame들의 중간값을 background로 설정
        frame_bg_gray = np.median(bg_list, axis=0)
        ############################################

        ##### make result file
        ##### Please don't modify path
        cv.imwrite(os.path.join(result_path, 'result%06d.png' % (image_idx + 1)), result)
        cv.imwrite(os.path.join(bg_result_path, 'bg%06d.png' % (image_idx + 1)), bg)

        ##### end of input
        if image_idx == len(input) - 1:
            break

        ##### read next frame
        frame_current = cv.imread(os.path.join(input_path, input[image_idx + 1]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        ##### If you want to stop, press ESC key
        k = cv.waitKey(30) & 0xff
        if k == 27:
            print("ESC로 종료됨")
            break

    ##### evaluation result
    eval.cal_result(gt_path, result_path)

if __name__ == '__main__':
    main()