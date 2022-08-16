import cv2 as cv
import glob
import numpy as np
import json


def calibrate_single_camera(images_folder):
    """ calibrate_camera is function to calibrate single camera which takes
    the input as images path and return the camera calibration matrix and distortion
    The distance between chess board and Camera center point is considered approximately
    180-200mm"""

    # Size of chessboard used
    chessboardSize = (8, 5)

    # Frame size captured using ximea cam tool
    frameSize = (1296, 972)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    size_of_chessboard_squares_mm = 20
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    #Iterates over the the file folder where images are been saved
    images = glob.glob(images_folder)

    for image in images:

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            #at subpixel level
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    #calculates the camera matrix, rmse, distortion values
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    # print('rmse:', ret)
    # print('camera matrix:\n', cameraMatrix)
    # print('distortion coeffs:', dist)
    # print('Rs:\n', rvecs)
    # print('Ts:\n', tvecs)

    return cameraMatrix , dist



def Stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    """ Function Stereo calibrate takes the input as RMSE and distrotion value we calculate in previous function
    and return s the rotation and translation matrix of camera1 with respect to camera2 """

    # Size of chessboard used
    chessboardSize = (8, 5)

    # Frame size captured using ximea cam tool
    frameSize = (1296, 972)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 90, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    size_of_chessboard_squares_mm = 20
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #Iterates over the the file folder where images are been saved
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)

    # To create partition between left and right images
    images_names_1 = images_names[:len(images_names)//2]
    images_names_2 = images_names[len(images_names)//2:]

    for im1, im2 in zip(images_names_1, images_names_2):

        img_1 = cv.imread(im1)
        gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)

        img_2 = cv.imread(im2)
        gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

        # Find the chess board corners

        ret_1, corners1 = cv.findChessboardCorners(gray_1, chessboardSize, None)
        ret_2, corners2 = cv.findChessboardCorners(gray_2, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret_1 == True and ret_2 == True:
            corners1 = cv.cornerSubPix(gray_1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray_2, corners2, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, Rot_mat, Trans_mat, E, F = cv.stereoCalibrate(objpoints,
                                                               imgpoints_left, imgpoints_right, mtx1,
                                                                 dist1,
                                                                 mtx2, dist2, frameSize, criteria=criteria,
                                                                 flags=stereocalibration_flags)

    return Rot_mat, Trans_mat


if __name__ == "__main__":

    left_image_folder = r"..\images\left\*.png"
    right_image_folder = r"..\images\right\*.png"
    stereo_image_folder = r"..\images\stereo\*.png"

    #calibaration params of single camera.
    left_camera_matrix, left_dist = calibrate_single_camera(images_folder = left_image_folder )
    right_camera_matrix, right_dist = calibrate_single_camera(images_folder= right_image_folder)

    #Converting nd array to list
    left_camera_matrix_list = left_camera_matrix.tolist()
    left_dist_list = left_dist.tolist()
    right_camera_matrix_list = right_camera_matrix.tolist()
    right_dist_list = right_dist.tolist()

    #save the params to a json file.
    cam_params = {
        "left_camera_matrix":left_camera_matrix_list,
        "left_dist": left_dist_list,
        "right_camera_matrix": right_camera_matrix_list,
        "right_dist": right_dist_list
    }

    # Serializing json
    json_object = json.dumps(cam_params, indent=4)

    # Writing to sample.json
    with open("../params/single_camera_params.json", "w") as outfile:
        outfile.write(json_object)


    #calibaration params of stereo camera.
    R, T = Stereo_calibrate(left_camera_matrix, right_camera_matrix,
                            right_camera_matrix, right_dist, frames_folder = stereo_image_folder)

    R = R.tolist()
    T = T.tolist()

    cam_params = {
        "rotation":R,
        "translation": T

    }

    # Serializing json
    json_object = json.dumps(cam_params, indent=4)

    # Writing to sample.json
    with open("../params/stereo_camera_params.json", "w") as outfile:
        outfile.write(json_object)






