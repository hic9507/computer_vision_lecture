import cv2
import matplotlib.pyplot as plt
import numpy as np

prev_pts =None
prev_gray_frame=None
tracks = None

images = []
images.append(cv2.imread('./stitching/dog_a.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('./stitching/dog_b.jpg', cv2.IMREAD_COLOR))

show_img = np.copy(images)


def display_flow(img, flow, stride=40):
    for index in np.ndindex(flow[::stride, ::stride, ::stride].shape[:2]):
        pt1 = tuple(i * stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10 * delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(img, pt1[::-1], pt2[::-1],
                            (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4)

    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1,
                                  cv2.NORM_MINMAX)

    cv2.imshow('optical flow', img)
    cv2.imshow('optical flow magnitude', norm_opt_flow)
    cv2.waitKey()
    # k = cv2.waitKey(1)
    #
    # if k == 27:
    #     return 1
    # else:
    #     return 0


prev_frame = cv2.cvtColor(show_img[0], cv2.COLOR_BGR2GRAY)  # prev_frame
prev_frame = cv2.resize(prev_frame, (0, 0), None, 0.5, 0.5)
init_flow = True

gray_frame = cv2.cvtColor(show_img[1], cv2.COLOR_BGR2GRAY)  # frame
gray_frame = cv2.resize(gray_frame, (0, 0), None, 0.5, 0.5)

opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 5, 13, 10, 5, 1.1,
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

if init_flow:
    opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 5, 13, 10, 5, 1.1,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    init_flow = False

else:
    opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, opt_flow, 0.5, 5, 13, 10, 5, 1.1,
                                            cv2.OPTFLOW_USE_INITIAL_FLOW)
prev_frame = np.copy(gray_frame)

if display_flow(gray_frame, opt_flow):
    pass
# display_flow(gray_frame, opt_flow)

##########################################################################################################
prev_frame = cv2.cvtColor(show_img[0], cv2.COLOR_BGR2GRAY)  # prev_frame
prev_frame = cv2.resize(prev_frame, (0, 0), None, 0.5, 0.5)
init_flow = True

gray_frame = cv2.cvtColor(show_img[1], cv2.COLOR_BGR2GRAY)  # frame
gray_frame = cv2.resize(gray_frame, (0, 0), None, 0.5, 0.5)

flow_DualTVL1 = cv2.createOptFlow_DualTVL1()

opt_flow = flow_DualTVL1.calc(prev_frame, gray_frame, None)

if init_flow:
    opt_flow = flow_DualTVL1.calc(prev_frame, gray_frame, None)
    init_flow = False

if not flow_DualTVL1.getUseInitialFlow():
    opt_flow = flow_DualTVL1.calc(prev_frame, gray_frame, None)
    flow_DualTVL1.setUseInitialFlow(True)
else:
    opt_flow = flow_DualTVL1.calc(prev_frame, gray_frame, opt_flow)

prev_frame = np.copy(gray_frame)

if display_flow(gray_frame, opt_flow):
    pass