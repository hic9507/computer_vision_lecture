import argparse
import cv2, random, numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--path', default='./lena.png', help='Image path')
params = parser.parse_args()
img = cv2.imread(params.path)


w, h = img.shape[1], img.shape[0]


def rand_pt(mult=1.):
    return(random.randrange(int(w*mult)),
    random.randrange(int(h*mult)))


mouse_pressed = False

def mouse_callback_draw_rectangle(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image_to_show)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed=False
        e_x, e_y = x,y
        cv2.rectangle(image_to_show, (s_x,s_y),
            (x,y), (0,255,0),1)
        cv2.imshow('image', image_to_show)


def mouse_callback_draw_line(event, x,y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image_to_show)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed=False
        e_x, e_y = x, y
        cv2.line(image_to_show, (s_x, s_y),
            (x, y), (0,255,0),1)
        cv2.imshow('image', image_to_show)


def mouse_callback_draw_arrowedline(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image_to_show)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed=False
        e_x, e_y = x, y
        cv2.arrowedLine(image_to_show, (s_x, s_y),
            (x, y), (0,255,0),1)
        cv2.imshow('image', image_to_show)


image_to_show = np.copy(img)
finish = False

while not finish:
    cv2.imshow('image_to_show', image_to_show)
    key = cv2.waitKey(0)

    if key == ord('r'):
        cv2.setMouseCallback('image_to_show', mouse_callback_draw_rectangle)

    elif key == ord('l'):
        cv2.setMouseCallback('image_to_show', mouse_callback_draw_line)

    elif key == ord('a'):
        cv2.setMouseCallback('image_to_show', mouse_callback_draw_arrowedline)
        image_to_show = np.copy(image_to_show)

    elif key == ord('c'):
        image_to_show = np.copy(img)

    elif key == 27:
        cv2.destroyAllWindows()
        break
    elif key == ord('w'):
        cv2.imwrite('lena_draw.png', image_to_show)
