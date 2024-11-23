from math import degrees, radians, cos, sin, atan2
import cv2
import numpy as np

from geo.utils import *


def bearing(lon1, lat1, lon2, lat2):
    rad_lat1 = radians(lat1)
    rad_lon1 = radians(lon1)
    rad_lat2 = radians(lat2)
    rad_lon2 = radians(lon2)
    delta_lon = rad_lon2 - rad_lon1
    y = sin(delta_lon) * cos(rad_lat2)
    x = cos(rad_lat1) * sin(rad_lat2) - sin(rad_lat1) * cos(rad_lat2) * cos(delta_lon)
    return (degrees(atan2(y, x)) + 360) % 360


def rendering():
    points, p1, p2 = parse_node_csv()
    segments = parse_line_csv()

    base_img = np.ones(shape=(38200, 21600, 3))
    height, width, channel = base_img.shape
    print(width, height, channel)

    (base_min_x, base_max_y) = p1
    (base_max_x, base_min_y) = p2
    x_len = base_max_x - base_min_x
    y_len = base_max_y - base_min_y

    check_list = []
    for key, seg in segments.items():
        # print(key, seg)
        is_bi = len(seg) >= 2
        for seg_id, (f_node_id, t_node_id) in seg.items():
            # From which node
            x1, y1 = points[f_node_id]
            pos1 = (int((x1 - base_min_x) / x_len * width), int((base_max_y - y1) / y_len * height))
            # To which node
            x2, y2 = points[t_node_id]
            pos2 = (int((x2 - base_min_x) / x_len * width), int((base_max_y - y2) / y_len * height))

            dias_value = 0
            deg = (bearing(x1, y1, x2, y2) + 90) % 360
            rad = radians(deg)
            if is_bi:
                dias = (int(sin(rad)*dias_value), -int(cos(rad)*dias_value))
            else:
                dias = (0, 0)

            if f_node_id not in check_list:
                cv2.circle(base_img, pos1, 2, (0, 255, 0), thickness=-1)
            else:
                check_list.append(f_node_id)
            if t_node_id not in check_list:
                cv2.circle(base_img, pos2, 2, (0, 255, 0), thickness=-1)
            else:
                check_list.append(t_node_id)
            # Draw segment line
            cv2.line(base_img,
                     (pos1[0] + dias[0], pos1[1] + dias[1]),
                     (pos2[0] + dias[0], pos2[1] + dias[1]),
                     (0, 255, 0), thickness=1)

    rotated = cv2.rotate(base_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('base image', rotated)
    cv2.imwrite('./geo/base_image_point.png', rotated)


def main():
    parse_osm()
    # rendering()

    points, *_ = parse_node_csv()
    stat_dict = parse_line_csv2()
    print(len(stat_dict), len(points))

    count = 1
    for i in range(len(points)):
        if i not in points.keys():
            print('p: ', i)
        if i not in stat_dict.keys():
            print('s: ', i)

    for i, key in enumerate(sorted(stat_dict.keys())):
        value = stat_dict[key]
        # print(i, key, [points[v] for v in stat_dict[key]])
        if len(value) > count:
            print(key, type(key), value)
            count = len(value)


if __name__ == '__main__':
    main()
