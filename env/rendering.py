import copy

import cv2
import numpy as np


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


def transform(pos, box, width, height, w_p=0, h_p=0):
    min_x, min_y, max_x, max_y = box
    _width = width - 2 * w_p
    _height = height - 2 * h_p
    return (w_p + int((pos[0] - min_x) / (max_x - min_x) * _width),
            h_p + int((pos[1] - min_y) / (max_y - min_y) * _height))


class CVRender:
    def __init__(self, env, width=1200, height=1200, padding=0.05):
        self.video = cv2.VideoWriter(
            'trained/pickup_delivery.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            8,
            (width, height)
        )

        self.width = width
        self.height = height
        self.w_p = int(width * padding)
        self.h_p = int(height * padding)
        self.env = env
        self.box = (0, 0, env.size, env.size)

        self.__initialize(env, height, width)

    def __initialize(self, env, height, width):
        base_image = np.ones((height, width, 3), np.uint8) * 255
        w_p, h_p = self.w_p, self.h_p
        # Border
        cv2.rectangle(
            base_image,
            (w_p, h_p),
            (width - w_p, height - h_p),
            (0, 0, 0),
            thickness=5
        )
        # Merchants
        for i, task in enumerate(env.scenario.tasks):
            merchant = task.merchant
            pos = merchant.state.p_pos
            pos = transform(pos, self.box, width, height, w_p=w_p, h_p=h_p)
            radius = int(merchant.size / 30 * (width - 2 * w_p))
            cv2.circle(base_image, pos, radius, (255, 0, 0), thickness=2)
            cv2.putText(
                base_image,
                str(i + 1),
                (pos[0] - 5, pos[1] + 5) if i < 10 else (pos[0] - 10, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1,
                cv2.LINE_AA
            )
        # Walls
        # for pos in env.walls:
        #     pos = transform(pos, self.box, width, height, w_p=w_p, h_p=h_p)
        #     cv2.rectangle(base_image,
        #                   (pos[0] - radius, pos[1] - radius),
        #                   (pos[0] + radius, pos[1] + radius),
        #                   (0, 0, 0),
        #                   thickness=-1)
        self.base_img = base_image
        cv2.imwrite('trained/base_image.png', base_image)

    def draw(self, show=False):
        box = self.box
        width, height, w_p, h_p = self.width, self.height, self.w_p, self.h_p
        base_img = copy.deepcopy(self.base_img)

        # Buyers
        env = self.env
        for i, task in enumerate(env.scenario.tasks):
            buyer = task.buyer
            pos = buyer.state.p_pos
            pos = transform(pos, box, width, height, w_p=w_p, h_p=h_p)
            radius = int(buyer.size / 30 * (width - 2 * w_p))
            cv2.circle(base_img, pos, radius, (0, 255, 0), thickness=2)
            cv2.putText(
                base_img, str(i + 1),
                (pos[0] - 5, pos[1] + 5) if i+1 < 9 else (pos[0] - 10, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA
            )
        # Drones
        for agent in env.scenario.agents:
            pos = agent.state.p_pos
            pos = transform(pos, box, width, height, w_p=w_p, h_p=h_p)
            radius = int(agent.size / 30 * (width - 2 * w_p))
            cv2.circle(base_img, pos, radius, (255, 100, 0), thickness=-1)

            # last_pos = drone.last_pos
            # if last_pos is not None:
            #     last_pos = transform(last_pos, box, width, height, w_p=w_p, h_p=h_p)
            #     cv2.line(self.base_img, pos, last_pos, (0, 0, 255), thickness=1)
        # # Global Information
        # cv2.putText(
        #     base_img, env.global_info(),
        #     (40, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5, (0, 0, 0), 1, cv2.LINE_AA
        # )

        self.video.write(base_img)
        if show:
            cv2.imshow('base image', base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()

    def draw_pos(self, poses, radius=0.5, show=False):
        width, height, w_p, h_p = self.width, self.height, self.w_p, self.h_p

        r = int(radius / 30 * (width - 2 * w_p))
        for pos in poses:
            pos = transform(pos, self.box, width, height, w_p=w_p, h_p=h_p)
            cv2.circle(self.base_img, pos, r, (0, 0, 0), thickness=-1)

        if show:
            cv2.imshow('base image', self.base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()

    def draw_line(self, poses, show=False):
        width, height, w_p, h_p = self.width, self.height, self.w_p, self.h_p

        for i, pos1 in enumerate(poses[1:]):
            pos1 = transform(pos1, self.box, width, height, w_p=w_p, h_p=h_p)
            pos2 = transform(poses[i], self.box, width, height, w_p=w_p, h_p=h_p)
            cv2.line(self.base_img, pos1, pos2, (255, 255, 0), thickness=1)

        if show:
            cv2.imshow('base image', self.base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None
