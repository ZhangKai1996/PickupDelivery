import copy

import cv2
import numpy as np


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


class CVRender:
    def __init__(self, env):
        self.width, self.height = env.scenario.size
        self.range = env.scenario.range_p
        self.env = env
        self.video = cv2.VideoWriter(
            'trained/pickup_delivery.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            8, (self.width, self.height)
        )
        self.__init_img(env, self.height, self.width)

    def __init_img(self, env, height, width):
        base_image = np.ones((height, width, 3), np.uint8) * 255
        self.base_img = base_image
        cv2.imwrite('trained/base_image.png', base_image)

    def transform(self, pos, w_p=0, h_p=0):
        """
        align the coordinate system of rendering with that of scenario.
        """
        min_x, max_x = min_y, max_y = self.range
        _width = self.width - 2 * w_p
        _height = self.height - 2 * h_p
        return (w_p + int((pos[0] - min_x) / (max_x - min_x) * _width),
                h_p + int((pos[1] - min_y) / (max_y - min_y) * _height))

    def draw(self, show=False):
        base_img = copy.deepcopy(self.base_img)

        env = self.env
        for i, task in enumerate(env.scenario.tasks):
            # Merchant
            merchant = task.merchant
            pos = self.transform(pos=merchant.state.p_pos)
            radius = int(merchant.size / 3 * self.width)
            thickness = -1 if merchant.occupied else 2
            cv2.circle(base_img, pos, radius, merchant.color, thickness=thickness)
            cv2.putText(
                base_img, str(i),
                (pos[0] - 5, pos[1] + 5) if i < 9 else (pos[0] - 10, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1,
                cv2.LINE_AA
            )
            # Buyer
            buyer = task.buyer
            pos = self.transform(pos=buyer.state.p_pos)
            radius = int(buyer.size / 3 * self.width)
            thickness = -1 if buyer.occupied else 2
            cv2.circle(base_img, pos, radius, buyer.color, thickness=thickness)
            cv2.putText(
                base_img, str(i),
                (pos[0] - 5, pos[1] + 5) if i < 9 else (pos[0] - 10, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA
            )
        # Drones
        for agent in env.scenario.agents:
            pos = self.transform(pos=agent.state.p_pos)
            radius = int(agent.size / 3 * self.width)
            cv2.circle(base_img, pos, radius, agent.color, thickness=-1)
            info = [task.name for task in agent.tasks]
            if len(info) > 0:
                cv2.putText(
                    base_img, ','.join(info),
                    (pos[0]-radius+10, pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA
                )

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
            cv2.waitKey(10)
            # cv2.destroyAllWindows()
            # if cv2.waitKey(0) == 113:
            #     cv2.destroyAllWindows()

    def draw_pos(self, poses, radius=0.5, show=False):
        r = int(radius / 30 * self.width)
        for pos in poses:
            pos = self.transform(pos)
            cv2.circle(self.base_img, pos, r, (0, 0, 0), thickness=-1)

        if show:
            cv2.imshow('base image', self.base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()

    def draw_line(self, poses, show=False):
        for i, pos1 in enumerate(poses[1:]):
            pos1 = self.transform(pos1)
            pos2 = self.transform(poses[i])
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
