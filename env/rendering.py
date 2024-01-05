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
            30, (self.width, self.height)
        )
        self.base_img = np.ones((self.height, self.width, 3), np.uint8) * 255
        # cv2.imwrite('trained/base_image.png', self.base_img)

    def transform(self, pos, w_p=0, h_p=0):
        """
        align the coordinate system of rendering with that of scenario.
        """
        min_x, max_x = min_y, max_y = self.range
        _width = self.width - 2 * w_p
        _height = self.height - 2 * h_p
        return (w_p + int((pos[0] - min_x) / (max_x - min_x) * _width),
                h_p + int((pos[1] - min_y) / (max_y - min_y) * _height))

    def _draw_task(self, tasks, base_img, delta):
        for i, task in enumerate(tasks):
            # Merchant
            merchant = task.merchant
            pos = self.transform(pos=merchant.state.p_pos)
            radius = int(merchant.size / delta * self.width)
            thickness = -1 if merchant.occupied else 2
            cv2.circle(base_img, pos, radius, merchant.color, thickness=thickness)
            cv2.putText(
                base_img, str(i),
                (pos[0] - 5, pos[1] + 5) if i < 10 else (pos[0] - 10, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1,
                cv2.LINE_AA
            )
            # Buyer
            buyer = task.buyer
            pos = self.transform(pos=buyer.state.p_pos)
            radius = int(buyer.size / delta * self.width)
            thickness = -1 if buyer.occupied else 2
            cv2.circle(base_img, pos, radius, buyer.color, thickness=thickness)
            cv2.putText(
                base_img, str(i),
                (pos[0] - 5, pos[1] + 5) if i < 10 else (pos[0] - 10, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA
            )

    def _draw_rider(self, agents, base_img, delta):
        for agent in agents:
            pos = self.transform(pos=agent.state.p_pos)
            radius = int(agent.size / delta * self.width)
            cv2.circle(base_img, pos, radius, agent.color, thickness=-1)
            info = [task.name for task in agent.tasks if not task.is_finished]
            if len(info) > 0:
                cv2.putText(
                    base_img, ','.join(info),
                    (pos[0]-radius+10, pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA
                )
            last_pos = agent.last_state
            if last_pos is not None:
                last_pos = self.transform(pos=agent.last_state.p_pos)
                cv2.line(self.base_img, pos, last_pos, (100, 100, 100), thickness=2)

    def _draw_barrier(self, barriers, base_img, delta):
        for i, barrier in enumerate(barriers):
            pos = self.transform(pos=barrier.state.p_pos)
            radius = int(barrier.size / delta * self.width)
            cv2.circle(base_img, pos, radius, barrier.color, thickness=-1)

    def draw(self, mode=None, clear=False, show=False):
        base_img = copy.deepcopy(self.base_img)
        # Global Information
        if mode is not None:
            cv2.putText(
                base_img, mode,
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
        delta = self.range[1] - self.range[0]
        scenario = self.env.scenario
        # Tasks (Merchants and Buyers)
        self._draw_task(scenario.tasks, base_img, delta)
        # Riders
        self._draw_rider(scenario.agents, base_img, delta)
        # Barriers
        self._draw_barrier(scenario.barriers, base_img, delta)
        # clear the objects of base image when another episode starts.
        if clear:
            self.base_img = np.ones((self.height, self.width, 3), np.uint8) * 255
        # take base image as a frame of video.
        self.video.write(base_img)
        if show:
            cv2.imshow('base image', base_img)
            cv2.waitKey(10)
            # cv2.destroyAllWindows()
            # if cv2.waitKey(0) == 113:
            #     cv2.destroyAllWindows()

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None
