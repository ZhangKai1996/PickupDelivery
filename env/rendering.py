import copy

import cv2
import numpy as np


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


class CVRender:
    def __init__(self, env, width=1200, height=1200, side=400):
        self.env = env
        self.width = width
        self.height = height
        self.range = env.scenario.range_p
        self.video = cv2.VideoWriter(
            'trained/pickup_delivery.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            30, (self.width+side, self.height)
        )

        self.base_img = None
        self.make_base_img()
        self.side_bar = np.ones((self.height, side, 3), np.uint8) * 255
        cv2.rectangle(self.side_bar, (0, 0), (side, self.height), color=(0, 0, 0), thickness=1)
        # cv2.imwrite('trained/base_image.png', self.base_img)

    def make_base_img(self):
        self.base_img = np.ones((self.height, self.width, 3), np.uint8) * 255
        cv2.rectangle(self.base_img, (0, 0), (self.width, self.height), color=(0, 0, 0), thickness=1)

    def transform(self, pos, w_p=0, h_p=0):
        """
        align the coordinate system of rendering with that of scenario.
        """
        min_x, max_x = min_y, max_y = self.range
        _width = self.width - 2 * w_p
        _height = self.height - 2 * h_p
        return (w_p + int((pos[0] - min_x) / (max_x - min_x) * _width),
                h_p + int((pos[1] - min_y) / (max_y - min_y) * _height))

    def __draw_person(self, entity, base_img, delta, i, text_args):
        pos = self.transform(pos=entity.state.p_pos)
        radius = int(entity.size / delta * self.width)
        thickness = -1 if entity.occupied is not None else 2
        cv2.circle(base_img, pos, radius, entity.color, thickness=thickness)
        text_args[2] = entity.color
        cv2.putText(base_img, str(i), (pos[0]-5, pos[1]+4), *text_args)

    def __draw_stone(self, entity, base_img, delta, i, text_args):
        pos = self.transform(pos=entity.state.p_pos)
        radius = int(entity.size / delta * self.width)
        cv2.circle(base_img, pos, radius, entity.color, thickness=-1)
        text_args[2] = entity.color
        cv2.putText(base_img, str(i), (pos[0]-5, pos[1]+4), *text_args)

    def __draw_agent(self, agent, base_img, side_bar, delta, text_pos, text_args):
        pos = self.transform(pos=agent.state.p_pos)
        radius = int(agent.size / delta * self.width)
        cv2.circle(base_img, pos, radius, agent.color, thickness=-1)
        cv2.putText(base_img, agent.name, pos, *text_args)

        cv2.putText(side_bar, agent.status(), text_pos, *text_args)
        for i, task in enumerate(agent.tasks):
            text_pos[1] += 20
            cv2.putText(side_bar, task.status(), text_pos, *text_args)
        text_pos[1] += 30
        last_pos = agent.last_state
        if last_pos is not None:
            last_pos = self.transform(pos=agent.last_state.p_pos)
            cv2.line(self.base_img, pos, last_pos, (100, 100, 100), thickness=2)
        return text_pos

    def draw(self, mode=None, clear=False, show=False):
        delta = self.range[1] - self.range[0]
        scenario = self.env.scenario

        base_img = copy.deepcopy(self.base_img)
        side_bar = copy.deepcopy(self.side_bar)
        text_args = [cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA]

        # Global Information
        if mode is not None:
            cv2.putText(side_bar, mode, (20, 40), *text_args)
        # Tasks
        for i, task in enumerate(scenario.tasks):
            self.__draw_person(task.merchant, base_img, delta, i, text_args[:])
            self.__draw_person(task.buyer, base_img, delta, i, text_args[:])
        # Drones
        text_pos = [20, 70]
        for i, agent in enumerate(scenario.agents):
            seq_lst = scenario.sequence[i]
            for j, pos in enumerate(seq_lst[:-1]):
                pos1 = self.transform(pos=pos)
                pos2 = self.transform(pos=seq_lst[j+1])
                cv2.line(base_img, pos1, pos2, (0, 255, 0), thickness=1)

            text_pos = self.__draw_agent(
                agent, base_img, side_bar,
                delta, text_pos, text_args
            )
        for i, barrier in enumerate(scenario.barriers):
            self.__draw_stone(barrier, base_img, delta, i, text_args[:])
        # Clear the objects of base image when another episode starts.
        if clear:
            self.make_base_img()

        base_img = np.hstack([base_img, side_bar])
        self.video.write(base_img)
        if show:
            cv2.imshow('base image', base_img)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
            # if cv2.waitKey(0) == 113:
            #     cv2.destroyAllWindows()

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None
