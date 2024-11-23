import copy
import time

import cv2
import numpy as np

from env.utils import coord2state


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


entity_radius = 8
subtitle_size = 0.4
subtitle_color = (0, 0, 0)
font_thickness = 1
outer_font_size = 0.4
outer_font_color = (0, 0, 0)
inner_font_size = 0.4
inner_font_color = (150, 150, 150)

text_kwargs1 = dict(
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=outer_font_size,
    color=outer_font_color,
    thickness=font_thickness,
    lineType=cv2.LINE_AA
)
text_kwargs2 = dict(
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=subtitle_size,
    color=subtitle_color,
    thickness=font_thickness,
    lineType=cv2.LINE_AA
)


class EnvRender:
    def __init__(self, env, width=2525, height=2525, padding=0.01):
        self.env = env
        self.width = width
        self.height = height
        self.padding = padding

        self.video = cv2.VideoWriter(
            filename='trained/snake_{}.avi'.format(int(time.time())),
            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
            fps=8, frameSize=(width, height)
        )

        self.base_img = None
        self.pos_dict = None

    def initialize(self):
        self.base_img = np.ones((self.height, self.width, 3), np.uint8) * 255
        w_p = int(self.width * self.padding)
        h_p = int(self.height * self.padding)
        size = self.env.size
        border_len = int((self.width - w_p * 2) / size)

        self.pos_dict = {}
        for i in range(size):
            column = int(border_len / 2 + i * border_len + w_p)
            # Draw row number
            cv2.putText(self.base_img, str(i), (column, int(self.height - h_p / 2)), **text_kwargs1)

            for j in range(size):
                row = int(border_len / 2 + j * border_len + h_p)
                idx = i + j * size
                self.pos_dict[idx] = (column, row)
                # Draw grids
                cv2.rectangle(
                    img=self.base_img,
                    pt1=(int(column - border_len / 2), int(row - border_len / 2)),
                    pt2=(int(column + border_len / 2), int(row + border_len / 2)),
                    color=(0, 0, 0),
                    thickness=1
                )
                # Draw column number
                if i == 0:
                    cv2.putText(self.base_img, str(j), (int(w_p / 2), row), **text_kwargs1)

        self.__draw_agents(size, color=(255, 0, 0), traj=False)
        self.__draw_stones(size)
        self.__draw_orders(size)
        cv2.imwrite('trained/base_image.png', self.base_img)

    def __draw_agents(self, size, color=None, traj=True):
        if color is None: color = (0, 0, 255)

        for agent in self.env.agents:
            pos = coord2state(agent.state, size)
            pos = self.pos_dict[pos]
            if traj:
                cv2.circle(self.base_img, pos, int(entity_radius/2), color, thickness=-1)
                if agent.last_state is not None:
                    last_pos = coord2state(agent.last_state, size)
                    last_pos = self.pos_dict[last_pos]
                    cv2.line(self.base_img, last_pos, pos, color, thickness=1)
            else:
                cv2.circle(self.base_img, pos, entity_radius, color, thickness=-1)

    def __draw_stones(self, size, color=None):
        if color is None: color = (0, 0, 0)

        for stone in self.env.stones:
            pos = coord2state(stone.state, size)
            pos = self.pos_dict[pos]
            cv2.circle(self.base_img, pos, entity_radius, color, thickness=-1)

    def __draw_orders(self, size):
        for order in self.env.orders:
            # Buyer
            pos = coord2state(order.buyer.state, size)
            pos = self.pos_dict[pos]
            thickness = -1 if order.buyer.occupied is not None else 2
            cv2.circle(self.base_img, pos, entity_radius, (255, 0, 255), thickness=thickness)
            cv2.putText(self.base_img, order.buyer.name, pos, **text_kwargs2)
            # Merchant
            pos = coord2state(order.merchant.state, size)
            pos = self.pos_dict[pos]
            thickness = -1 if order.merchant.occupied is not None else 2
            cv2.circle(self.base_img, pos, entity_radius, (255, 255, 0), thickness=thickness)
            cv2.putText(self.base_img, order.merchant.name, pos, **text_kwargs2)

    def draw(self, mode=None, show=False):
        base_img = copy.deepcopy(self.base_img)

        # Text: mode (MC/TD/PI/VI)
        if mode is not None:
            pos = (int(self.width * self.padding), int(self.height * self.padding / 2))
            cv2.putText(base_img, mode, pos, **text_kwargs2)

        size = self.env.size
        self.__draw_agents(size)
        self.__draw_stones(size)
        self.__draw_orders(size)

        if show:
            cv2.imshow('basic image', base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()
        self.video.write(base_img)

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None


class CVRender:
    def __init__(self, env, width=1200, height=1200, side=400):
        self.env = env
        self.width = width
        self.height = height
        self.range = env.scenario.range_p

        self.video = cv2.VideoWriter(
            filename='trained/pickup_delivery.avi',
            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
            fps=30,
            frameSize=(self.width+side, self.height)
        )

        self.base_img = None
        self.make_base_img()
        self.side_bar = np.ones((self.height, side, 3), np.uint8) * 255
        cv2.rectangle(self.side_bar, (0, 0), (side, self.height), color=(0, 0, 0), thickness=1)
        cv2.imwrite('trained/base_image.png', self.base_img)

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
