import copy

import cv2
import numpy as np


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


class CVRender:
    def __init__(self, env, width=1200, height=1200, padding=0.05):
        self.video = cv2.VideoWriter(
            'figs/pickup_delivery.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            8,
            (width+400, height)
        )

        self.width = width
        self.height = height
        self.padding = padding
        self.env = env

        self.__initialize(env, height, width, padding)

    def __draw_legends(self, base_image, width, height):
        cv2.circle(base_image, (100, height-30), 8, (255, 0, 0), thickness=2)
        cv2.putText(
            base_image,
            'Merchants',
            (120, height-26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        cv2.circle(base_image, (240, height-30), 8, (255, 100, 0), thickness=-1)
        cv2.putText(
            base_image,
            'Drones',
            (260, height-26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        cv2.circle(base_image, (360, height-30), 8, (0, 255, 0), thickness=2)
        cv2.putText(
            base_image,
            'Buyers',
            (380, height-26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        cv2.rectangle(
            base_image,
            (462, height-22),
            (478, height-38),
            (0, 0, 0),
            thickness=-1
        )
        cv2.putText(
            base_image,
            'Obstacles',
            (490, height-26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    def __initialize(self, env, height, width, padding):
        base_image = np.ones((height, width+400, 3), np.uint8) * 255
        w_p = int(width * padding)
        h_p = int(height * padding)
        size = env.size
        border_len = int((width - w_p * 2) / size)

        self.pos_dict = {}
        for i in range(size):
            column = int(border_len / 2 + i * border_len + w_p)
            # Draw row state
            cv2.putText(
                base_image, str(i), (column, int(h_p/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

            for j in range(size):
                row = int(border_len / 2 + j * border_len + h_p)
                pos = (column, row)
                # idx = i + j * size
                idx = (j, i)
                self.pos_dict[idx] = pos

                # Draw grids
                thickness = 1 if idx not in env.walls else -1
                cv2.rectangle(
                    base_image,
                    (int(pos[0] - border_len / 2), int(pos[1] - border_len / 2)),
                    (int(pos[0] + border_len / 2), int(pos[1] + border_len / 2)),
                    (0, 0, 0),
                    thickness=thickness
                )

                # Draw merchants
                if idx in env.merchants:
                    cv2.circle(base_image, pos, 12, (255, 0, 0), thickness=2)
                    m_id = env.merchants.index(idx) + 1
                    new_pos = (pos[0]-5, pos[1]+5) if m_id < 10 else (pos[0]-10, pos[1]+5)
                    cv2.putText(
                        base_image,
                        str(m_id),
                        new_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1,
                        cv2.LINE_AA
                    )

                # Draw column state
                if i == 0:
                    cv2.putText(
                        base_image, str(j), (int(h_p/2), row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )

        self.__draw_legends(base_image, width, height)
        self.base_img = base_image
        cv2.imwrite('figs/base_image.png', base_image)

    def draw(self, show=False):
        width, height = self.width, self.height
        base_img = copy.deepcopy(self.base_img)

        # Buyers
        env = self.env
        merchants = env.merchants
        for buyer in env.buyers:
            pos = self.pos_dict[buyer.address]
            cv2.circle(base_img, pos, 12, (0, 255, 0), thickness=2)

            for merchant in buyer.get_merchants():
                m_id = merchants.index(merchant) + 1
                new_pos = (pos[0] - 5, pos[1] + 5) if m_id < 10 else (pos[0] - 10, pos[1] + 5)
                cv2.putText(
                    base_img,
                    str(m_id),
                    new_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1,
                    cv2.LINE_AA
                )
        # Drones
        for drone in env.drones:
            pos = self.pos_dict[drone.position]
            color = (255, 100, 0) if not drone.is_collision else (255, 0, 255)
            cv2.circle(base_img, pos, 10, color, thickness=-1)
            last_pos = drone.last_pos
            if last_pos is not None:
                cv2.line(self.base_img, pos, self.pos_dict[last_pos], (0, 0, 255), thickness=1)
        # Global Information
        cv2.putText(
                base_img,
                env.global_info(),
                (width, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

        self.video.write(base_img)
        if show:
            cv2.imshow('base image', base_img)
            if cv2.waitKey(0) == 113:
                cv2.destroyAllWindows()

    def draw_visited(self, visited=None, size=1):
        if visited is None:
            return

        for visit in visited:
            width, height, padding = self.width, self.height, self.padding
            w_p = int(width * padding)
            h_p = int(height * padding)
            border_len = int((width - w_p * 2) / size)

            color = make_random_color()
            for pos in visit:
                pos = self.pos_dict[pos]
                cv2.rectangle(
                    self.base_img,
                    (int(pos[0] - border_len / 2), int(pos[1] - border_len / 2)),
                    (int(pos[0] + border_len / 2), int(pos[1] + border_len / 2)),
                    color,
                    thickness=1
                )

    def draw_dynamic(self, poses):
        for pos in poses:
            pos = self.pos_dict[pos]
            cv2.circle(self.base_img, pos, 10, (0, 0, 0), thickness=-1)

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None
