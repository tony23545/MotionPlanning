import os
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt


def depth_to_xy(depth, pos, heading, fov):
    '''
    Convert depth values to x,y locations.
    :param depth: a numpy array of size N containing N depth rays
    :param pos: position of the sensor. A numpy array of size 2.
    :param heading: orientation of the sensor (in radians). A float.
    :param fov: field of view of the sensor.
    :return: the global x, y coordinates of depth points.
    '''
    n_bins = depth.shape[0]
    theta = np.linspace(-fov * 0.5, fov * 0.5, n_bins, endpoint=False)
    xs = np.cos(theta + heading)
    ys = np.sin(theta + heading)
    xy = np.stack([xs * depth, ys * depth], axis=1)
    if pos is None:
        return xy
    return pos + xy


def rotate_2d(v, angle):
    '''
    Rotate one or more points.
    :param v: a numpy array of size 2 or Nx2 containing one or more points.
    :angle: a float.
    :return: a numpy array of size 2 or Nx2 containing the rotated points.
    '''
    c = np.cos(angle)
    s = np.sin(angle)
    if isinstance(v, tuple) or len(v.shape) == 1:
        x, y = v
        x2 = c * x - s * y
        y2 = s * x + c * y
        return np.array([x2, y2], np.float32)
    elif len(v.shape) == 2:
        assert v.shape[1] == 2
        m = np.array([[c, -s], [s, c]])
        return np.dot(v, m.T)
    else:
        raise ValueError


class Map(object):
    def __init__(self, config_file, laser_max_range=10.0, downsample_factor=1):
        '''
        :config_file: a .yaml file containing meta data of the map.
        :laser_max_range: maximum range of the laser scanner (in meters).
        :downsample_factor: an integer for downsampling the occupancy map.
                            E.g., 2 means that the width and height will be divided by 2.
        '''
        path = os.path.abspath(config_file)
        self.cfg = yaml.load(open(config_file).read(), Loader=yaml.SafeLoader)

        img_file = os.path.join(os.path.dirname(path), self.cfg['image'])

        bitmap = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        free_space = (bitmap >= 250).astype(np.uint8)

        if downsample_factor > 1:
            free_space = cv2.resize(free_space, None, fx=1.0/downsample_factor, fy=1.0/downsample_factor)

        # 0 means no obstacle
        self.occupancy_grid = (1 - free_space) * 255

        # 255 means no obstacle. For visualization purpose.
        self.inv_occupancy_grid = (255 - self.occupancy_grid)

        # Pixel coordinate representing the origin of the occupancy grid.
        self.origin = np.array(self.cfg['origin']) / downsample_factor

        self.resolution = float(self.cfg['resolution']) * downsample_factor
        self.n_division = int(1.0 / self.resolution)  # Number of pixels / m
        # 1.0 / self.resolution should be an integer.
        assert self.n_division - 1.0 / self.resolution < 1, 'Bad downsampling factor'

        self.map_bbox = self._compute_map_bbox()
        self.area = self._compute_free_area()

        self.laser_max_range = laser_max_range

    def _compute_free_area(self):
        return np.sum((self.occupancy_grid == 0)) * self.resolution**2

    def _compute_map_bbox(self):
        # Compute visible map bbox for visualization purposes
        nz_indices = np.transpose(np.nonzero(self.occupancy_grid == 0))
        # Note that axis 0 is y and axis 1 is x
        # bbox is (x_min, x_max, y_min, y_max)
        grid_x_min, grid_x_max, grid_y_min, grid_y_max = (
            np.min(nz_indices[:, 1]), np.max(nz_indices[:, 1]),
            np.min(nz_indices[:, 0]), np.max(nz_indices[:, 0]))

        x_min, y_min = self.world_coord(grid_x_min, grid_y_min)
        x_max, y_max = self.world_coord(grid_x_max, grid_y_max)

        return x_min, x_max, y_min, y_max

    def grid_coord(self, x, y):
        '''
        :return: the grid coordinates where (x, y) falls
        '''
        return int((x - self.origin[0]) * self.n_division), int((y - self.origin[1]) * self.n_division)

    def grid_coord_batch(self, xys):
        """
        Batch version of grid_coord()
        :param xys: N x 2 np array
        """
        return ((xys - np.array(self.origin)) * self.n_division).astype(np.int32)

    def world_coord(self, x, y):
        '''
        :return: the world coordinates of grid coord (x, y).
        '''
        return float(x) / self.n_division + self.origin[0], float(y) / self.n_division + self.origin[1]

    def get_1d_depth(self, pos, heading, fov, n_depth_ray, resolution=None):
        '''
        :param pos: a numpy array of size 2 representing the global location of the robot (in meters).
        :param heading: heading of the robot (in radians). A float.
        :param fov: the field of view of the laser scanner (in radians). A float.
        :param n_depth_ray: number of depth rays. An integer.
        :param resolution: Resolution when estimating depths. Higher value will decrease the depth accuracy but
                           increase speed. If None will use the map resolution.
        :return: a numpy array of size `n_depth_ray` representing measured depths (in meters).
        '''
        thetas = np.linspace(-fov * 0.5, fov * 0.5, n_depth_ray, endpoint=False)
        ray_dir_x = np.cos(thetas + heading)
        ray_dir_y = np.sin(thetas + heading)
        ray_dirs = np.stack([ray_dir_x, ray_dir_y], axis=-1)

        if resolution is None:
            resolution = self.resolution

        n_steps = int(self.laser_max_range / resolution)

        # (n_steps, n_ray, 2)
        ray_endpoints = pos.reshape((1, 1, 2)) + \
                        ray_dirs.reshape(1, n_depth_ray, 2) * resolution * \
                        np.arange(n_steps).reshape(n_steps, 1, 1)
        ray_grid_coords = self.grid_coord_batch(ray_endpoints.reshape(-1, 2))

        # Note that ray_grid_coords contains x, y coordinates, whereas occupancy_grid.shape contains (height, width)
        np.clip(ray_grid_coords[:, 0], 0, self.occupancy_grid.shape[1]-1, out=ray_grid_coords[:, 0])
        np.clip(ray_grid_coords[:, 1], 0, self.occupancy_grid.shape[0]-1, out=ray_grid_coords[:, 1])

        values = self.occupancy_grid[ray_grid_coords[:, 1], ray_grid_coords[:, 0]].reshape(n_steps, n_depth_ray)
        hits = values != 0
        hits[-1, :] = 1
        first_nonzero_idx = np.argmax(hits, axis=0)
        depth = first_nonzero_idx * resolution

        return depth

    def local_to_global(self, pos, heading, xys):
        '''
        Transform local points into the global coordinate system
        :param pos: a numpy array of size 2
        :param heading: a floating point number in radians
        :param xys: a numpy array of size N x 2 containing N points.
        '''
        return rotate_2d(xys, heading) + pos


class Visualizer:
    def __init__(self, map, ax):
        self.map = map
        self.ax = ax

    def draw_map(self):
        m = self.map
        ax = self.ax

        ax.grid(False)
        ax.set_aspect('equal', 'datalim')

        left = m.origin[0]
        bottom = m.origin[1]
        grid = m.inv_occupancy_grid
        right = left + grid.shape[1] * m.resolution
        top = bottom + grid.shape[0] * m.resolution

        # stack() returns a new copy so code below won't modify m.occupancy_grid
        bitmap = np.stack([grid, grid, grid], axis=2)
        ax.imshow(bitmap, origin='lower', extent=(left, right, bottom, top))

        x_min, x_max, y_min, y_max = m.map_bbox
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

    def draw_obstacles(self, xys, *args, **kwargs):
        '''
        xys: a numpy array of size Nx2 representing locations of obstacles in the global coordinate system.
        '''
        self.ax.plot(xys[:, 0], xys[:, 1], '+', *args, **kwargs)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    m = Map('map/map.yaml', laser_max_range=4, downsample_factor=1)

    vis = Visualizer(m, ax)
    vis.draw_map()

    pos = np.array((3, 10))
    heading = np.deg2rad(90)
    n_ray = 100
    fov = np.deg2rad(240.0)

    depth = m.get_1d_depth(pos, heading, fov, n_ray, resolution=0.01)
    obstacles = depth_to_xy(depth, pos, heading, fov)

    vis.draw_obstacles(obstacles, markeredgewidth=1.5)
    plt.show()
