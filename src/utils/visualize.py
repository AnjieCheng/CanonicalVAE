import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
import torch, scipy
import json
import _thread as thread
import visdom
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
import matplotlib.animation as animation
import umap
import seaborn as sns

from collections import defaultdict
import seaborn as sns
import trimesh
from src.utils.render_mitsuba2_pc import *

def tsboard_log_scalar(logger, scalars, it, prefrix='train'):
    for k, v in scalars.items():
        # logger.add_scalar('%s/%s' % (prefrix, k), v, it)
        logger.log_metrics({'%s/%s' % (prefrix, k): v}, it)
        
def print_current_scalars(epoch, i, scalars):
    message = '(epoch: %d, iters: %d) ' % (epoch, i)
    for k, v in scalars.items():
        message += '%s: %.3f ' % (k, v)
    print(message)

def visualize_pointcloud_stp(points, stp_points=None, out_file=None, show=False, set_limit=False, set_view=True):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)

    ax.scatter(points[:, 2], points[:, 0], points[:, 1], c='grey')
    ax.scatter(stp_points[:, 2], stp_points[:, 0], stp_points[:, 1], c='red', s=80)

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    if set_limit:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)

    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        # plt.axis('off')
        plt.grid(b=None)
        plt.savefig(out_file, dpi=120, transparent=True)

    plt.close(fig)

def visualize_pointcloud(points, color=None, out_file=None, show=False, set_limit=False, set_view=True):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    
    viridis  = sns.color_palette("gist_rainbow", as_cmap=True) # cm.get_cmap('nipy_spectral') # gist_rainbow
    correspondance = np.linspace(0, 1, points.shape[0])

    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)

    if color is not None:
        ax.scatter(points[:, 2], points[:, 0], points[:, 1], c=color/255.)
    else:   
        ax.scatter(points[:, 2], points[:, 0], points[:, 1], c=correspondance, cmap=viridis)

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    if set_limit:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)

    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        plt.grid(b=None)
        plt.savefig(out_file, dpi=120, transparent=True)

    plt.close(fig)


def visualize_pointcloud_mitsuba2(points, out_file=None):
    PATH_TO_MITSUBA2 = "/home/vslab2018/3d/mitsuba2/build/dist/mitsuba"  # mitsuba exectuable
    pcl = points

    pcl = standardize_bbox(pcl, 2048)
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    xml_segments = [xml_head]
    for i in range(pcl.shape[0]):
        color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    xmlFile = ("%s_.xml" % (out_file))

    with open(xmlFile, 'w') as f:
        f.write(xml_content)
    f.close()

    exrFile = ("%s_.exr" % (out_file))
    if (not os.path.exists(exrFile)):
        print(['Running Mitsuba, writing to: ', xmlFile])
        subprocess.run([PATH_TO_MITSUBA2, xmlFile])
    else:
        print('skipping rendering because the EXR file already exists')

    print(['Converting EXR to JPG...'])
    # ConvertEXRToJPG(exrFile, out_file)
    convert_exr_to_jpg(exrFile, ("%s.jpg" % (out_file)))

    print(['Remove XML'])
    os.remove(xmlFile)
    print(['Remove EXR'])
    os.remove(exrFile)


def visualize_embedding_with_umap(weight, out_file=None):
    # Create plot
    fig = plt.figure()

    proj = umap.UMAP(n_neighbors=3,
                     min_dist=0.1,
                     metric='cosine').fit_transform(weight)
    
    plt.scatter(proj[:,0], proj[:,1], alpha=0.3)

    if out_file is not None:
        # plt.axis('off')
        plt.grid(b=None)
        plt.savefig(out_file, dpi=120, transparent=False)

    plt.close(fig)

class Visualizer(object):
    def __init__(self, port, env):
        super(Visualizer, self).__init__()
        thread.start_new_thread(os.system, (f"visdom -p {port} > /dev/null 2>&1",))
        vis = visdom.Visdom(port=port, env=env)
        self.vis = vis

    def show_pointclouds(self, points, title=None, Y=None):
        points = points.squeeze()
        if points.size(-1) == 3:
            points = points.contiguous().data.cpu()
        else:
            points = points.transpose(0, 1).contiguous().data.cpu()

        if Y is None:
            self.vis.scatter(X=points, win=title, opts=dict(title=title, markersize=2))
        else:
            if Y.min() < 1:
                Y = Y - Y.min() + 1
            self.vis.scatter(
                X=points, Y=Y, win=title, opts=dict(title=title, markersize=2)
            )

    def show_sphere_groups(self, points, title=None, Y=None, groups=None, num_groups=None):
        groups = groups.detach().cpu().numpy()
        COLORS = sns.color_palette("Spectral", as_cmap=True)
        g_value = groups/num_groups
        color = np.round(COLORS(g_value)* 255.0)[:,:3]

        points = points.squeeze()
        if points.size(-1) == 3:
            points = points.contiguous().data.cpu()
        else:
            points = points.transpose(0, 1).contiguous().data.cpu()

        if Y is None:
            self.vis.scatter(X=points, win=title, opts=dict(title=title, markersize=6, markercolor=color))
        else:
            if Y.min() < 1:
                Y = Y - Y.min() + 1
            self.vis.scatter(
                X=points, Y=Y, win=title, opts=dict(title=title, markersize=2, markercolor=color)
            )

    def show_histogram(self, groups=None, title=None, num_groups=None):
        self.vis.histogram(X=groups, win=title, opts=dict(numbins=num_groups))

    def show_heatmap(self, prob, title=None):
        self.vis.heatmap(X=prob, win=title)