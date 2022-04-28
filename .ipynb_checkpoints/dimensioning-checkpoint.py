import ifm3dpy
from ifm3dpy import * #O3RCamera, FrameGrabber, ImageBuffer
import json
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import pandas as pd
import numpy as np
import copy
import time
import tensorflow as tf
import pickle
#from open3d import JVisualizer
import segmentation_models as sm
#from PIL import Image
from scipy.interpolate import RectBivariateSpline
print(o3d)
print(o3d.core.cuda.is_available())

sm.set_framework('tf.keras')

sm.framework()


class HandlingUnitExtractor():
    
    def __init__(self):
        model_file = 'best_model_handling_unit.h5'
        self.model_img_size = (512, 512)
        self.model = sm.Unet('efficientnetb3', classes=1, activation='sigmoid')
        self.model.load_weights(model_file) 
        self.preprocessor = sm.get_preprocessing('efficientnetb3')
        
    def extract(self, img):
        h, w = img.shape[:2]
        image = cv2.resize(img, self.model_img_size, interpolation = cv2.INTER_AREA)
        image = self.preprocessor(image)
        
        #t = time.time()
        #mask = self.model.predict(np.expand_dims(image, axis=0)).round().squeeze()
        mask = self.model(np.expand_dims(image, axis=0)).numpy()
        #print(mask)
        mask = mask.round().squeeze()
        #print(time.time() -t)
        mask = cv2.resize(mask, (w, h), interpolation = cv2.INTER_AREA)
        
        mask = self.isolate_largest_component(mask)
        
        return mask.astype(np.uint8)

    def isolate_largest_component(self, img):
        result = np.zeros((img.shape))
        labels, stats = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)[1:3]                   
        
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result[labels == largest_label] = 255 

        return result

def turnoffO3R(o3r):
    config = o3r.get()
    config['ports']['port0']['state'] = 'IDLE'
    config['ports']['port2']['state'] = 'IDLE'
    o3r.set(config)
    
def turnonO3R(o3r):
    config = o3r.get()
    config['ports']['port0']['state'] = 'RUN'
    config['ports']['port2']['state'] = 'RUN'
    o3r.set(config)

def RectificationalizationColor(image, intrinsics):    

    fx, fy, mx, my, alpha, k1, k2, k5, k3, k4, *_ = intrinsics

    dmap = image
    
    KKNoAlpha = np.array([[fx, 0.0, mx],
                          [0,  fy,  my],
                          [0,  0,   1]])

    # Generate the pixel coordinates
    px,py = np.meshgrid(np.arange(0, dmap.shape[1]),
                        np.arange(0, dmap.shape[0]))

    # Homogeneous coordinates
    coords = np.hstack([px.reshape(-1,1) + 0.5, py.reshape(-1,1) + 0.5, np.ones([px.size, 1])]).T

    # Transfer into a normalized coordinate frame
    # (f=1, origin is the center, Dim: -1..1)
    coords_norm = np.linalg.solve(KKNoAlpha, coords)
    
    # Apply distortion
    R2 = np.square(coords_norm[0]) + np.square(coords_norm[1])
    R4 = np.square(R2)
    R6 = np.power(R2,3)
    radial_dist = 1 + k1*R2 + k2*R4 + k5*R6

    a1 = 2*coords_norm[0]*coords_norm[1]
    a2 = R2 + 2*np.square(coords_norm[0])
    a3 = R2 + 2*np.square(coords_norm[1])
    tangential_dist = np.vstack([k3*a1 + k4*a2, k3*a3 + k4*a1])

    coord_dist = np.vstack([np.ones([2,1])*radial_dist*coords_norm[0:2] + tangential_dist, np.ones([1,px.size])])

    # Transformer --> Pixel
    KK = KKNoAlpha
    KK[0,1] = fx*alpha
    #KK[0,1] = alpha
    coord_dist = KK.dot(coord_dist)
    
    
    fr = RectBivariateSpline(np.arange(0, dmap.shape[0]),
                        np.arange(0, dmap.shape[1]),
                        dmap[:,:,0])
    fg = RectBivariateSpline(np.arange(0, dmap.shape[0]),
                        np.arange(0, dmap.shape[1]),
                        dmap[:,:,1])
    fb = RectBivariateSpline(np.arange(0, dmap.shape[0]),
                        np.arange(0, dmap.shape[1]),
                        dmap[:,:,2])

    rectifiedr = fr(coord_dist[1]-0.5, coord_dist[0]-0.5, grid=False) /255
    rectifiedg = fg(coord_dist[1]-0.5, coord_dist[0]-0.5, grid=False) /255
    rectifiedb =  fb(coord_dist[1]-0.5, coord_dist[0]-0.5, grid=False)/255

    


    amp_rectifiedr = rectifiedr.reshape(dmap.shape[:2])
    amp_rectifiedg = rectifiedg.reshape(dmap.shape[:2])
    amp_rectifiedb = rectifiedb.reshape(dmap.shape[:2])

    
    img = np.stack([amp_rectifiedr,amp_rectifiedg,amp_rectifiedb], axis=2)
    #img = img*255
    #print(img)
    #print(img.shape)
    #plt.imshow(img)
    return img
    
def Rectificationalization(image, intrinsics):
    
    amp = image
    # Unpack intrinsics values
    fx, fy, mx, my, alpha, k1, k2, k5, k3, k4, *_ = intrinsics

    # Create the camera matrix
    # If the current application is using the 23k imager, fx/fy/mx/my must be divided by two.
    # If the current application is using the full resolution imager, fx/fy/mx/my can be used 'as-is'
    
    
    KKNoAlpha = np.array([[fx, 0.0, mx],
                              [0,  fy,  my],
                              [0,  0,   1]])

    # Generate the pixel coordinates
    px,py = np.meshgrid(np.arange(0, amp.shape[1]),
                        np.arange(0, amp.shape[0]))

    # Homogeneous coordinates
    coords = np.hstack([px.reshape(-1,1) + 0.5, py.reshape(-1,1) + 0.5, np.ones([px.size, 1])]).T

    # Transfer into a normalized coordinate frame
    # (f=1, origin is the center, Dim: -1..1)
    coords_norm = np.linalg.solve(KKNoAlpha, coords)
    
    # Apply distortion
    R2 = np.square(coords_norm[0]) + np.square(coords_norm[1])
    R4 = np.square(R2)
    R6 = np.power(R2,3)
    radial_dist = 1 + k1*R2 + k2*R4 + k5*R6

    a1 = 2*coords_norm[0]*coords_norm[1]
    a2 = R2 + 2*np.square(coords_norm[0])
    a3 = R2 + 2*np.square(coords_norm[1])
    tangential_dist = np.vstack([k3*a1 + k4*a2, k3*a3 + k4*a1])

    coord_dist = np.vstack([np.ones([2,1])*radial_dist*coords_norm[0:2] + tangential_dist,
                       np.ones([1,px.size])])

    # Transformer --> Pixel
    KK = KKNoAlpha
    KK[0,1] = fx*alpha
    #KK[0,1] = alpha
    coord_dist = KK.dot(coord_dist)
    
    f = RectBivariateSpline(np.arange(0, amp.shape[0]),
                        np.arange(0, amp.shape[1]),
                        amp)
    rectified = f(coord_dist[1]-0.5, coord_dist[0]-0.5, grid=False)
    amp_rectified = rectified.reshape(amp.shape)
    return amp_rectified

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 8))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')) + f' {image.shape} {image.shape[1] / image.shape[0]:.2f}'.title())
        plt.imshow(image)
    plt.show()

def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    #pcd_down = pcd_down.to_legacy()
    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    #print(":: Apply fast global registration with distance threshold %.3f" \ % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    #distance_threshold = voxel_size * 0.4
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return result

def refine_registrationP2P(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return result

def undistort(img):
    roi = (125, 102, 860, 515)
    #roi = (0, 0, 1041, 799)
    
    # mtx = np.array([[607.43997952,   0.        , 645.96889836],
    #             [  0.        , 605.58954233, 391.50563407],
    #             [  0.        ,   0.        ,   1.        ]]
                   
    
    
    mtx = np.array([[607.43997952,   0.        , 645.96889836],
                    [  0.        , 605.58954233, 391.50563407],
                    [  0.        ,   0.        ,   1.        ]])

    dist = np.array([[-0.35962477,  0.16171162, -0.00081826, -0.00444003, -0.03848117]])

    #dist = np.array([[-0.35427392,  0.28329617, -0.00391964, -0.00128499, -0.16376514]])    
    newcameramtx = np.array([[408.22506714,   0.        , 558.75454709],
                          [  0.        , 390.17321777, 354.4384903 ],
                          [  0.        ,   0.        ,   1.        ]])
                   
    # newcameramtx = np.array([[572.56311035,   0.,         520.36453768],
    #                      [  0.,         568.80871582, 394.30206237],
    #                      [  0.,          0.,           1.        ]])

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    d = dst.copy()
    dst = dst[y:y+h, x:x+w]
    return dst

def loadpickle(filename = 'bag'):
    with open(filename, "rb") as fp:
            frames = pickle.load(fp)
    return frames


    

            

# rgb = cv2.imdecode(frames[0][1], cv2.IMREAD_UNCHANGED) 
# rgb = undistort(rgb)
# rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  
# rgb = rgb[:, 95:-95]
# rgb = cv2.resize(rgb, (224,172), interpolation = cv2.INTER_AREA)
# plt.imshow(rgb)


    
    
def load_point_clouds(frames = None, numframes=200, picklefile= 'bag2'):    
       
        
    
    pcds = []
    maskpcds = []
    hue = HandlingUnitExtractor()
    for frame in frames:    
        rgb = frame[1]
        xyz = frame[0]  
        
        rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        #    pic = pic[:, 119:-119]
        # pic = cv2.resize(pic, (224,172), interpolation = cv2.INTER_AREA)
        # print(pic.shape)
        # pic = pic/255
        
        
       
        
        
#         rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
#         rgb = undistort(rgb)         
#         rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  
#         rgb = rgb[:, 95:-95]
        
        #mask = hue.extract(rgb)
        
        rgb = rgb[:, 119:-119]
        rgb = cv2.resize(rgb, (224,172), interpolation = cv2.INTER_AREA)
        
        #mask = mask[:, 119:-119]        
        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) 
        #mask[:,:,1] = 0   
        #mask = cv2.resize(mask, (224,172), interpolation = cv2.INTER_AREA)
       
        #rgb = undistort(rgb)
        
        rgb = rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))
        rgb = rgb/255
        
        #mask = mask.reshape((mask.shape[0] * mask.shape[1], 3))
        #mask = mask/255
        
        xyz = xyz.reshape((xyz.shape[0] * xyz.shape[1], 3))     
        
        
        pcl = o3d.geometry.PointCloud()        
        pcl.points = o3d.utility.Vector3dVector(xyz)    
        pcl.colors = o3d.utility.Vector3dVector(rgb)         
        pcds.append(pcl)    
        
        # maskpcl = o3d.geometry.PointCloud()        
        # maskpcl.points = o3d.utility.Vector3dVector(xyz)    
        # maskpcl.colors = o3d.utility.Vector3dVector(mask)         
        # maskpcds.append(maskpcl)    
        
    
    return  pcds, maskpcds

def pairwise_registration(source, target):    
    voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])

    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=20),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 15),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 10)
    ]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])

    # Initial alignment or source to target transform.
    init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

    mu, sigma = 0, .1  # mean and standard deviation
    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    #estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    #estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane(o3d.t.pipelines.registration.robust_kernel.RobustKernel(
    #        o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss, sigma))
    #estimation = o3d.t.pipelines.registration.TransformationEstimationForGeneralizedICP()
    
    estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()#o3d.t.pipelines.registration.robust_kernel.RobustKernel(
    #        o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss, sigma))

    #Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    save_loss_log = True
    
    s = time.time()
    

    registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(source, target,
                                           voxel_sizes, criteria_list,
                                           max_correspondence_distances, estimation_method = estimation)
    
    transformation_icp = registration_ms_icp.transformation
    information_icp = o3d.t.pipelines.registration.get_information_matrix(source, target, max_correspondence_distances[2], registration_ms_icp.transformation)
    
    ms_icp_time = time.time() - s
    #print("Time taken by Multi-Scale ICP: ", ms_icp_time)
    #print("Inlier Fitness: ", registration_ms_icp.fitness)
    #print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)
    
    return transformation_icp, information_icp


def full_registration(pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)

    
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id])
            
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp.numpy(), odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp.numpy(),
                                                             information_icp.numpy(),
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp.numpy(),
                                                             information_icp.numpy(),
                                                             uncertain=True))
    return pose_graph



def plot_rmse(registration_result):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
    axes.set_title("Inlier RMSE vs Iteration")
    axes.plot(registration_result.loss_log["index"].numpy(),
              registration_result.loss_log["inlier_rmse"].numpy())


def plot_scale_wise_rmse(registration_result):
    scales = registration_result.loss_log["scale"].numpy()
    iterations = registration_result.loss_log["iteration"].numpy()

    num_scales = scales[-1][0] + 1

    fig, axes = plt.subplots(nrows=1, ncols=num_scales, figsize=(20, 5))

    masks = {}
    
    for scale in range(0, num_scales):
        masks[scale] = registration_result.loss_log["scale"] == scale

        rmse = registration_result.loss_log["inlier_rmse"][masks[scale]].numpy()
        iteration = registration_result.loss_log["iteration"][
            masks[scale]].numpy()

        title_prefix = "Scale Index: " + str(scale)
        axes[scale].set_title(title_prefix + " Inlier RMSE vs Iteration")
        axes[scale].plot(iteration, rmse)
        
def collectframes(picklefile = 'label', numframes=300, savebag = False):    
    frames = []
    for i in range(numframes):
        success3d = fg3d.wait_for_frame(im3d)
        success2d = fg2d.wait_for_frame(im2d)
        if not success2d:
            print('2d failed!')
        if not success3d:
            print('3d failed!')
        frames.append((im3d.xyz_image() , im2d.jpeg_image(), im3d.distance_image()))   
        #time.sleep(.1)
    
    if savebag:
        with open(picklefile, "wb") as fp:
            pickle.dump(frames, fp)
    
    return frames

def updatevis(vis, geo):
    vis.update_geometry(geo)
    vis.poll_events()
    vis.update_renderer()

if __name__ == "__main__":
    s = time.time()
    frames = loadpickle('bag4')
    #frames = collectframes('bag4', 600, True)
    print('DONE Collecting ' + str(time.time() - s))

    s = time.time()
    pcds, maskpcds = load_point_clouds(frames)
    print(time.time() - s) 
    #o3d.visualization.draw_geometries(pcds)

    voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
    #voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025, 0.012])
    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        #o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=30),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=30),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 15),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 100)        
    ]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    max_correspondence_distances = o3d.utility.DoubleVector([ 0.1, 0.05, 0.025])

    # Initial alignment or source to target transform.
    #init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

    mu, sigma = 0, 1  # mean and standard deviation
    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    #estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    #estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
    #estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP(o3d.t.pipelines.registration.robust_kernel.RobustKernel(
    #            o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss, sigma))
    #estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane(o3d.t.pipelines.registration.robust_kernel.RobustKernel(
    #        o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss, sigma))
    newpcd = []
    cnt = 1
    checkevery = 600
    for i in range(0,500):     
        pcd = pcds[i].voxel_down_sample(voxel_size=0.01)

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.055, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model

        if cnt % checkevery == 0:
            o3d.visualization.draw_geometries([pcd])

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        pcd = pcd.select_by_index(inliers, invert=True)
        #pcd, o = outlier_cloud.remove_radius_outlier(5, .1,True)

        if cnt % checkevery == 0:
            o3d.visualization.draw_geometries([pcd])

        #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.15, min_points=200))   


        unique, counts = np.unique(labels[labels > -1], return_counts=True)


        idxs = np.where(labels == unique[np.argmax(counts)])[0]

        cpcd = pcd.select_by_index(idxs)

        cpcd, o = cpcd.remove_radius_outlier(10, .1, True)

        #print(labels)
        max_label = labels.max()
        #print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        if cnt % checkevery == 0:
            o3d.visualization.draw_geometries([cpcd])
        if cnt % checkevery == 0:    
            o3d.visualization.draw_geometries([pcd])
        pcd = cpcd

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        newpcd.append(pcd)
        cnt = cnt + 1

    pcds = newpcd 
    tottime = time.time()
    voxel_size=0.1
    master = o3d.t.geometry.PointCloud().from_legacy(pcds[0])
    #hotpink =  o3d.t.geometry.PointCloud().from_legacy(maskpcds[0])
    #master.estimate_normals(max_nn=30, radius=.05)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(master.to_legacy())
    #updatevis(vis, master.to_legacy())
           
    for i in range(1, 50):    
        s = time.time()
        
        source_cuda = o3d.t.geometry.PointCloud().from_legacy(pcds[i-1])
        #source_cuda = source_cuda.cuda(0) 
        
        target_cuda = o3d.t.geometry.PointCloud().from_legacy(pcds[i])
        #target_cuda = target_cuda.cuda(0)  
        
        registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(source_cuda, target_cuda,
                                               voxel_sizes, criteria_list,
                                               max_correspondence_distances, estimation_method = estimation)#, init_source_to_target = result_ransac.transformation)#, save_loss_log=False)#,
                                               #result_ransac.transformation, estimation,
                                               #save_loss_log)
    
        source_cuda = source_cuda.cpu()
        target_cuda = target_cuda.cpu()
        ms_icp_time = time.time() - s
        
        
        if registration_ms_icp.fitness < .99:
            s = source_cuda.to_legacy()
            s.paint_uniform_color([1.0, 0, 0])
            t = target_cuda.to_legacy()
            t.paint_uniform_color([0, 0, 1.0])
            updatevis(vis, t)
            
        
        print("Time taken by Multi-Scale ICP: ", ms_icp_time)
        print("Inlier Fitness: ", registration_ms_icp.fitness)
        print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)
        
        master = master.transform(registration_ms_icp.transformation) + target_cuda
        #master = master.voxel_down_sample(voxel_size=0.01)
        #hotpink = hotpink.transform(registration_ms_icp.transformation) + o3d.t.geometry.PointCloud().from_legacy(maskpcds[i]) 
        #time.sleep(10)
            
        
    print(time.time() - tottime)
    master = master.to_legacy()
    #hotpink = hotpink.to_legacy()
    #o3d.visualization.draw_geometries([master])
    #o3d.visualization.draw_geometries([hotpink])