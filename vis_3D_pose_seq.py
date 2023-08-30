#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

import json

#Nose = 0,
#Neck = 1,
#RShoulder = 2,
#RElbow = 3,
#RWrist = 4,
#LShoulder = 5,
#LElbow = 6,
#LWrist = 7,
#MidHip = 8,
#RHip = 9,
#RKnee = 10,
#RAnkle = 11,
#LHip = 12,
#LKnee = 13,
#LAnkle = 14,
#REye = 15,
#LEye = 16,
#REar = 17,
#LEar = 18,
##Head = 19,  #unused
##Belly = 20, #unused
##LBToe = 21, #unused
##LSToe = 22, #unused
##LHeel = 23, #unused
##RBToe = 24, #unused
##RSToe = 25, #unused
##RHeel = 26, #unused
##NUM_KEYPOINTS = 27;

NUM_KPS_USED = 19
KPS_PARENT=[-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16]

CocoColors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
              (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (50, 0, 255), (100, 0, 255),
              (170, 0, 255), (255, 0, 255), (255, 150, 0), (85, 170, 0), (42, 128, 85), (0, 85, 170),
              (255, 0, 170), (255, 0, 85), (242, 165, 65)]

if __name__ == '__main__':
    in_file_name = sys.argv[1]
    
    with open(in_file_name, 'r') as f:
        pose_data = json.load(f)
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_aspect("equal")
    ax.set_box_aspect([1,1,1])

    all_xs = []
    all_ys = []
    all_zs = []
    
    for frame in pose_data:
        kps = frame['person']['keypoints']
        
        for kp_idx in range(NUM_KPS_USED):
            kp = kps[kp_idx]
            if kp['score'] == 0:
                continue
            
            all_xs.append(kps[kp_idx]['pos'][0])
            all_ys.append(kps[kp_idx]['pos'][1])
            all_zs.append(kps[kp_idx]['pos'][2])
            
    min_x = min(all_xs)
    max_x = max(all_xs)
    min_y = min(all_ys)
    max_y = max(all_ys)
    min_z = min(all_zs)
    max_z = max(all_zs)
    
    print('x: [{}, {}], y: [{}, {}], z: [{}, {}],'.format(min_x, max_x, min_y, max_y, min_z, max_z))
    
    ax.set_box_aspect([(max_x - min_x), (max_y - min_y), (max_z - min_z) ])
    
    assert len(KPS_PARENT) == NUM_KPS_USED
        
    idx = 0    
    for frame in pose_data:
        sys.stdout.write('at frame %05d.\r' % idx)
        sys.stdout.flush()
                
        kps = frame['person']['keypoints']
        
        xs = []
        ys = []
        zs = []
        colors = []
        parent_idx = []
        
        for kp_idx in range(NUM_KPS_USED):
            kp = kps[kp_idx]
            if kp['score'] == 0:
                continue
            
            xs.append(kps[kp_idx]['pos'][0])
            ys.append(kps[kp_idx]['pos'][1])
            zs.append(kps[kp_idx]['pos'][2])
            colors.append((CocoColors[kp_idx][0] / 255., CocoColors[kp_idx][1] / 255., CocoColors[kp_idx][2] / 255.))
        
        ax.cla()
        ax.scatter(xs, ys, zs, c=colors, s=3)
        
        if(len(xs) == NUM_KPS_USED): # only works when all kps are present
            for kp_idx in range(1,len(xs)):
                lxs = [xs[KPS_PARENT[kp_idx]], xs[kp_idx]]
                lys = [ys[KPS_PARENT[kp_idx]], ys[kp_idx]]
                lzs = [zs[KPS_PARENT[kp_idx]], zs[kp_idx]]
                ax.plot(lxs, lys, lzs, color='green', linewidth=1)
                
        ax.set_xlim(min_x-0.5, max_x+0.5)
        ax.set_ylim(min_y-0.5, max_y+0.5)
        ax.set_zlim(min_z-0.1, max_z+0.1)
        plt.draw()
        
        #fname_frame = 'frames/frame_{:05d}.png'.format(idx)
        idx += 1
        #plt.savefig(fname_frame, dpi=600, bbox_inches='tight')
        
        plt.pause(0.001)
    
    print('done')
    plt.show()
            
            
