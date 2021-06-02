import math
import numpy as np
import PyKDL as kdl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from DE import *


#meter 
link1 = 0.17282 * 100
link2 = 0.049194  * 100
link3 = 0.207937 * 100
link4 = 0.364028  * 100
    
link = [link1, link2, link3,link4]



def draw_axis(ax, scale=1.0, O=np.eye(4), style='-'):
    xaxis = np.array([[0, 0, 0, 1], [scale, 0, 0, 1]]).T
    yaxis = np.array([[0, 0, 0, 1], [0, scale, 0, 1]]).T
    zaxis = np.array([[0, 0, 0, 1], [0, 0, scale, 1]]).T
    xc = O.dot(xaxis)
    yc = O.dot(yaxis)
    zc = O.dot(zaxis) 
    ax.plot(xc[0,:], xc[1,:], xc[2,:], 'r' + style)
    ax.plot(yc[0,:], yc[1,:], yc[2,:], 'g' + style)
    ax.plot(zc[0,:], zc[1,:], zc[2,:], 'b' + style)
    
def draw_links(ax, origin_frame=np.eye(4), target_frame=np.eye(4)):
    x = [origin_frame[0,3], target_frame[0,3]]
    y = [origin_frame[1,3], target_frame[1,3]]
    z = [origin_frame[2,3], target_frame[2,3]]
    ax.plot(x, y, z, linewidth = 3)


def RX(yaw):
    return np.array([[1, 0, 0], 
                     [0, math.cos(yaw), -math.sin(yaw)], 
                     [0, math.sin(yaw), math.cos(yaw)]])   

def RY(delta):
    return np.array([[math.cos(delta), 0, math.sin(delta)], 
                     [0, 1, 0], 
                     [-math.sin(delta), 0, math.cos(delta)]])

def RZ(theta):
    return np.array([[math.cos(theta), -math.sin(theta), 0], 
                     [math.sin(theta), math.cos(theta), 0], 
                     [0, 0, 1]])

def TF(rot_axis=None, q=0, dx=0, dy=0, dz=0):
    if rot_axis == 'x':
        R = RX(q)
    elif rot_axis == 'y':
        R = RY(q)
    elif rot_axis == 'z':
        R = RZ(q)
    elif rot_axis == None:
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    
    T = np.array([[R[0,0], R[0,1], R[0,2], dx],
                  [R[1,0], R[1,1], R[1,2], dy],
                  [R[2,0], R[2,1], R[2,2], dz],
                  [0, 0, 0, 1]])
    return T
    
def FK(angle, link):
    base = TF()
    T0 = TF('y', q = angle[0], dy = link[0])
    T0_0 = base.dot(T0)
    T1 = TF('x', q = angle[1], dy = link[1])
    T1_0 = T0_0.dot(T1)
    
    T2 = TF('y', q = angle[2], dz = -link[2])
    T2_1 = T1_0.dot(T2)
    
    T3 = TF('z', q = angle[3], dz = -link[3])
    T3_2 = T2_1.dot(T3)

    
    return base, T0_0, T1_0, T2_1, T3_2
    
    
def obj_func (f_target, thetas, link):
    P1, P2, P3, P4, sole = FK(thetas,link)
    f_result = kdl.Frame(kdl.Rotation(sole[0,0], sole[0,1], sole[0,2],
                                      sole[1,0], sole[1,1], sole[1,2],
                                      sole[2,0], sole[2,1], sole[2,2]),
                         kdl.Vector(sole[0,3], sole[1,3], sole[2,3]))

    f_diff = f_target.Inverse() * f_result
    
    [dx, dy, dz] = f_diff.p
    [drz, dry, drx] = f_diff.M.GetEulerZYX()
    
    error = np.sqrt(dx**2 + dy**2 + dz**2 + drz**2) #pilih yaw aja
    
    return error, thetas
    

def cekError(f_target, sole):
    f_result = kdl.Frame(kdl.Rotation(sole[0,0], sole[0,1], sole[0,2],
                                      sole[1,0], sole[1,1], sole[1,2],
                                      sole[2,0], sole[2,1], sole[2,2]),
                         kdl.Vector(sole[0,3], sole[1,3], sole[2,3]))
    
    f_diff = f_target.Inverse() * f_result
    
    [dx, dy, dz] = f_diff.p
    [drz, dry, drx] = f_diff.M.GetEulerZYX()
    
    error = np.sqrt(dx**2 + dy**2 + dz**2 + drx**2 + dry**2 + drz**2)
    
    error_pos = np.sqrt(dx**2 + dy**2 + dz**2)
    error_rot = np.sqrt(drz**2)
    
    error_list = [dx, dy, dz, drx, dry, drz]
    
    return error, error_list, f_result, error_pos, error_rot

 
def main():
    global target, angle, link, yaw
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("Differential Evolution - Inverse Kinematics", fontsize=12)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    yaw = 30
    yaw = np.radians(yaw)
 
    target = [-0.24766811 * 100 ,  0.5079965 * 100 , -0.42897375 * 100]
    f_target = kdl.Frame(kdl.Rotation.RPY(0, 0, yaw), kdl.Vector(target[0], target[1], target[2]))
    
    #jumlah yg di inisialisasi
    n_params = 4
    
    #batas bawah dan atas 
    lb = [(-np.radians(60), -np.pi/2, 0 , -np.pi)]
    ub = [(np.pi, np.radians(45), (np.radians(160)) , np.pi)]
    
    angle = np.array([0,0,0,0])
    
    
    #inverse Kinematics
    error, angle = DE(obj_func, f_target, angle, link, n_params, lb, ub)
 
    if (error > 1): 
       print("IK Error")
    else:
       print("IK Solved")
    
    
    #forward Kinematics
    p0, base, p1, p2, p3 = FK(angle,link)
    
    err, err_list, f_r, err_p, err_r = cekError(f_target, p3)

    [drz, dry, drx] = f_target.M.GetEulerZYX()
    [drz2, dry2, drx2] = f_r.M.GetEulerZYX()
    
    

    angle = np.rad2deg(angle)
    
    angle = angle%(360)
    for i in range (n_params):
        if (angle[i] >= 180):
           angle[i] = angle[i] - 360
            
    print("error pos", err_p)
    print("error rot", err_r)
    print("""""""""""""""""")    
    print("yaw target", drz)
    print("yaw result", drz2)
    print("""""""""""""""""")    

   
    print("target", target)
    print("end effector", p3[:3,3])
 #   print("error list" , err_list)
 #   print("frame target", f_target)
    print("""""""""""""""""")    
  #  print("frame result", f_r)

    print("angle", angle)
    print("angle %.2f" % angle [0])
    print("angle %.2f" % angle [1])
    print("angle %.2f" % angle [2])
    print("angle %.2f" % angle [3])

   
    print("[(-60,180) , (-90, 45), (0,160), (-180, 180)")
#    print("batas bawah", lb)
#    print("batas atas", ub)
    draw_axis(ax, scale=0.05* 100 , O=p0)
    draw_links(ax, origin_frame=p0, target_frame=base)
    draw_axis(ax, scale=0.05* 100 , O=base)
    draw_links(ax, origin_frame=base, target_frame=p1)
    draw_axis(ax, scale=0.05* 100, O=p1)
    draw_links(ax, origin_frame=p1, target_frame=p2)
    draw_axis(ax, scale=0.05* 100, O=p2)
    draw_links(ax, origin_frame=p2, target_frame=p3)
    draw_axis(ax, scale=0.05* 100, O=p3)
    ax.scatter3D(target[0], target[1], target[2], color = "black", marker = "x")

    plt.show()
   
if __name__ == "__main__":
    main()
    
    
    
    
    

