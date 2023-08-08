from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import *
from PyQt6.QtGui import *
from PyQt6 import uic
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np

from blackboard import Config, Data

class Skeleton(QObject):
    def __init__(self,  skeleton_structure,
                         joint_names,
                          color_scheme,
                           show_joints=True,
                           annotate_joints=True,
                           joint_radius=0.5,
                            show_bones=True,
                            bone_width=0.2,
                            bone_color=(255,0,0)):
        super().__init__()
        self.skeleton_structure = skeleton_structure
        self.joint_names = joint_names
        self.joint_positions = {}
        self.show_joints = show_joints
        self.annotate_joints = annotate_joints
        self.show_bones = show_bones
        self.joint_radius = joint_radius
        self.bone_width = bone_width
        self.bone_color = bone_color
        self.color_scheme = dict(zip(joint_names, color_scheme[:len(joint_names)]))
        for n in self.joint_names:
            self.joint_positions[n] = []

    def set_joint_positions(self, joint_positions):
        self.joint_positions = joint_positions

    def draw(self):
        if self.show_joints:
            self.draw_joints()
        if self.show_bones:
            self.draw_bones()

    def draw_joints(self):
        for joint_name, position in self.joint_positions.items():
            glColor3f(*self.color_scheme[joint_name])
            glPushMatrix()
            glTranslatef(*position)
            quadric = gluNewQuadric()
            gluSphere(quadric, self.joint_radius, 16, 16)
            glPopMatrix()
    
    def draw_bones(self):
        glBegin(GL_LINES)
        for id, (cur_joint, par_joint) in enumerate(self.skeleton_structure):
            glColor3f(*self.bone_color)
            glVertex3f(*self.joint_positions[cur_joint])
            glVertex3f(*self.joint_positions[par_joint])
        glEnd()
    
class PoseVisualizer(QOpenGLWidget):
    def __init__(self, 
                 parent=None):
        super().__init__(parent)

        self.data = Data()
        self.config = Config()

        self.current_data = None
        self.skeletons = {}

        self.font = QFont("Arial", 12)
        self.axis_limits = self.config.axis_limits
        self.origin = self.config.origin
        self.paint_frame = self.config.paint_coordinate_frame
        self.paint_grid = self.config.paint_grid
        self.num_ticks = self.config.num_ticks

        self.viewpoint = self.config.viewpoint
        self.camera_distance = self.config.camera_distance
        self.default_rotation = self.config.default_rotation

        # View interactions
        self.last_pos = None
        self.x_rotation = self.default_rotation[0]
        self.y_rotation = self.default_rotation[1]
        self.z_rotation = self.default_rotation[2]
        self.update_rate = 0.01

        return
    
    
    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0) 
        glEnable(GL_DEPTH_TEST)
    
    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def paintGL(self):
        # Perform OpenGL drawing
        if self.data is None:
            print("VISUALIZER: No data to draw")
            return
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set camera view
        glTranslatef(0.0, 0.0, -self.camera_distance)
        glRotatef(self.x_rotation, 1.0, 0.0, 0.0)
        glRotatef(self.y_rotation, 0.0, 1.0, 0.0)
        glRotatef(self.z_rotation, 0.0, 0.0, 1.0)

        self.draw_coordinate_frame()

        if len(self.skeletons.keys()) == 0:
            return
        for skeleton_name, skeleton in self.skeletons.items():
            skeleton.draw()
    
    def initialize_skeleton(self, skeleton_name):
        self.skeletons[skeleton_name] = Skeleton(   skeleton_structure = self.data.skeleton_structure[skeleton_name],
                                                    joint_names = self.data.joint_names[skeleton_name],
                                                    color_scheme = self.config.color_scheme,
                                                    show_joints = self.config.show_joints,
                                                    annotate_joints = self.config.annotate_joints,
                                                    joint_radius = self.config.joint_dot_radius,
                                                    show_bones = self.config.show_bones,
                                                    bone_width = self.config.bone_width,
                                                    bone_color = self.config.bone_color)
        return True
        


    def draw_coordinate_frame(self):
        """
            Function to draw the coordinate frame including a grids that span the x-y, y-x, x-z plane respectively.
            Depending on the config the grid can be omitted. The origin and axis limits can be defined through the config.
        """

        x_ticks = np.linspace(self.origin[0], self.axis_limits[0], 10)
        y_ticks = np.linspace(self.origin[1], self.axis_limits[1], 10)
        z_ticks = np.linspace(self.origin[2], self.axis_limits[2], 10)


        # Draw grid planes in the background
        if self.paint_grid:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glLineWidth(1.0)
            glColor4f(0.0,0.0,0.0,0.5)

            glBegin(GL_LINES)
            # x-y plane
            for i in x_ticks:
                glVertex3f(i, self.origin[1], self.origin[2])
                glVertex3f(i, self.origin[1]+self.axis_limits[1], self.origin[2])
            for i in y_ticks:
                glVertex3f(self.origin[0], i, self.origin[2])
                glVertex3f(self.origin[0]+self.axis_limits[0], i, self.origin[2])
            # y-z plane
            for i in y_ticks:
                glVertex3f(self.origin[0], i, self.origin[2])
                glVertex3f(self.origin[0], i, self.origin[2]+self.axis_limits[2])
            for i in z_ticks:
                glVertex3f(self.origin[0], self.origin[1], i)
                glVertex3f(self.origin[0], self.origin[1]+self.axis_limits[1], i)
            # x-z plane
            for i in x_ticks:
                glVertex3f(i, self.origin[1], self.origin[2])
                glVertex3f(i, self.origin[1], self.origin[2]+self.axis_limits[2])
            for i in z_ticks:
                glVertex3f(self.origin[0], self.origin[1], i)
                glVertex3f(self.origin[0]+self.axis_limits[0], self.origin[1], i)
            glEnd()
            glDisable(GL_BLEND)

        # Draw X, Y, and Z axes
        if self.paint_frame:
            glLineWidth(3.0)
            glBegin(GL_LINES)
            glColor3f(1.0, 0.0, 0.0)  # X-axis (red)
            glVertex3f(self.origin[0], self.origin[1], self.origin[2])
            glVertex3f(self.axis_limits[0], self.origin[1], self.origin[2])


            glColor3f(0.0, 1.0, 0.0)  # Y-axis (green)
            glVertex3f(self.origin[0], self.origin[1], self.origin[2])
            glVertex3f(self.origin[0], self.axis_limits[1], self.origin[2])
            
            glColor3f(0.0, 0.0, 1.0)  # Z-axis (blue)
            glVertex3f(self.origin[0], self.origin[1], self.origin[2])
            glVertex3f(self.origin[0], self.origin[1], self.axis_limits[2])
            glEnd()

            # X-axis ticks and labels
            glLineWidth(1.0)
            glColor3f(0.0, 0.0, 0.0)
            for i in x_ticks:
                glBegin(GL_LINES)
                glVertex3f(i, self.origin[1], self.origin[2])
                glVertex3f(i, self.origin[1] - 0.05, self.origin[2])
                glEnd()
                #self.draw_label(str(i), i, self.origin[1] - 0.07, self.origin[2])

            # Y-axis ticks and labels
            for i in y_ticks:
                glBegin(GL_LINES)
                glVertex3f(self.origin[0], i, self.origin[2])
                glVertex3f(self.origin[0] - 0.05, i, self.origin[2])
                glEnd()
                #self.draw_label(str(i), self.origin[0] - 0.07, i, self.origin[2])

            # Z-axis ticks and labels
            for i in z_ticks:
                glBegin(GL_LINES)
                glVertex3f(self.origin[0], self.origin[1], i)
                glVertex3f(self.origin[0] - 0.05, self.origin[1], i)
                glEnd()
                #self.draw_label(str(i), self.origin[0] - 0.07, self.origin[1], i)


    def draw_label(self, label, x, y, z):
        # Draw a label at the specified position (x, y, z) using the given font
        glRasterPos3f(x, y, z)
        for char in label:
            glutBitmapCharacter(self.font, ord(char))
    
    ##-- View Interactions --##
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            dx = event.position().x() - self.last_pos.x()
            dy = event.position().y() - self.last_pos.y()

            self.x_rotation += self.update_rate*dy
            self.x_rotation %= 360
            self.y_rotation += self.update_rate*dx
            self.y_rotation %= 360

            self.update()

            self.last_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = None

    
    ##-- Slots --##
    @pyqtSlot()
    def update_data(self):
        self.current_data = self.data.get_current_timeframe()
        for sk_name, data in self.current_data.items():
            if sk_name not in self.skeletons.keys():
                print("VISUALIZER: Skeleton does not exist")
                continue
            self.skeletons[sk_name].set_joint_positions(data)
        self.update()
            
    
    @pyqtSlot()
    def update_config(self):
        for sk_name in self.skeletons.keys():
            self.initialize_skeleton(sk_name)
        self.update()

    @pyqtSlot()
    def rotateX(self, angle):
        self.x_rotation = angle
        self.update()
    
    @pyqtSlot()
    def rotateY(self, angle):
        self.y_rotation = angle
        self.update()
    
    @pyqtSlot()
    def rotateZ(self, angle):
        self.z_rotation = angle
        self.update()