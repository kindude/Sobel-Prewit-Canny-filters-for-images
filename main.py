import os
import sys
import cv2
import numpy as np
from skimage import io
from skimage import feature
from skimage.metrics import structural_similarity as ssim

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import *


class MainWindow(QtWidgets.QMainWindow):
    G_height = 3
    G_width = 3
    sigma1 = 1
    sigma2 = 1
    image_ = []

    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), "UI\\main_4.ui"), self)
        self.comboBox.addItems(["Квадрат", "Крест", "Эллипс"])
        self.set_icon()
        self.menu_ = {}
        self.form = None
        self.kernel1 = None
        self.getItems()
        self.kernel_f_cnt = None
        self.connect()
        self.filename = 'media/photos/8f72bdb90d3c35736c80b032d7c6bc61.png'
        self.set_image()

    def getItems(self):
        self.menu_ = {'upload': self.down_2,
            'save': {
                'RGB': self.RGB_3,
                'Grey': self.Grey_3,
                'GreyNormalize': self.GreyNormalize_3,
                'Contour': self.Contour_3,
                'Canni': self.Canni_3,
                'Prevvit':  self.Prevvit_3,
                'Sobel':  self.Sobel_3
            },
            'Cont': self.Contour,
        }

    def connect(self):
        self.menu_['upload'].triggered.connect(self.open_file)
        for item in self.menu_['save'].values():
            item.triggered.connect(self.save_image)
        self.menu_['Cont'].triggered.connect(self.contour_params)
        self.pushButton.clicked.connect(self.OnBtnClick)
        self.comboBox.activated[str].connect(self.onActivated)

    def save_image(self, action):
        sender = self.sender()
        name = str(sender.objectName().strip('_3') + '_2')
        label = self.findChild(QtWidgets.QLabel, name)
        label.pixmap().save('media/savedImages/' + name.strip('_3') + '.png')

    def OnBtnClick(self):
        self.get_data()
        self.gradient = cv2.morphologyEx(self.norm_grey_image, cv2.MORPH_GRADIENT, self.kernel_f_cnt)
        self.Contour_2.setPixmap(convert(self.gradient))
        self.Console()

    def onActivated(self, text):
        self.form = text

    def get_data(self):
        if self.form == "Квадрат":
            self.kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
        if self.form == "Эллипс":
            self.kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
        if self.form == "Крест":
            self.kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                                     ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
        self.kernel_f_cnt = self.kernel1


    def open_file(self):
        self.filename = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', os.path.join(os.path.dirname(__file__), 'media\\photos'))[0])
        if self.filename:
            self.set_image()

    def set_icon(self):
        appIcon = QIcon('media/icons/icons8-пустой-фильтр-50.png')
        self.setWindowIcon(appIcon)

    def set_image(self):
        self.image_ = cv2.imread(self.filename)
        self.RGB_2.setPixmap(convert(self.image_))
        self.grey = cv2.cvtColor(self.image_, cv2.COLOR_RGB2GRAY)
        self.Grey_2.setPixmap(convert(self.grey))
        self.norm_grey_image = cv2.normalize(self.grey, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.GreyNormalize_2.setPixmap(convert(self.norm_grey_image))
        self.edges = cv2.Canny(self.grey, 25, 255, L2gradient=False)
        self.Canni.setPixmap(convert(self.edges))
        self.PrevvitImg = self.Prewitt(self.grey)
        self.Prevvit.setPixmap(convert(self.PrevvitImg))
        self.kernel_f_cnt = np.ones((5, 5), 'uint8')
        self.gradient = cv2.morphologyEx(self.norm_grey_image, cv2.MORPH_GRADIENT, self.kernel_f_cnt)
        self.Contour_2.setPixmap(convert(self.gradient))
        self.SobelImg = self.fSobel(self.grey)
        self.Sobel.setPixmap(convert(self.SobelImg))
        self.Console()

    def Console(self):
        print('##################################################################')
        ssim_none = ssim(self.gradient, self.edges, data_range=self.edges.max() - self.edges.min())
        print(f'Сходство исходного контура и Кэнни:{round(ssim_none * 100, 3)}%')
        ssim_none = ssim(self.gradient, self.PrevvitImg, data_range=self.PrevvitImg.max() - self.PrevvitImg.min())
        print(f'Сходство исходного контура и Преввита:{round(ssim_none * 100, 3)}%')
        ssim_none = ssim(self.gradient, self.SobelImg, data_range=self.SobelImg.max() - self.SobelImg.min())
        print(f'Сходство исходного контура и Собеля:{round(ssim_none * 100, 3)}%')
        print('##################################################################')

    def fSobel(self, grayImage):
        # Оператор Собеля
        x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  # Найдем первую производную x
        y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)  # Найдем первую производную y
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return Sobel

    def Prewitt(self, grayImage):
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
        # Turn uint8
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return Prewitt


    def updateMorphology(self, event):
        if self.Name == 'Erode':
            self.th4 = cv2.erode(self.th3, self.kernel, cv2.BORDER_REFLECT, iterations=self.iterations)
        elif self.Name == 'Dilate':
            self.th4 = cv2.dilate(self.th3, self.kernel, cv2.BORDER_REFLECT, iterations=self.iterations)
        elif self.Name == 'Opening':
            self.th4 = cv2.morphologyEx(self.th3, cv2.MORPH_OPEN, self.kernel)
        elif self.Name == 'Closing':
            self.th4 = cv2.morphologyEx(self.th3, cv2.MORPH_CLOSE, self.kernel)
        self.GreyMorph_2.setPixmap(convert(self.th4))
        gradient = cv2.morphologyEx(self.th4, cv2.MORPH_GRADIENT, self.kernel)
        self.Contour_2.setPixmap(convert(gradient))

    def updateCnt(self, event):
        gradient = cv2.morphologyEx(self.th4, cv2.MORPH_GRADIENT, self.kernel_f_cnt)
        self.Contour_2.setPixmap(convert(gradient))

    def second_window(self, action):
        pass
        # sender = self.sender()
        # self.child_window = ChildWindow(sender.objectName(), self)
        # self.child_window.show()
        # self.child_window.setFocus()
        # self.child_window.closeEvent = self.updateMorphology

    def contour_params(self):
        pass
        # sender = self.sender()
        # self.child_window = ChildWindow(sender.objectName(), self)
        # self.child_window.show()
        # self.child_window.closeEvent = self.updateCnt


def convert(image):
    im_resize = cv2.resize(image, (500, 500))
    is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
    qp = QPixmap()
    qp.loadFromData(im_buf_arr)
    return qp


def main():

    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.showMaximized()
    app.exec_()


if __name__ == '__main__':
    main()