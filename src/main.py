import sys
from PyQt5.QtWidgets import QApplication
from src.fundus_segmentation_tool import FundusSegmentationTool

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FundusSegmentationTool()
    ex.show()
    sys.exit(app.exec_())