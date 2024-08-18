import os
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QComboBox, QSlider, QColorDialog,
                             QListWidget, QInputDialog, QScrollArea, QListWidgetItem, QCheckBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QCursor, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect
import cv2
import numpy as np
from utils.image_processing import apply_image_adjustments

class FundusSegmentationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.undo_stack = []
        self.redo_stack = []
        self.max_stack_size = 20  # Adjust as needed

    def initUI(self):
        self.setWindowTitle('Fundus Segmentation Tool')
        self.setGeometry(100, 100, 1200, 700)

        pen_icon = QIcon(r'C:\Users\Betin\Desktop\fundus_segmentation\pen.png')
        eraser_icon = QIcon(r'C:\Users\Betin\Desktop\fundus_segmentation\eraser.png')

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Header section
        header_layout = QVBoxLayout()
        self.file_name_label = QLabel('No Image Loaded')
        self.file_name_label.setAlignment(Qt.AlignCenter)  # Center align the file name label
        header_layout.addWidget(self.file_name_label)

        # Image display
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.scroll_area.setWidget(self.image_label)
        header_layout.addWidget(self.scroll_area, 3)

        main_layout.addLayout(header_layout, 3)

        # Controls layout
        controls_layout = QVBoxLayout()
        
        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_button)

        self.clear_button = QPushButton('Clear Segmentation')
        self.clear_button.clicked.connect(self.clear_segmentation)
        controls_layout.addWidget(self.clear_button)

        self.save_button = QPushButton('Save Segmentation')
        self.save_button.clicked.connect(self.save_segmentation)
        controls_layout.addWidget(self.save_button)

        # Tool selection
        self.tool_combo = QComboBox()
        self.tool_combo.addItem(pen_icon, 'Draw')
        self.tool_combo.addItem(eraser_icon, 'Erase')
        controls_layout.addWidget(QLabel('Tool:'))
        controls_layout.addWidget(self.tool_combo)

        # Brush shape selection
        self.brush_shape_combo = QComboBox()
        self.brush_shape_combo.addItems(['Circle', 'Square'])
        controls_layout.addWidget(QLabel('Brush Shape:'))
        controls_layout.addWidget(self.brush_shape_combo)

        # Brush size slider
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 100)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.setTracking(True)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size_label)
        controls_layout.addWidget(QLabel('Brush Size:'))
        controls_layout.addWidget(self.brush_size_slider)
        self.brush_size_label = QLabel('5')
        controls_layout.addWidget(self.brush_size_label)

        # Opacity slider
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_display)
        controls_layout.addWidget(QLabel('Opacity:'))
        controls_layout.addWidget(self.opacity_slider)

        # Segmentation layers
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.select_layer)
        controls_layout.addWidget(QLabel('Segmentation Layers:'))
        controls_layout.addWidget(self.layer_list)

        self.add_layer_button = QPushButton('Add Layer')
        self.add_layer_button.clicked.connect(self.add_layer)
        controls_layout.addWidget(self.add_layer_button)

        self.delete_layer_button = QPushButton('Delete Layer')
        self.delete_layer_button.clicked.connect(self.delete_layer)
        controls_layout.addWidget(self.delete_layer_button)

        self.undo_button = QPushButton('Undo')
        self.undo_button.clicked.connect(self.undo)
        controls_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton('Redo')
        self.redo_button.clicked.connect(self.redo)
        controls_layout.addWidget(self.redo_button)

        # Add image adjustment sliders
        self.brightness_slider = self.create_adjustment_slider(controls_layout, "Brightness", -100, 100, 0)
        self.contrast_slider = self.create_adjustment_slider(controls_layout, "Contrast", -100, 100, 0)
        self.saturation_slider = self.create_adjustment_slider(controls_layout, "Saturation", -100, 100, 0)
        self.hue_slider = self.create_adjustment_slider(controls_layout, "Hue", -180, 180, 0)
        self.sharpness_slider = self.create_adjustment_slider(controls_layout, "Sharpness", 0, 10, 0)

        self.reset_settings_button = QPushButton('Reset Settings')
        self.reset_settings_button.clicked.connect(self.reset_settings)
        controls_layout.addWidget(self.reset_settings_button)

        main_layout.addLayout(controls_layout, 1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Image and segmentation data
        self.image = None
        self.segmentations = []
        self.current_layer = 0
        self.last_point = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.image_name = ""
        self.base_directory = ""

    def update_brush_size_label(self):
        size = self.get_brush_size()
        self.brush_size_label.setText(str(size))

    def get_brush_size(self):
        # Convert slider value to a logarithmic scale
        min_size = 1
        max_size = 50
        slider_value = self.brush_size_slider.value()
        log_value = np.log(slider_value)
        log_min = np.log(1)
        log_max = np.log(100)
        size = int(np.exp(log_min + (log_value - log_min) * (np.log(max_size) - np.log(min_size)) / (log_max - log_min)))
        return max(1, size)

    def reset_settings(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(0)
        self.hue_slider.setValue(0)
        self.sharpness_slider.setValue(0)
        self.opacity_slider.setValue(50)
        self.update_display()

    def create_adjustment_slider(self, layout, name, min_val, max_val, default_val):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(self.update_display)
        layout.addWidget(QLabel(f'{name}:'))
        layout.addWidget(slider)
        return slider

    def load_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
            if file_name:
                self.image = cv2.imread(file_name)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image_name = os.path.splitext(os.path.basename(file_name))[0]
                self.file_name_label.setText(f"Loaded File: {self.image_name}")
                self.segmentations = []
                self.layer_list.clear()
                self.add_layer()
                self.original_image = self.image.copy()  # Store the original image
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def add_layer(self):
        try:
            if self.image is not None:
                color = QColorDialog.getColor()
                if color.isValid():
                    name, ok = QInputDialog.getText(self, "Layer Name", "Enter layer name:")
                    if ok and name:
                        self.segmentations.append({
                            'mask': np.zeros(self.image.shape[:2], dtype=np.uint8),
                            'color': color.getRgb()[:3],
                            'name': name,
                            'visible': True
                        })
                        item = QListWidgetItem(name)
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Checked)
                        self.layer_list.addItem(item)
                        self.current_layer = len(self.segmentations) - 1
                        self.layer_list.setCurrentRow(self.current_layer)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add layer: {str(e)}")

    def delete_layer(self):
        try:
            if self.segmentations:
                current_row = self.layer_list.currentRow()
                if current_row != -1:
                    del self.segmentations[current_row]
                    self.layer_list.takeItem(current_row)
                    if self.segmentations:
                        self.current_layer = min(current_row, len(self.segmentations) - 1)
                        self.layer_list.setCurrentRow(self.current_layer)
                    else:
                        self.current_layer = -1
                    self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete layer: {str(e)}")

    def select_layer(self, item):
        try:
            self.current_layer = self.layer_list.row(item)
            if item.checkState() == Qt.Checked:
                self.segmentations[self.current_layer]['visible'] = True
            else:
                self.segmentations[self.current_layer]['visible'] = False
            self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to select layer: {str(e)}")

    def update_display(self):
        try:
            if self.image is not None:
                # Apply image adjustments
                adjusted_image = self.apply_image_adjustments(self.original_image)

                h, w, _ = adjusted_image.shape
                bytes_per_line = 3 * w
                overlay = adjusted_image.copy()

                for segmentation in self.segmentations:
                    if segmentation['visible']:
                        mask = segmentation['mask']
                        color = segmentation['color']
                        overlay[mask > 0] = color

                opacity = self.opacity_slider.value() / 100.0
                display = cv2.addWeighted(adjusted_image, 1 - opacity, overlay, opacity, 0)

                q_image = QImage(display.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)

                self.scale_factor = min(self.image_label.width() / w, self.image_label.height() / h)
                self.offset = QPoint(
                    (self.image_label.width() - scaled_pixmap.width()) // 2,
                    (self.image_label.height() - scaled_pixmap.height()) // 2
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update display: {str(e)}")

    def apply_image_adjustments(self, image):
        return apply_image_adjustments(image, 
                                       self.brightness_slider.value(),
                                       self.contrast_slider.value(),
                                       self.saturation_slider.value(),
                                       self.hue_slider.value(),
                                       self.sharpness_slider.value())

    def mousePressEvent(self, event):
        try:
            if self.image is not None and event.button() == Qt.LeftButton:
                pos = self.image_label.mapFrom(self, event.pos())
                self.last_point = self.scale_point(pos)
                self.draw_segmentation(self.last_point)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process mouse press event: {str(e)}")

    def mouseMoveEvent(self, event):
        try:
            if self.image is not None and event.buttons() & Qt.LeftButton:
                pos = self.image_label.mapFrom(self, event.pos())
                scaled_pos = self.scale_point(pos)
                self.draw_segmentation(scaled_pos)
                self.last_point = scaled_pos
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process mouse move event: {str(e)}")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = None

    def scale_point(self, point):
        return QPoint(
            int((point.x() - self.offset.x()) / self.scale_factor),
            int((point.y() - self.offset.y()) / self.scale_factor)
        )

    def draw_segmentation(self, pos):
        try:
            if self.image is not None and self.segmentations and self.current_layer != -1:
                # Save the current state for undo
                self.save_state()

                brush_size = self.get_brush_size()
                brush_shape = self.brush_shape_combo.currentText()
                mask = self.segmentations[self.current_layer]['mask'].copy()

                if self.tool_combo.currentText() == 'Draw':
                    value = 255
                else:
                    value = 0

                if self.last_point:
                    # Interpolate between last_point and current pos
                    points = self.interpolate_points(self.last_point, pos)
                    for point in points:
                        self.draw_brush(mask, point, brush_size, brush_shape, value)
                else:
                    # Single point
                    self.draw_brush(mask, pos, brush_size, brush_shape, value)

                self.segmentations[self.current_layer]['mask'] = mask
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to draw segmentation: {str(e)}")

    def draw_brush(self, mask, point, brush_size, brush_shape, value):
        if brush_shape == 'Circle':
            cv2.circle(mask, (point.x(), point.y()), brush_size, value, -1)
        else:  # Square
            cv2.rectangle(mask, 
                        (point.x() - brush_size, point.y() - brush_size),
                        (point.x() + brush_size, point.y() + brush_size), 
                        value, -1)

    def interpolate_points(self, p1, p2):
        """Interpolate points between p1 and p2"""
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        dist = int(np.sqrt(dx*dx + dy*dy))
        if dist == 0:
            return [p1]
        points = []
        for i in range(dist + 1):
            t = i / dist
            x = int(p1.x() + t * dx)
            y = int(p1.y() + t * dy)
            points.append(QPoint(x, y))
        return points

    def save_state(self):
        current_state = [seg['mask'].copy() for seg in self.segmentations]
        self.undo_stack.append(current_state)
        if len(self.undo_stack) > self.max_stack_size:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            current_state = [seg['mask'].copy() for seg in self.segmentations]
            self.redo_stack.append(current_state)
            previous_state = self.undo_stack.pop()
            for i, mask in enumerate(previous_state):
                self.segmentations[i]['mask'] = mask
            self.update_display()

    def redo(self):
        if self.redo_stack:
            current_state = [seg['mask'].copy() for seg in self.segmentations]
            self.undo_stack.append(current_state)
            next_state = self.redo_stack.pop()
            for i, mask in enumerate(next_state):
                self.segmentations[i]['mask'] = mask
            self.update_display()

    def clear_segmentation(self):
        try:
            if self.image is not None and self.segmentations:
                for segmentation in self.segmentations:
                    segmentation['mask'].fill(0)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear segmentation: {str(e)}")

    def save_segmentation(self):
        try:
            if self.image is None or not self.segmentations:
                QMessageBox.warning(self, "Warning", "No segmentation to save.")
                return

            if not self.base_directory:
                self.base_directory = QFileDialog.getExistingDirectory(self, "Select Base Directory")
                if not self.base_directory:
                    return

            save_path = os.path.join(self.base_directory, f"{self.image_name}_segmentation")

            format_options = ["PNG", "JPEG", "TIFF"]
            format, ok = QInputDialog.getItem(self, "Select Format", "Image Format:", format_options, 0, False)
            if not ok:
                return

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            combined_mask = np.zeros(self.image.shape, dtype=np.uint8)

            # Save each layer separately (black and white)
            for idx, segmentation in enumerate(self.segmentations):
                layer_image = self.create_layer_image(segmentation, color=False)  # Black and white
                file_name = os.path.join(save_path, f"{self.image_name}_{segmentation['name']}.{format.lower()}")
                cv2.imwrite(file_name, layer_image)

                # Update combined mask (colored)
                if segmentation['visible']:
                    color = np.array(segmentation['color'], dtype=np.uint8)
                    mask = segmentation['mask']
                    combined_mask[mask > 0] = color

            # Save the combined image (colored)
            combined_file_name = os.path.join(save_path, f"{self.image_name}_combined_segmentation.{format.lower()}")
            cv2.imwrite(combined_file_name, cv2.cvtColor(combined_mask, cv2.COLOR_RGB2BGR))

            QMessageBox.information(self, "Success", "Segmentation saved successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save segmentation: {str(e)}")

    def create_layer_image(self, segmentation, color=True):
        """Create an image from the layer's mask with optional color."""
        mask = segmentation['mask']
        if color:
            color = np.array(segmentation['color'], dtype=np.uint8)
            layer_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            layer_image[mask > 0] = color
        else:
            layer_image = np.zeros(mask.shape, dtype=np.uint8)
            layer_image[mask > 0] = 255

        return layer_image
