import os
import sys
import json
import subprocess
import threading
import re
import sys
import os
import subprocess
import json
import glob
import cv2
from safevision_utils import cv2_imread, open_video_capture
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QMessageBox, QProgressBar, 
                            QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QTabWidget, 
                            QGroupBox, QRadioButton, QLineEdit, QTextEdit, QFrame,
                            QSizePolicy, QColorDialog, QSplitter, QScrollArea, QGridLayout,
                            QTreeWidget, QTreeWidgetItem, QAbstractItemView, QMenu, QAction,
                            QStatusBar, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QIcon, QFont, QPainter, QTextCharFormat, QTextCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
import cv2

class WorkerSignals(QObject):
    """Defines signals available for worker threads."""
    progress = pyqtSignal(str)  # Progress text update
    status = pyqtSignal(str)    # Status message
    finished = pyqtSignal(bool, str)  # Success, Result message
    output_line = pyqtSignal(str)  # Real-time output line
    frame_detected = pyqtSignal(str)  # Frame path for preview

class PreviewWidget(QLabel):
    """Widget to display image/video frame preview."""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("No preview available")
        self.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa; 
                border: 2px solid #dee2e6;
                border-radius: 8px;
                font-size: 16px;
                color: #6c757d;
                padding: 20px;
            }
        """)
        self.setMinimumSize(400, 300)
        self.setScaledContents(False)
        self._original_pixmap = None
        self.video_player = None
        self.current_file = None
        
    def setImage(self, image_path):
        """Set image from file path.""" 
        self.clear_video_player()
        self.current_file = image_path
        
        if os.path.exists(image_path):
            if image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                # Create video controls
                self.create_video_controls(image_path)
            else:
                # Display image
                pixmap = QPixmap(image_path)
                self._original_pixmap = pixmap
                scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled_pixmap)
        else:
            self.setText("Preview not available")
            
    def create_video_controls(self, video_path):
        """Create video player controls overlay."""
        # Remove any existing layout
        if self.layout():
            QWidget().setLayout(self.layout())
            
        # Show first frame as background
        try:
            cap = open_video_capture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self._original_pixmap = pixmap
                    scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.setPixmap(scaled_pixmap)
                cap.release()
        except Exception:
            self.setText("Video preview failed")
            
        # Add play button overlay
        play_button = QPushButton("▶ Play Video", self)
        play_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(102, 126, 234, 200);
                color: white;
                border-radius: 20px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid white;
            }
            QPushButton:hover {
                background-color: rgba(90, 111, 216, 220);
            }
        """)
        play_button.clicked.connect(lambda: self.open_video_viewer(video_path))
        
        # Position button in center
        play_button.resize(play_button.sizeHint())
        play_button.move(
            (self.width() - play_button.width()) // 2,
            (self.height() - play_button.height()) // 2
        )
        play_button.show()
        
    def open_video_viewer(self, video_path):
        """Open video in viewer window."""
        self.viewer = ImageVideoViewer(video_path)
        self.viewer.show()
        
    def clear_video_player(self):
        """Clear any video player components."""
        # Remove play button if exists
        for child in self.findChildren(QPushButton):
            child.setParent(None)
            
    def setQImage(self, qimage):
        """Set image from QImage."""
        self.clear_video_player()
        pixmap = QPixmap.fromImage(qimage)
        self._original_pixmap = pixmap
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        
    def resizeEvent(self, event):
        """Handle resize event to rescale image and reposition controls."""
        super().resizeEvent(event)
        if self._original_pixmap:
            scaled_pixmap = self._original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            
        # Reposition play button if exists
        for button in self.findChildren(QPushButton):
            button.move(
                (self.width() - button.width()) // 2,
                (self.height() - button.height()) // 2
            )

class VideoPlayerWidget(QWidget):
    """Video player widget with play/pause controls."""
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.fps = 30
        
        self.init_ui()
        self.load_video()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #dee2e6;
                border-radius: 6px;
            }
        """)
        self.video_label.setMinimumSize(400, 300)
        layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(40, 30)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                border-radius: 15px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6fd8;
            }
        """)
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        back_button = QPushButton("‹")
        back_button.setFixedSize(30, 30)
        back_button.clicked.connect(lambda: self.step_frames(-1))
        controls_layout.addWidget(back_button)

        forward_button = QPushButton("›")
        forward_button.setFixedSize(30, 30)
        forward_button.clicked.connect(lambda: self.step_frames(1))
        controls_layout.addWidget(forward_button)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #dee2e6;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
        """)
        self.progress_slider.sliderMoved.connect(self.seek_to_frame)
        controls_layout.addWidget(self.progress_slider)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #495057; font-size: 12px;")
        controls_layout.addWidget(self.time_label)
        
        layout.addLayout(controls_layout)
        
    def load_video(self):
        """Load video file and get info."""
        try:
            self.cap = open_video_capture(self.video_path)
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = fps if fps > 0 else 30
                self.duration = self.total_frames / self.fps if self.fps > 0 else 0
                self.progress_slider.setRange(0, max(0, self.total_frames - 1))
                
                # Show first frame
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                    self.update_time_label()
                    
        except Exception as e:
            self.video_label.setText(f"Error loading video:\n{str(e)}")
            
    def display_frame(self, frame):
        """Display a frame in the video label."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit label
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        except Exception:
            pass
            
    def toggle_play(self):
        """Toggle play/pause."""
        if not self.cap or not self.cap.isOpened():
            return
            
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False
        else:
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            self.timer.start(interval)
            self.play_button.setText("⏸")
            self.is_playing = True

    def seek_to_frame(self, frame_index):
        """Seek to an exact frame from the slider."""
        if not self.cap or not self.cap.isOpened():
            return
        frame_index = max(0, min(int(frame_index), max(0, self.total_frames - 1)))
        self.current_frame = frame_index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.progress_slider.setValue(frame_index)
            self.update_time_label()

    def step_frames(self, amount):
        """Step one frame backward or forward."""
        self.seek_to_frame(self.current_frame + amount)
            
    def update_frame(self):
        """Update to next frame."""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            self.display_frame(frame)
            self.progress_slider.setValue(self.current_frame)
            self.update_time_label()
        else:
            # End of video, stop playback
            self.timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False
            
    def update_time_label(self):
        """Update time display."""
        if self.total_frames > 0:
            current_time = self.current_frame / self.fps if self.fps > 0 else 0
            total_time = self.total_frames / self.fps if self.fps > 0 else 0
            
            current_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            total_str = f"{int(total_time//60):02d}:{int(total_time%60):02d}"
            self.time_label.setText(f"{current_str} / {total_str}")
            
    def closeEvent(self, event):
        """Clean up when closing."""
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()

class ImageVideoViewer(QWidget):
    """Enhanced image/video viewer window with zoom support."""
    def __init__(self, file_path, embedded=False):
        super().__init__()
        self.file_path = file_path
        self.embedded = embedded
        self.zoom_factor = 1.0
        self.original_pixmap = None
        
        if not embedded:
            self.setWindowTitle(f"SafeVision Viewer - {os.path.basename(file_path)}")
            self.setWindowFlags(Qt.Window)
            self.resize(1000, 700)
            
            # Center window on screen
            screen = QApplication.desktop().screenGeometry()
            self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header with file info and controls
        header_layout = QHBoxLayout()
        
        title_label = QLabel(f"📁 {os.path.basename(self.file_path)}")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #495057;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Zoom controls for images
        if self.file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            zoom_out_btn = QPushButton("🔍-")
            zoom_out_btn.setFixedSize(40, 30)
            zoom_out_btn.clicked.connect(self.zoom_out)
            header_layout.addWidget(zoom_out_btn)
            
            self.zoom_label = QLabel("100%")
            self.zoom_label.setStyleSheet("color: #495057; font-weight: bold; padding: 0 10px;")
            header_layout.addWidget(self.zoom_label)
            
            zoom_in_btn = QPushButton("🔍+")
            zoom_in_btn.setFixedSize(40, 30)
            zoom_in_btn.clicked.connect(self.zoom_in)
            header_layout.addWidget(zoom_in_btn)
            
            reset_btn = QPushButton("↻")
            reset_btn.setFixedSize(40, 30)
            reset_btn.setToolTip("Reset Zoom")
            reset_btn.clicked.connect(self.reset_zoom)
            header_layout.addWidget(reset_btn)
        
        # Close button
        close_btn = QPushButton("✕ Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Content area with scroll for images
        if os.path.exists(self.file_path):
            if self.file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Image viewer with scroll area
                self.scroll_area = QScrollArea()
                self.scroll_area.setWidgetResizable(True)
                self.scroll_area.setAlignment(Qt.AlignCenter)
                
                self.image_label = QLabel()
                self.image_label.setAlignment(Qt.AlignCenter)
                self.image_label.setStyleSheet("""
                    QLabel {
                        background-color: #f8f9fa;
                        border: 2px solid #dee2e6;
                        border-radius: 8px;
                    }
                """)
                
                self.original_pixmap = QPixmap(self.file_path)
                self.update_image_display()
                
                self.scroll_area.setWidget(self.image_label)
                layout.addWidget(self.scroll_area)
                
            elif self.file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Enhanced video player
                video_player = EnhancedVideoPlayer(self.file_path)
                layout.addWidget(video_player)
                
        else:
            error_label = QLabel("❌ File not found")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: #dc3545; font-size: 18px;")
            layout.addWidget(error_label)
    
    def zoom_in(self):
        """Zoom in on the image."""
        self.zoom_factor = min(self.zoom_factor * 1.25, 5.0)  # Max 500%
        self.update_image_display()
        
    def zoom_out(self):
        """Zoom out on the image."""
        self.zoom_factor = max(self.zoom_factor / 1.25, 0.1)  # Min 10%
        self.update_image_display()
        
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_factor = 1.0
        self.update_image_display()
        
    def update_image_display(self):
        """Update the image display with current zoom."""
        if self.original_pixmap:
            scaled_size = self.original_pixmap.size() * self.zoom_factor
            scaled_pixmap = self.original_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
            
            # Update zoom label
            self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if self.original_pixmap:  # Only for images
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            if self.original_pixmap:
                self.zoom_in()
        elif event.key() == Qt.Key_Minus:
            if self.original_pixmap:
                self.zoom_out()
        elif event.key() == Qt.Key_0:
            if self.original_pixmap:
                self.reset_zoom()

class EnhancedVideoPlayer(QWidget):
    """Enhanced video player with volume control and better UI."""
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.volume = 50  # Volume percentage
        
        self.init_ui()
        self.load_video()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #dee2e6;
                border-radius: 6px;
            }
        """)
        self.video_label.setMinimumSize(600, 400)
        layout.addWidget(self.video_label)
        
        # Progress bar (clickable)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 10px;
                background: #dee2e6;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.progress_slider.sliderMoved.connect(self.seek_to_frame)
        layout.addWidget(self.progress_slider)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 5, 10, 5)
        
        # Play/Pause button
        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(50, 40)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                border-radius: 20px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6fd8;
            }
        """)
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        back_button = QPushButton("‹")
        back_button.setFixedSize(40, 40)
        back_button.clicked.connect(lambda: self.step_frames(-1))
        controls_layout.addWidget(back_button)

        forward_button = QPushButton("›")
        forward_button.setFixedSize(40, 40)
        forward_button.clicked.connect(lambda: self.step_frames(1))
        controls_layout.addWidget(forward_button)
        
        # Stop button
        stop_button = QPushButton("⏹")
        stop_button.setFixedSize(40, 40)
        stop_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        stop_button.clicked.connect(self.stop_video)
        controls_layout.addWidget(stop_button)
        
        controls_layout.addStretch()
        
        # Volume controls
        vol_label = QLabel("🔊")
        controls_layout.addWidget(vol_label)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(self.volume)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #dee2e6;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.volume_slider.valueChanged.connect(self.change_volume)
        controls_layout.addWidget(self.volume_slider)
        
        controls_layout.addStretch()
        
        # Time and file info
        info_layout = QVBoxLayout()
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #495057; font-size: 14px; font-weight: bold;")
        info_layout.addWidget(self.time_label)
        
        # File size info
        try:
            file_size = os.path.getsize(self.video_path) / (1024 * 1024)  # MB
            self.size_label = QLabel(f"Size: {file_size:.1f} MB")
            self.size_label.setStyleSheet("color: #6c757d; font-size: 10px;")
            info_layout.addWidget(self.size_label)
        except:
            pass
            
        controls_layout.addLayout(info_layout)
        
        layout.addLayout(controls_layout)
        
    def load_video(self):
        """Load video file and get info."""
        try:
            self.cap = open_video_capture(self.video_path)
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = fps if fps > 0 else 30
                self.duration = self.total_frames / self.fps
                self.progress_slider.setRange(0, max(0, self.total_frames - 1))
                
                # Show first frame
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                    self.update_time_label()
                    
        except Exception as e:
            self.video_label.setText(f"Error loading video:\n{str(e)}")
            
    def display_frame(self, frame):
        """Display a frame in the video label."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit label
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        except Exception:
            pass
            
    def toggle_play(self):
        """Toggle play/pause."""
        if not self.cap or not self.cap.isOpened():
            return
            
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False
        else:
            interval = int(1000 / self.fps)
            self.timer.start(interval)
            self.play_button.setText("⏸")
            self.is_playing = True
            
    def stop_video(self):
        """Stop video and return to beginning."""
        self.timer.stop()
        self.play_button.setText("▶")
        self.is_playing = False
        self.current_frame = 0
        
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.progress_slider.setValue(0)
                self.update_time_label()
            
    def update_frame(self):
        """Update to next frame."""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            self.display_frame(frame)
            self.progress_slider.setValue(self.current_frame)
            self.update_time_label()
        else:
            # End of video, stop playback
            self.timer.stop()
            self.play_button.setText("▶")
            self.is_playing = False
            
    def seek_to_frame(self, frame_index):
        """Seek to an exact frame from the slider."""
        if not self.cap or not self.cap.isOpened() or self.total_frames <= 0:
            return
        seek_frame = max(0, min(int(frame_index), self.total_frames - 1))
        self.current_frame = seek_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.progress_slider.setValue(seek_frame)
            self.update_time_label()

    def step_frames(self, amount):
        """Step one frame backward or forward."""
        self.seek_to_frame(self.current_frame + amount)

    def change_volume(self, value):
        """Store the UI volume value for future audio-capable playback."""
        self.volume = max(0, min(int(value), 100))
        
    def update_time_label(self):
        """Update time display."""
        if self.total_frames > 0:
            current_time = self.current_frame / self.fps
            total_time = self.total_frames / self.fps
            
            current_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            total_str = f"{int(total_time//60):02d}:{int(total_time%60):02d}"
            self.time_label.setText(f"{current_str} / {total_str}")
            
    def closeEvent(self, event):
        """Clean up when closing."""
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()

class FullScreenImageViewer(QWidget):
    """Full screen image viewer widget."""
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.setWindowFlags(Qt.Window)
        self.setStyleSheet("background-color: black;")
        self.showFullScreen()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        
        # Load and display image
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Scale to fit screen
            screen_size = QApplication.desktop().screenGeometry()
            scaled_pixmap = pixmap.scaled(screen_size.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        
        layout.addWidget(self.image_label)
        
        # Close instructions
        close_label = QLabel("Press ESC or Click to Close")
        close_label.setAlignment(Qt.AlignCenter)
        close_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
                padding: 10px;
                background-color: rgba(0, 0, 0, 128);
                border-radius: 5px;
            }
        """)
        layout.addWidget(close_label)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            
    def mousePressEvent(self, event):
        self.close()

class GridPreviewWidget(QLabel):
    """Clickable preview widget for grid display."""
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
            }
            QLabel:hover {
                border-color: #667eea;
                background-color: #f0f2ff;
                transform: scale(1.02);
            }
        """)
        self.setFixedSize(220, 165)  # More consistent sizing
        self.setCursor(Qt.PointingHandCursor)
        
        # Load and display image
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Scale to fit with some padding
            scaled_pixmap = pixmap.scaled(200, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
        else:
            self.setText("Image not found")
            self.setStyleSheet(self.styleSheet() + "color: #dc3545;")
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and os.path.exists(self.image_path):
            # Find the parent MultiPreviewWidget and use its parent_gui reference
            parent_multi_preview = self.find_parent(MultiPreviewWidget)
            
            if parent_multi_preview and parent_multi_preview.parent_gui and \
               hasattr(parent_multi_preview.parent_gui, 'output_preview_type_combo') and \
               parent_multi_preview.parent_gui.output_preview_type_combo.currentText() == "Included":
                # Use embedded preview - this will be implemented in parent GUI
                if hasattr(parent_multi_preview.parent_gui, "show_media_preview"):
                    parent_multi_preview.parent_gui.show_media_preview(self.image_path)
            else:
                # Default to dialog viewer
                self.viewer = ImageVideoViewer(self.image_path)
                self.viewer.show()
                
    def find_parent(self, widget_class):
        """Find parent widget of a specific class type."""
        parent = self.parent()
        while parent:
            if isinstance(parent, widget_class):
                return parent
            parent = parent.parent()
        return None
            
    def enterEvent(self, event):
        """Add hover effect."""
        self.setStyleSheet("""
            QLabel {
                background-color: #f0f2ff;
                border: 3px solid #667eea;
                border-radius: 8px;
                padding: 7px;
            }
        """)
        
    def leaveEvent(self, event):
        """Remove hover effect."""
        self.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
            }
        """)

class MultiPreviewWidget(QWidget):
    """Widget to display multiple output images/videos in a grid."""
    def __init__(self, parent_gui=None):
        super().__init__()
        self.parent_gui = parent_gui  # Store reference to main GUI
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
            }
        """)
        
        # Create scroll area for multiple outputs
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #e9ecef;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #6c757d;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #495057;
            }
        """)
        
        # Content widget inside scroll area
        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
        # Default empty state
        self.empty_label = QLabel("Output will appear here after processing...")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 16px;
                padding: 40px;
                border: none;
            }
        """)
        self.content_layout.addWidget(self.empty_label, 0, 0)
        
        self.scroll_area.setWidget(self.content_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area)
        
        self.preview_widgets = []
        
    def clear_previews(self):
        """Clear all preview widgets."""
        for widget in self.preview_widgets:
            widget.setParent(None)
        self.preview_widgets.clear()
        
        # Clear grid layout
        for i in reversed(range(self.content_layout.count())):
            self.content_layout.itemAt(i).widget().setParent(None)
        
        # Show empty state
        self.content_layout.addWidget(self.empty_label, 0, 0)
        
    def add_preview(self, file_path, title=""):
        """Add a preview widget for a file."""
        # Hide empty state
        if self.empty_label.parent():
            self.empty_label.setParent(None)
            
        # Create preview container
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin: 5px;
            }
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        
        # Title label
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                QLabel {
                    color: #495057;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 3px;
                    border: none;
                }
            """)
            title_label.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(title_label)
        
        # Preview widget
        if os.path.exists(file_path):
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                preview = GridPreviewWidget(file_path)
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Show first frame of video with click functionality
                preview = QLabel()
                preview.setAlignment(Qt.AlignCenter)
                preview.setStyleSheet("""
                    QLabel {
                        background-color: #ffffff;
                        border: 2px solid #dee2e6;
                        border-radius: 8px;
                        padding: 8px;
                    }
                    QLabel:hover {
                        border-color: #667eea;
                        background-color: #f0f2ff;
                    }
                """)
                preview.setFixedSize(220, 165)
                preview.setCursor(Qt.PointingHandCursor)
                
                try:
                    cap = open_video_capture(file_path)
                    ret, frame = cap.read()
                    if ret:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                        scaled_pixmap = pixmap.scaled(200, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        preview.setPixmap(scaled_pixmap)
                    cap.release()
                    
                    # Make it clickable - check parent GUI's output preview setting
                    def open_video():
                        try:
                            # Check if parent GUI exists and get its output preview setting
                            if (self.parent_gui and 
                                hasattr(self.parent_gui, 'output_preview_type_combo') and 
                                self.parent_gui.output_preview_type_combo.currentText() == "Included"):
                                # Use embedded preview through parent GUI
                                if hasattr(self.parent_gui, "show_media_preview"):
                                    self.parent_gui.show_media_preview(file_path)
                            else:
                                # Fall back to dialog viewer
                                viewer = ImageVideoViewer(file_path)
                                viewer.show()
                                # Keep reference to prevent garbage collection
                                if not hasattr(preview, '_viewer_refs'):
                                    preview._viewer_refs = []
                                preview._viewer_refs.append(viewer)
                        except Exception as e:
                            print(f"Error opening video: {e}")
                    
                    preview.mousePressEvent = lambda event: open_video() if event.button() == Qt.LeftButton else None
                    
                except Exception:
                    preview.setText(f"Video preview\n{os.path.basename(file_path)}")
                    # Still make it clickable even if preview fails
                    def open_video():
                        try:
                            # Check if parent GUI exists and get its output preview setting
                            if (self.parent_gui and 
                                hasattr(self.parent_gui, 'output_preview_type_combo') and 
                                self.parent_gui.output_preview_type_combo.currentText() == "Included"):
                                # Use embedded preview through parent GUI
                                if hasattr(self.parent_gui, "show_media_preview"):
                                    self.parent_gui.show_media_preview(file_path)
                            else:
                                # Fall back to dialog viewer
                                viewer = ImageVideoViewer(file_path)
                                viewer.show()
                                # Keep reference to prevent garbage collection
                                if not hasattr(preview, '_viewer_refs'):
                                    preview._viewer_refs = []
                                preview._viewer_refs.append(viewer)
                        except Exception as e:
                            print(f"Error opening video: {e}")
                    preview.mousePressEvent = lambda event: open_video() if event.button() == Qt.LeftButton else None
            elif file_path.lower().endswith(('.json', '.csv', '.edl', '.fcpxml', '.xml')):
                preview = QLabel()
                preview.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                preview.setFixedSize(220, 165)
                preview.setWordWrap(True)
                preview.setStyleSheet("""
                    QLabel {
                        background-color: #f8f9fa;
                        border: 2px solid #dee2e6;
                        border-radius: 8px;
                        padding: 8px;
                        color: #343a40;
                        font-family: Consolas, monospace;
                        font-size: 10px;
                    }
                """)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as text_file:
                        text = text_file.read(600)
                    preview.setText(text or os.path.basename(file_path))
                except Exception:
                    preview.setText(os.path.basename(file_path))
            else:
                preview = QLabel(os.path.basename(file_path))
                preview.setAlignment(Qt.AlignCenter)
                preview.setMinimumSize(200, 150)
        else:
            preview = QLabel("File not found")
            preview.setAlignment(Qt.AlignCenter)
            preview.setMinimumSize(200, 150)
        
        container_layout.addWidget(preview)
        
        # File info
        file_size = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
        info_label = QLabel(f"{os.path.basename(file_path)}\n({file_size:.1f} MB)")
        info_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 10px;
                padding: 3px;
                border: none;
            }
        """)
        info_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(info_label)
        
        # Add to grid layout (3 columns)
        row = len(self.preview_widgets) // 3
        col = len(self.preview_widgets) % 3
        self.content_layout.addWidget(container, row, col)
        
        self.preview_widgets.append(container)
        
        # Scroll to bottom to show newest
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()))


class SplittableTabWidget(QTabWidget):
    """Custom tab widget with split functionality."""
    def __init__(self):
        super().__init__()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.split_container = None
        self.is_split = False
        self.active_split = None
    
    def show_context_menu(self, position):
        """Show context menu for tab splitting."""
        menu = QMenu(self)
        
        if not self.is_split:
            split_action = QAction("📱 Split Vertically", self)
            split_action.triggered.connect(self.split_vertically)
            menu.addAction(split_action)
            
            split_h_action = QAction("📋 Split Horizontally", self)
            split_h_action.triggered.connect(self.split_horizontally)
            menu.addAction(split_h_action)
        else:
            restore_action = QAction("🔄 Restore Single View", self)
            restore_action.triggered.connect(self.restore_single_view)
            menu.addAction(restore_action)
        
        menu.exec_(self.mapToGlobal(position))
    
    def split_vertically(self):
        """Split the tab widget vertically."""
        if self.is_split:
            return
            
        self.create_split_view(Qt.Horizontal)
    
    def split_horizontally(self):
        """Split the tab widget horizontally."""
        if self.is_split:
            return
            
        self.create_split_view(Qt.Vertical)
    
    def create_split_view(self, orientation):
        """Create split view with the specified orientation."""
        # Get parent layout
        parent = self.parent()
        layout = parent.layout()
        
        # Create splitter
        self.split_container = QSplitter(orientation)
        self.split_container.setHandleWidth(5)
        self.split_container.setStyleSheet("""
            QSplitter::handle {
                background-color: #dee2e6;
                border: 1px solid #adb5bd;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #667eea;
            }
        """)
        
        # Create second tab widget with same styling
        self.split_widget = SplittableTabWidget()
        self.split_widget.setStyleSheet(self.styleSheet())  # Copy styling
        self.split_widget.active_split = "right"
        self.active_split = "left"
        
        # Copy tabs to second widget with functional content
        for i in range(self.count()):
            tab_text = self.tabText(i)
            
            # Create functional tab copies based on original tab type
            if "Input Preview" in tab_text:
                new_tab = self.create_functional_input_tab()
            elif "Output Preview" in tab_text:
                new_tab = self.create_functional_output_tab()
            elif "TreeView" in tab_text:
                new_tab = self.create_functional_tree_tab()
            elif "Media Preview" in tab_text:
                new_tab = self.create_functional_media_tab()
            else:
                # Default functional tab
                new_tab = self.create_default_split_tab(tab_text)
            
            self.split_widget.addTab(new_tab, tab_text)
        
        # Add both widgets to splitter
        self.split_container.addWidget(self)
        self.split_container.addWidget(self.split_widget)
        self.split_container.setSizes([300, 300])  # Equal sizes
        
        # Replace current widget with splitter
        layout_index = layout.indexOf(self)
        layout.removeWidget(self)
        layout.insertWidget(layout_index, self.split_container)
        
        self.is_split = True
        self.split_widget.is_split = True
        
        # Disconnect any existing signals first
        try:
            self.tabBarClicked.disconnect()
        except:
            pass
        try:
            self.split_widget.tabBarClicked.disconnect()
        except:
            pass
        
        # Connect split widget selection with proper error handling
        self.split_widget.tabBarClicked.connect(lambda: self.set_active_split("right") if self.split_widget is not None else None)
        self.tabBarClicked.connect(lambda: self.set_active_split("left") if self.split_widget is not None else None)
    
    def create_functional_input_tab(self):
        """Create a functional input preview tab for split view."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Preview type selector
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Preview Type:"))
        preview_combo = QComboBox()
        preview_combo.addItems(["Dialog", "Included"])
        preview_combo.setCurrentText("Included")  # Default to Included
        preview_combo.setFixedHeight(25)
        preview_layout.addWidget(preview_combo)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)
        
        # Preview widget
        preview_widget = PreviewWidget()
        layout.addWidget(preview_widget)
        
        return tab
    
    def create_functional_output_tab(self):
        """Create a functional output preview tab for split view."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Preview type selector
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Preview Type:"))
        preview_combo = QComboBox()
        preview_combo.addItems(["Dialog", "Included"])
        preview_combo.setCurrentText("Included")  # Default to Included
        preview_combo.setFixedHeight(25)
        preview_layout.addWidget(preview_combo)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)
        
        # Preview widget
        preview_widget = MultiPreviewWidget(None)  # No parent reference in split view
        layout.addWidget(preview_widget)
        
        return tab
    
    def create_functional_tree_tab(self):
        """Create a functional tree view tab for split view."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Preview type selector
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Media Preview Type:"))
        preview_combo = QComboBox()
        preview_combo.addItems(["Dialog", "Included"])
        preview_combo.setCurrentText("Included")  # Default to Included
        preview_combo.setFixedHeight(25)
        preview_layout.addWidget(preview_combo)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)
        
        # Tree view widget
        tree_view = FileTreeWidget()
        tree_view.setHeaderLabel("📁 File Explorer (Split View)")
        tree_view.load_directory()  # Fixed: use load_directory instead of populate_tree
        layout.addWidget(tree_view)
        
        return tab
    
    def create_functional_media_tab(self):
        """Create a functional media preview tab for split view."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Default message
        label = QLabel("Select a media file from TreeView to preview here (Split View)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 16px;
                padding: 40px;
                border: 2px dashed #dee2e6;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
        """)
        layout.addWidget(label)
        
        return tab
    
    def create_default_split_tab(self, tab_text):
        """Create a default split tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        label = QLabel(f"Split view of: {tab_text}")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #6c757d; font-size: 14px; padding: 20px;")
        layout.addWidget(label)
        
        return tab
    
    def set_active_split(self, split):
        """Set which split is active."""
        self.active_split = split
        if hasattr(self, 'split_widget') and self.split_widget is not None:
            self.split_widget.active_split = split
    
    def restore_single_view(self):
        """Restore single tab view."""
        if not self.is_split or not self.split_container:
            return
            
        # Disconnect any existing signals to prevent errors
        try:
            self.tabBarClicked.disconnect()
        except:
            pass
            
        # Get parent layout
        parent = self.split_container.parent()
        layout = parent.layout()
        
        # Remove splitter and restore original widget
        layout_index = layout.indexOf(self.split_container)
        layout.removeWidget(self.split_container)
        
        # Remove from splitter using setParent(None) instead of removeWidget
        self.setParent(None)
        
        # Add back to layout
        layout.insertWidget(layout_index, self)
        
        # Clean up split widget first
        if hasattr(self, 'split_widget') and self.split_widget:
            try:
                self.split_widget.tabBarClicked.disconnect()
            except:
                pass
            self.split_widget.setParent(None)
            self.split_widget.deleteLater()
        
        # Clean up container
        self.split_container.setParent(None)
        self.split_container.deleteLater()
        
        # Reset all split-related attributes
        self.split_container = None
        self.split_widget = None
        self.is_split = False
        self.active_split = None


class ColoredTextEdit(QTextEdit):
    """Enhanced QTextEdit with colored output support."""
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
            }
            /* Custom hidden scrollbar that maintains functionality */
            QTextEdit QScrollBar:vertical {
                background-color: transparent;
                width: 8px;
                border-radius: 4px;
                margin: 0px;
            }
            QTextEdit QScrollBar::handle:vertical {
                background-color: rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                min-height: 20px;
                margin: 2px;
            }
            QTextEdit QScrollBar::handle:vertical:hover {
                background-color: rgba(255, 255, 255, 0.4);
            }
            QTextEdit QScrollBar::handle:vertical:pressed {
                background-color: rgba(255, 255, 255, 0.6);
            }
            QTextEdit QScrollBar::add-line:vertical,
            QTextEdit QScrollBar::sub-line:vertical {
                height: 0px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QTextEdit QScrollBar::add-page:vertical,
            QTextEdit QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
        
    def append_colored(self, text, color="#ffffff"):
        """Append text with specific color."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        format = QTextCharFormat()
        format.setForeground(QColor(color))
        cursor.setCharFormat(format)
        cursor.insertText(text + "\n")
        
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
    def append_info(self, text):
        """Append info message in cyan."""
        self.append_colored(f"ℹ {text}", "#17a2b8")
        
    def append_success(self, text):
        """Append success message in green."""
        self.append_colored(f"✓ {text}", "#28a745")
        
    def append_warning(self, text):
        """Append warning message in yellow."""
        self.append_colored(f"⚠ {text}", "#ffc107")
        
    def append_error(self, text):
        """Append error message in red."""
        self.append_colored(f"✗ {text}", "#dc3545")
        
    def append_progress(self, text):
        """Append progress message in blue."""
        self.append_colored(f"⏳ {text}", "#007bff")

class CommandWorker(QThread):
    """Thread for running command-line processing in background."""
    def __init__(self, command, working_directory):
        super().__init__()
        self.command = command
        self.working_directory = working_directory
        self.signals = WorkerSignals()
        self.is_running = True
        self.process = None
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.check_for_frames)
        
    def run(self):
        try:
            self.signals.status.emit(f"Starting command: {' '.join(self.command)}")
            
            # Start the process
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'  # Fix encoding issues
            env['PYTHONLEGACYWINDOWSFSENCODING'] = '1'  # Additional Windows fix
            
            self.process = subprocess.Popen(
                self.command,
                cwd=self.working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',  # Replace problematic characters instead of failing
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Start frame monitoring timer
            self.frame_timer.start(500)  # Check every 500ms
            
            # Read output line by line
            while self.is_running and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    self.signals.output_line.emit(line)
                    self.signals.progress.emit(line)
            
            # Stop frame monitoring
            self.frame_timer.stop()
            
            # Get any remaining output
            remaining_output, _ = self.process.communicate()
            if remaining_output:
                for line in remaining_output.strip().split('\n'):
                    if line.strip():
                        self.signals.output_line.emit(line.strip())
            
            # Check return code
            if self.process.returncode == 0:
                self.signals.finished.emit(True, "Processing completed successfully")
            else:
                self.signals.finished.emit(False, f"Process failed with return code: {self.process.returncode}")
                
        except Exception as e:
            self.signals.finished.emit(False, f"Error running command: {str(e)}")
            
    def check_for_frames(self):
        """Check for new frames in output directory."""
        try:
            # Look for frame files in output_frames directory
            output_frames_dir = os.path.join(self.working_directory, "output_frames")
            if os.path.exists(output_frames_dir):
                frame_files = [f for f in os.listdir(output_frames_dir) if f.startswith("frame_") and f.endswith(".jpg")]
                if frame_files:
                    # Get the latest frame
                    latest_frame = max(frame_files, key=lambda x: os.path.getctime(os.path.join(output_frames_dir, x)))
                    frame_path = os.path.join(output_frames_dir, latest_frame)
                    self.signals.frame_detected.emit(frame_path)
        except Exception:
            pass  # Silently ignore errors in frame detection
            
    def stop(self):
        self.is_running = False
        if self.frame_timer:
            self.frame_timer.stop()
        if self.process:
            self.process.terminate()

class DropArea(QLabel):
    """Widget that accepts drag & drop of files."""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("📁 Drag & Drop Media Files Here\n\nOr Click to Browse Files\n\n✨ Supported formats: MP4, AVI, MOV, JPG, PNG ✨")
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #f8f9fa, stop:1 #e9ecef);
                font-size: 14px;
                font-weight: bold;
                min-height: 120px;
                color: #2c3e50;
                text-align: center;
            }
        """)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QLabel {
                    border: 3px dashed #5a6fd8;
                    border-radius: 15px;
                    padding: 40px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #e8ecf7, stop:1 #d1d9f4);
                    font-size: 16px;
                    font-weight: bold;
                    min-height: 180px;
                    color: #2c3e50;
                    text-align: center;
                }
            """)
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #f8f9fa, stop:1 #e9ecef);
                font-size: 16px;
                font-weight: bold;
                min-height: 180px;
                color: #2c3e50;
                text-align: center;
            }
        """)
        
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            # Find the main window by traversing up the parent hierarchy
            widget = self
            while widget.parent():
                widget = widget.parent()
                if hasattr(widget, 'load_file'):
                    widget.load_file(file_path)
                    break
        else:
            event.ignore()
        
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #f8f9fa, stop:1 #e9ecef);
                font-size: 16px;
                font-weight: bold;
                min-height: 180px;
                color: #2c3e50;
                text-align: center;
            }
        """)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Find the main window by traversing up the parent hierarchy
            widget = self
            while widget.parent():
                widget = widget.parent()
                if hasattr(widget, 'browse_file'):
                    widget.browse_file()
                    break

class ModernProgressBar(QWidget):
    """Modern animated progress bar with text overlay."""
    def __init__(self):
        super().__init__()
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 20px;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 8, 15, 8)
        
        # Progress text
        self.progress_label = QLabel("Processing...")
        self.progress_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
                border: none;
                background: transparent;
            }
        """)
        layout.addWidget(self.progress_label)
        
        # Animated dots
        self.dots_label = QLabel("...")
        self.dots_label.setStyleSheet("""
            QLabel {
                color: #667eea;
                font-weight: bold;
                font-size: 14px;
                border: none;
                background: transparent;
            }
        """)
        layout.addWidget(self.dots_label)
        layout.addStretch()
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_dots)
        self.dot_count = 0
        
    def start_animation(self, text="Processing..."):
        """Start the progress animation."""
        self.progress_label.setText(text)
        self.animation_timer.start(500)  # Update every 500ms
        self.show()
        
    def stop_animation(self):
        """Stop the progress animation."""
        self.animation_timer.stop()
        self.hide()
        
    def animate_dots(self):
        """Animate the dots."""
        self.dot_count = (self.dot_count + 1) % 4
        dots = "." * self.dot_count + " " * (3 - self.dot_count)
        self.dots_label.setText(dots)
        
    def update_text(self, text):
        """Update progress text."""
        self.progress_label.setText(text)

class ModernToggle(QCheckBox):
    """Modern style toggle switch."""
    def __init__(self, text=""):
        super().__init__(text)
        self.setStyleSheet("""
            QCheckBox {
                spacing: 5px;
                font-size: 14px;
            }
            
            QCheckBox::indicator {
                width: 40px;
                height: 20px;
                border-radius: 10px;
            }
            
            QCheckBox::indicator:unchecked {
                background-color: #e0e0e0;
            }
            
            QCheckBox::indicator:checked {
                background-color: #2196F3;
            }
            
            QCheckBox::indicator:unchecked:hover {
                background-color: #d0d0d0;
            }
            
            QCheckBox::indicator:checked:hover {
                background-color: #1E88E5;
            }
        """)

class ColorPickerButton(QPushButton):
    """Button that opens a color picker dialog."""
    def __init__(self, initial_color=None):
        super().__init__()
        self.color = initial_color or [0, 0, 0]  # BGR format
        self.setText("Select Color")
        self.setStyleSheet(self._get_style())
        self.clicked.connect(self.pick_color)
        
    def pick_color(self):
        # Convert BGR to RGB for the color dialog
        rgb_color = QColor(self.color[2], self.color[1], self.color[0])
        color = QColorDialog.getColor(rgb_color, self, "Select Color")
        
        if color.isValid():
            # Store as BGR
            self.color = [color.blue(), color.green(), color.red()]
            self.setStyleSheet(self._get_style())
            
    def _get_style(self):
        r, g, b = self.color[2], self.color[1], self.color[0]
        return f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                color: {'white' if (r+g+b)/3 < 128 else 'black'};
                border-radius: 4px;
                padding: 5px 10px;
            }}
        """
        
    def get_color(self):
        return self.color
    
    def set_color(self, color):  # BGR format
        self.color = color
        self.setStyleSheet(self._get_style())

class FileTreeWidget(QTreeWidget):
    """File tree widget for exploring current directory."""
    def __init__(self):
        super().__init__()
        self.setHeaderLabel("📁 File Explorer")
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Callback for embedded preview (set by parent when needed)
        self.file_selected_for_preview = None
        
        # Style the tree
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #ffffff;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                font-size: 14px;
            }
            QTreeWidget::item {
                padding: 4px;
                border-bottom: 1px solid #f1f3f4;
            }
            QTreeWidget::item:hover {
                background-color: #f0f2ff;
            }
            QTreeWidget::item:selected {
                background-color: #667eea;
                color: white;
            }
            QTreeWidget::branch:closed:has-children {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAJCAYAAADgkQYQAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAFVSURBVBiVjZA7SwNBEIafgwQSCxsrwcJaG1sLG0uxsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsA==);
            }
            QTreeWidget::branch:open:has-children {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAJCAYAAADgkQYQAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAFVSURBVBiVjZA7SwNBEIafgwQSCxsrwcJaG1sLG0uxsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsNGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsA==);
            }
        """)
        
        # Connect double-click event
        self.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        # Load current directory
        self.load_directory()
        
    def load_directory(self):
        """Load the current working directory into tree."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_item = QTreeWidgetItem([os.path.basename(current_dir) or current_dir])
        root_item.setIcon(0, self.style().standardIcon(self.style().SP_DirIcon))
        root_item.setData(0, Qt.UserRole, current_dir)
        self.addTopLevelItem(root_item)
        
        self.populate_directory(root_item, current_dir)
        
        # Expand root item
        root_item.setExpanded(True)
        
    def populate_directory(self, parent_item, directory_path):
        """Recursively populate directory tree."""
        try:
            if not os.path.exists(directory_path):
                return
                
            items = []
            
            # Get directories and files
            for item_name in os.listdir(directory_path):
                if item_name.startswith('.'):  # Skip hidden files
                    continue
                    
                item_path = os.path.join(directory_path, item_name)
                
                if os.path.isdir(item_path):
                    # Directory
                    tree_item = QTreeWidgetItem([f"📁 {item_name}"])
                    tree_item.setIcon(0, self.style().standardIcon(self.style().SP_DirIcon))
                    tree_item.setData(0, Qt.UserRole, item_path)
                    items.append((0, tree_item))  # 0 for directories (sort first)
                    
                    # Add placeholder child for expansion
                    placeholder = QTreeWidgetItem(["Loading..."])
                    tree_item.addChild(placeholder)
                    
                elif item_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                    # Media file
                    icon_text = "🎥" if item_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')) else "🖼️"
                    tree_item = QTreeWidgetItem([f"{icon_text} {item_name}"])
                    if item_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                        tree_item.setIcon(0, self.style().standardIcon(self.style().SP_FileIcon))
                    else:
                        tree_item.setIcon(0, self.style().standardIcon(self.style().SP_FileIcon))
                    tree_item.setData(0, Qt.UserRole, item_path)
                    items.append((1, tree_item))  # 1 for files (sort after directories)
            
            # Sort items (directories first, then files)
            items.sort(key=lambda x: (x[0], x[1].text(0).lower()))
            
            # Add sorted items to parent
            for _, item in items:
                parent_item.addChild(item)
                
        except PermissionError:
            # Handle permission denied
            error_item = QTreeWidgetItem(["❌ Permission denied"])
            parent_item.addChild(error_item)
        except Exception as e:
            # Handle other errors
            error_item = QTreeWidgetItem([f"❌ Error: {str(e)}"])
            parent_item.addChild(error_item)
    
    def on_item_double_clicked(self, item, column):
        """Handle double-click events on tree items."""
        file_path = item.data(0, Qt.UserRole)
        
        if not file_path or not os.path.exists(file_path):
            return
            
        if os.path.isfile(file_path):
            # Handle file opening
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                try:
                    if self.file_selected_for_preview:
                        # Use embedded preview
                        self.file_selected_for_preview(file_path)
                    else:
                        # Use dialog preview
                        viewer = ImageVideoViewer(file_path)
                        viewer.show()
                        # Keep reference to prevent garbage collection
                        if not hasattr(self, '_viewer_refs'):
                            self._viewer_refs = []
                        self._viewer_refs.append(viewer)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to open file:\n{str(e)}")
    
    def mousePressEvent(self, event):
        """Handle mouse press events for folder expansion."""
        super().mousePressEvent(event)
        
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item:
                file_path = item.data(0, Qt.UserRole)
                if file_path and os.path.isdir(file_path):
                    # Toggle expansion on single click for directories
                    if item.isExpanded():
                        item.setExpanded(False)
                    else:
                        # Clear children and reload
                        item.takeChildren()
                        self.populate_directory(item, file_path)
                        item.setExpanded(True)

class SafeVisionGUI(QMainWindow):
    """Main GUI window for SafeVision using subprocess calls."""
    def __init__(self):
        super().__init__()
        
        # Configuration variable for hiding command output
        self.hideCommand = True  # Set to False to show commands in log
        
        # Load settings
        self.settings = self._load_settings()
        
        # Set up the UI
        self.init_ui()
        
        # Initialize variables
        self.current_file = None
        self.worker = None
        
    def create_info_button(self, info_text):
        """Create an info button that shows a dialog when clicked."""
        info_button = QPushButton("ℹ️")
        info_button.setFixedSize(25, 25)
        info_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        info_button.setToolTip("Click for information")
        info_button.clicked.connect(lambda: self.show_info_dialog(info_text))
        return info_button
        
    def show_info_dialog(self, info_text):
        """Show information dialog."""
        msg = QMessageBox()
        msg.setWindowTitle("Information")
        msg.setText(info_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
        
    def init_ui(self):
        """Set up the user interface."""
        # Set window properties
        self.setWindowTitle("SafeVision Content Filter v2.0")
        self.setMinimumSize(1400, 1000)
        self.resize(1600, 1100)
        
        # Set light theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
                color: #212529;
            }
            QWidget {
                background-color: #ffffff;
                color: #212529;
            }
            QTabWidget::pane {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                color: #495057;
                padding: 10px 20px;
                margin-right: 2px;
                border-radius: 4px 4px 0 0;
                border: 1px solid #dee2e6;
            }
            QTabBar::tab:selected {
                background-color: #667eea;
                color: #ffffff;
                border-color: #667eea;
            }
            QTabBar::tab:hover:!selected {
                background-color: #f1f3f4;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #667eea;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header with title and status
        header_widget = self.create_header()
        main_layout.addWidget(header_widget)
        
        # Main content - horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #dee2e6;
                width: 3px;
                margin: 2px;
                border-radius: 1px;
            }
            QSplitter::handle:hover {
                background-color: #4CAF50;
            }
        """)
        
        # Left panel - configuration and controls (30% of width)
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - preview and output (70% of width)
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 900])  # Left: 400px, Right: 900px
        main_splitter.setChildrenCollapsible(False)
        
        main_layout.addWidget(main_splitter)
        
        # Add status bar at bottom
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.file_info_label = QLabel("No file selected")
        
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.file_info_label)
        
        self.setCentralWidget(main_widget)
        
    def create_header(self):
        """Create the header section."""
        header_widget = QWidget()
        header_widget.setFixedHeight(70)  # Increased from 60 to 70
        header_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #667eea, stop:1 #764ba2);
                border-radius: 8px;
                border: 2px solid #667eea;
            }
        """)
        
        layout = QHBoxLayout(header_widget)
        layout.setContentsMargins(20, 12, 20, 12)  # Adjusted margins
        
        # Title and logo
        title_layout = QVBoxLayout()
        title_label = QLabel("SafeVision Content Filter")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff; border: none;")
        
        subtitle_label = QLabel("AI-Powered Content Detection & Filtering")
        subtitle_label.setFont(QFont("Arial", 11))
        subtitle_label.setStyleSheet("color: #e8ecf7; border: none;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        layout.addLayout(title_layout)
        
        layout.addStretch()
        
        return header_widget
        
    def create_left_panel(self):
        """Create the left configuration panel."""
        left_widget = QWidget()
        left_widget.setMaximumWidth(450)
        left_widget.setMinimumWidth(350)
        left_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 2px solid #dee2e6;
            }
        """)
        
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # File input section
        file_section = self.create_file_input_section()
        layout.addWidget(file_section)
        
        # Options tabs
        options_tabs = QTabWidget()
        options_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                color: #495057;
                padding: 8px 16px;
                margin-right: 1px;
                border-radius: 6px 6px 0 0;
                min-width: 80px;
                border: 1px solid #dee2e6;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: #ffffff;
                border-color: #4CAF50;
            }
            QTabBar::tab:hover:!selected {
                background-color: #f1f3f4;
            }
        """)
        
        # Basic options tab
        basic_tab = self.create_basic_options_tab()
        options_tabs.addTab(basic_tab, "⚙️ Basic")
        
        # Settings tab (new organization)
        settings_tab = self.create_settings_tab()
        options_tabs.addTab(settings_tab, "🛠️ Settings")
        
        # Advanced options tab  
        advanced_tab = self.create_advanced_options_tab()
        options_tabs.addTab(advanced_tab, "🔧 Advanced")
        
        layout.addWidget(options_tabs)
        
        # Process button and progress
        process_section = self.create_process_section()
        layout.addWidget(process_section)
        
        return left_widget
        
    def create_file_input_section(self):
        """Create the file input section."""
        section = QGroupBox("📁 File Input")
        section.setFixedHeight(250)  # Increased to 220 for more height
        section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #667eea;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        
        layout = QVBoxLayout(section)
        layout.setSpacing(12)  # Increased spacing
        layout.setContentsMargins(10, 25, 10, 12)  # Better margins with more space
        
        # Drop area with full available size
        self.drop_area = DropArea()
        self.drop_area.setFixedHeight(175)  # Increased to use full available space
        layout.addWidget(self.drop_area)
        
        return section
        
    def create_process_section(self):
        """Create the process section with button and progress."""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setSpacing(10)
        
        # Process button
        self.process_button = QPushButton("🚀 Start Processing")
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                border-radius: 8px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #5a6fd8;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_file)
        layout.addWidget(self.process_button)
        
        # Modern progress bar
        self.progress_bar = ModernProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.hide()
        
        clear_button = QPushButton("🗑️ Clear Log")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #666;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #777;
            }
        """)
        clear_button.clicked.connect(self.clear_output)
        
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(clear_button)
        layout.addLayout(control_layout)
        
        return section
        
    def create_right_panel(self):
        """Create the right preview and output panel."""
        right_widget = QWidget()
        right_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 2px solid #dee2e6;
            }
        """)
        
        layout = QVBoxLayout(right_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Vertical splitter for preview and log sections
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.setHandleWidth(8)
        self.right_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #dee2e6;
                border: 1px solid #adb5bd;
                border-radius: 3px;
            }
            QSplitter::handle:hover {
                background-color: #667eea;
            }
        """)
        
        # Preview tabs (top section - 60% of height)
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        # Global media preview type selector
        global_preview_layout = QHBoxLayout()
        global_preview_layout.addWidget(QLabel("Global Media Preview Type:"))
        self.global_preview_type_combo = QComboBox()
        self.global_preview_type_combo.addItems(["Dialog", "Included"])
        self.global_preview_type_combo.setCurrentText(self.settings.get("global_preview_type", "Included"))
        self.global_preview_type_combo.setFixedHeight(25)
        self.global_preview_type_combo.currentTextChanged.connect(self.on_global_preview_type_changed)
        global_preview_layout.addWidget(self.global_preview_type_combo)
        global_preview_layout.addStretch()
        preview_layout.addLayout(global_preview_layout)
        
        self.preview_tabs = SplittableTabWidget()
        self.preview_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                color: #495057;
                padding: 10px 20px;
                margin-right: 1px;
                border-radius: 6px 6px 0 0;
                min-width: 100px;
                border: 1px solid #dee2e6;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: #ffffff;
                border-color: #4CAF50;
            }
            QTabBar::tab:hover:!selected {
                background-color: #f1f3f4;
            }
        """)
        
        # Input preview tab with preview type selector
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        input_layout.setContentsMargins(10, 10, 10, 10)
        
        # Input-specific preview type selector
        input_preview_layout = QHBoxLayout()
        input_preview_layout.addWidget(QLabel("Preview Type:"))
        self.input_preview_type_combo = QComboBox()
        self.input_preview_type_combo.addItems(["Dialog", "Included"])
        self.input_preview_type_combo.setCurrentText(self.settings.get("input_preview_type", "Included"))
        self.input_preview_type_combo.setFixedHeight(25)
        self.input_preview_type_combo.currentTextChanged.connect(self.on_input_preview_type_changed)
        input_preview_layout.addWidget(self.input_preview_type_combo)
        input_preview_layout.addStretch()
        input_layout.addLayout(input_preview_layout)
        
        self.input_preview = PreviewWidget()
        input_layout.addWidget(self.input_preview)
        
        self.preview_tabs.addTab(input_tab, "📥 Input Preview")
        
        # Output preview tab with preview type selector
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.setContentsMargins(10, 10, 10, 10)
        
        # Output-specific preview type selector
        output_preview_layout = QHBoxLayout()
        output_preview_layout.addWidget(QLabel("Preview Type:"))
        self.output_preview_type_combo = QComboBox()
        self.output_preview_type_combo.addItems(["Dialog", "Included"])
        self.output_preview_type_combo.setCurrentText(self.settings.get("output_preview_type", "Included"))
        self.output_preview_type_combo.setFixedHeight(25)
        self.output_preview_type_combo.currentTextChanged.connect(self.on_output_preview_type_changed)
        output_preview_layout.addWidget(self.output_preview_type_combo)
        output_preview_layout.addStretch()
        output_layout.addLayout(output_preview_layout)
        
        self.output_preview = MultiPreviewWidget(self)
        output_layout.addWidget(self.output_preview)
        
        self.preview_tabs.addTab(output_tab, "📤 Output Preview")
        
        # Tree view tab for file explorer with preview type selector
        tree_tab = QWidget()
        tree_layout = QVBoxLayout(tree_tab)
        tree_layout.setContentsMargins(10, 10, 10, 10)
        
        # Media preview type selector for tree view
        preview_type_layout = QHBoxLayout()
        preview_type_layout.addWidget(QLabel("Media Preview Type:"))
        self.preview_type_combo = QComboBox()
        self.preview_type_combo.addItems(["Dialog", "Included"])
        self.preview_type_combo.setCurrentText(self.settings.get("tree_preview_type", "Included"))
        self.preview_type_combo.setFixedHeight(25)
        self.preview_type_combo.currentTextChanged.connect(self.on_preview_type_changed)
        preview_type_layout.addWidget(self.preview_type_combo)
        preview_type_layout.addStretch()
        tree_layout.addLayout(preview_type_layout)
        
        self.tree_view = FileTreeWidget()
        tree_layout.addWidget(self.tree_view)
        
        self.preview_tabs.addTab(tree_tab, "📁 TreeView")
        
        # Embedded media preview tab (conditionally shown)
        self.media_preview_tab = QWidget()
        media_preview_layout = QVBoxLayout(self.media_preview_tab)
        media_preview_layout.setContentsMargins(10, 10, 10, 10)
        
        # Default message when no file is selected
        self.media_preview_label = QLabel("Select a media file from TreeView to preview here")
        self.media_preview_label.setAlignment(Qt.AlignCenter)
        self.media_preview_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 16px;
                padding: 40px;
                border: 2px dashed #dee2e6;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
        """)
        media_preview_layout.addWidget(self.media_preview_label)
        
        # Embedded media viewer (initially hidden)
        self.embedded_viewer = None
        
        # Add the tab but show/hide based on tree preview setting
        self.media_preview_tab_index = self.preview_tabs.addTab(self.media_preview_tab, "🎬 Media Preview")
        
        # Set initial visibility based on loaded settings
        tree_preview_type = self.settings.get("tree_preview_type", "Included")
        if tree_preview_type == "Included":
            self.preview_tabs.setTabVisible(self.media_preview_tab_index, True)
            self.tree_view.file_selected_for_preview = self.show_embedded_preview
        else:
            self.preview_tabs.setTabVisible(self.media_preview_tab_index, False)
            self.tree_view.file_selected_for_preview = None
        
        preview_layout.addWidget(self.preview_tabs)
        self.right_splitter.addWidget(preview_container)
        
        # Output log section (bottom section - 40% of height)
        log_section = self.create_log_section()
        self.right_splitter.addWidget(log_section)
        
        # Set initial sizes (preview: 60%, log: 40%)
        self.right_splitter.setSizes([600, 400])
        self.right_splitter.setStretchFactor(0, 1)  # Preview section stretches
        self.right_splitter.setStretchFactor(1, 0)  # Log section has fixed preference
        
        layout.addWidget(self.right_splitter)
        
        return right_widget
        
    def create_log_section(self):
        """Create the output log section."""
        section = QGroupBox("📋 Processing Log")
        section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #4CAF50;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # Enhanced output log
        self.output_log = ColoredTextEdit()
        self.output_log.setMinimumHeight(200)
        layout.addWidget(self.output_log)
        
        return section
        
    def create_settings_tab(self):
        """Create the settings tab with organized options."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Processing Options Group
        process_group = QGroupBox("⚙️ Processing Options")
        process_group.setFixedHeight(120)  # Fixed height
        process_layout = QVBoxLayout(process_group)
        
        # Processing checkboxes
        checkbox_layout = QVBoxLayout()
        checkbox_layout.setSpacing(5)
        
        self.preserve_originals_cb = QCheckBox("📥 Preserve Original Files")
        self.preserve_originals_cb.setFixedHeight(25)  # Fixed height
        self.preserve_originals_cb.setChecked(True)
        checkbox_layout.addWidget(self.preserve_originals_cb)
        
        self.overwrite_mode_cb = QCheckBox("🔄 Overwrite Existing Files")
        self.overwrite_mode_cb.setFixedHeight(25)  # Fixed height
        checkbox_layout.addWidget(self.overwrite_mode_cb)
        
        self.debug_mode_cb = QCheckBox("🐛 Enable Debug Mode")
        self.debug_mode_cb.setFixedHeight(25)  # Fixed height
        checkbox_layout.addWidget(self.debug_mode_cb)
        
        process_layout.addLayout(checkbox_layout)
        layout.addWidget(process_group)
        
        # Output Settings Group
        output_group = QGroupBox("📤 Output Settings")
        output_group.setFixedHeight(100)  # Fixed height
        output_layout = QVBoxLayout(output_group)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("Output Dir:")
        output_dir_label.setFixedHeight(25)  # Fixed height
        output_dir_label.setFixedWidth(80)  # Fixed width
        output_dir_layout.addWidget(output_dir_label)
        self.output_settings_dir_edit = QLineEdit()
        self.output_settings_dir_edit.setPlaceholderText("Default output directory")
        self.output_settings_dir_edit.setFixedHeight(25)  # Fixed height
        output_browse = QPushButton("📁")
        output_browse.setFixedSize(30, 25)
        output_browse.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_settings_dir_edit)
        output_dir_layout.addWidget(output_browse)
        output_layout.addLayout(output_dir_layout)
        
        # Filename pattern
        pattern_layout = QHBoxLayout()
        pattern_label = QLabel("File Pattern:")
        pattern_label.setFixedHeight(25)  # Fixed height
        pattern_label.setFixedWidth(80)  # Fixed width
        pattern_layout.addWidget(pattern_label)
        self.filename_pattern_edit = QLineEdit()
        self.filename_pattern_edit.setPlaceholderText("{original_name}_processed.{ext}")
        self.filename_pattern_edit.setFixedHeight(25)  # Fixed height
        pattern_layout.addWidget(self.filename_pattern_edit)
        output_layout.addLayout(pattern_layout)
        
        layout.addWidget(output_group)
        
        # Script Settings Group (moved from Advanced)
        script_group = QGroupBox("🔍 Script Settings")
        script_group.setFixedHeight(180)  # Fixed height
        script_layout = QVBoxLayout(script_group)
        
        # Add header with info button
        header_layout = QHBoxLayout()
        script_label = QLabel("Automatically detect processing scripts")
        script_label.setFixedHeight(20)  # Fixed height
        header_layout.addWidget(script_label)
        header_layout.addStretch()
        script_info = self.create_info_button(
            "Script Auto-Detection:\n\n"
            "• Searches for main.py or safevision_image.exe for image processing\n"
            "• Searches for video.py or safevision_video.exe for video processing\n"
            "• Looks in current directory and 'detect' subdirectory\n"
            "• Manual browsing available if auto-detection fails"
        )
        header_layout.addWidget(script_info)
        script_layout.addLayout(header_layout)
        
        # Detection status
        self.detection_status_label = QLabel("🔄 Detecting scripts...")
        self.detection_status_label.setFixedHeight(20)  # Fixed height
        self.detection_status_label.setStyleSheet("color: #6c757d; font-style: italic;")
        script_layout.addWidget(self.detection_status_label)
        
        # Main script path
        main_script_layout = QHBoxLayout()
        main_label = QLabel("Image Script:")
        main_label.setFixedHeight(25)  # Fixed height
        main_label.setFixedWidth(100)  # Fixed width
        main_script_layout.addWidget(main_label)
        self.main_script_edit = QLineEdit()
        self.main_script_edit.setPlaceholderText("Auto-detected or browse for main.py/safevision_image.exe")
        self.main_script_edit.setFixedHeight(25)  # Fixed height
        self.main_script_browse = QPushButton("📁")
        self.main_script_browse.setFixedSize(30, 25)
        self.main_script_browse.clicked.connect(self.browse_main_script)
        main_script_layout.addWidget(self.main_script_edit)
        main_script_layout.addWidget(self.main_script_browse)
        script_layout.addLayout(main_script_layout)
        
        # Video script path
        video_script_layout = QHBoxLayout()
        video_label = QLabel("Video Script:")
        video_label.setFixedHeight(25)  # Fixed height
        video_label.setFixedWidth(100)  # Fixed width
        video_script_layout.addWidget(video_label)
        self.video_script_edit = QLineEdit()
        self.video_script_edit.setPlaceholderText("Auto-detected or browse for video.py/safevision_video.exe")
        self.video_script_edit.setFixedHeight(25)  # Fixed height
        self.video_script_browse = QPushButton("📁")
        self.video_script_browse.setFixedSize(30, 25)
        self.video_script_browse.clicked.connect(self.browse_video_script)
        video_script_layout.addWidget(self.video_script_edit)
        video_script_layout.addWidget(self.video_script_browse)
        script_layout.addLayout(video_script_layout)
        
        # Auto-detect button
        self.auto_detect_button = QPushButton("🔄 Re-detect Scripts")
        self.auto_detect_button.setFixedHeight(30)  # Fixed height
        self.auto_detect_button.clicked.connect(self.auto_detect_scripts)
        script_layout.addWidget(self.auto_detect_button)
        
        layout.addWidget(script_group)
        
        # FFmpeg Settings Group (moved from Advanced)
        ffmpeg_group = QGroupBox("🎥 FFmpeg Settings")
        ffmpeg_group.setFixedHeight(120)  # Fixed height
        ffmpeg_layout = QVBoxLayout(ffmpeg_group)
        
        # Add header with info button
        ffmpeg_header_layout = QHBoxLayout()
        ffmpeg_label = QLabel("FFmpeg configuration for video processing")
        ffmpeg_label.setFixedHeight(20)  # Fixed height
        ffmpeg_header_layout.addWidget(ffmpeg_label)
        ffmpeg_header_layout.addStretch()
        ffmpeg_info = self.create_info_button(
            "FFmpeg Settings:\n\n"
            "• Auto-detects FFmpeg installation from system PATH\n"
            "• Searches common installation directories\n"
            "• Required for video processing operations\n"
            "• Download from: https://ffmpeg.org/download.html\n"
            "• Leave empty to use system PATH"
        )
        ffmpeg_header_layout.addWidget(ffmpeg_info)
        ffmpeg_layout.addLayout(ffmpeg_header_layout)
        
        # FFmpeg status
        self.ffmpeg_status_label = QLabel("🔄 Detecting FFmpeg...")
        self.ffmpeg_status_label.setFixedHeight(20)  # Fixed height
        self.ffmpeg_status_label.setStyleSheet("color: #6c757d; font-style: italic;")
        ffmpeg_layout.addWidget(self.ffmpeg_status_label)
        
        ffmpeg_path_layout = QHBoxLayout()
        ffmpeg_path_label = QLabel("FFmpeg Path:")
        ffmpeg_path_label.setFixedHeight(25)  # Fixed height
        ffmpeg_path_label.setFixedWidth(100)  # Fixed width
        ffmpeg_path_layout.addWidget(ffmpeg_path_label)
        self.ffmpeg_path_edit = QLineEdit()
        self.ffmpeg_path_edit.setPlaceholderText("Auto-detected or specify custom path")
        self.ffmpeg_path_edit.setFixedHeight(25)  # Fixed height
        self.ffmpeg_browse_button = QPushButton("📁")
        self.ffmpeg_browse_button.setFixedSize(30, 25)
        self.ffmpeg_browse_button.clicked.connect(self.browse_ffmpeg)
        self.ffmpeg_detect_button = QPushButton("🔄")
        self.ffmpeg_detect_button.setFixedSize(30, 25)
        self.ffmpeg_detect_button.clicked.connect(self.auto_detect_ffmpeg)
        self.ffmpeg_detect_button.setToolTip("Auto-detect FFmpeg")
        ffmpeg_path_layout.addWidget(self.ffmpeg_path_edit)
        ffmpeg_path_layout.addWidget(self.ffmpeg_browse_button)
        ffmpeg_path_layout.addWidget(self.ffmpeg_detect_button)
        ffmpeg_layout.addLayout(ffmpeg_path_layout)
        
        layout.addWidget(ffmpeg_group)
        
        # UI Control Group (expanded)
        ui_group = QGroupBox("🎨 UI Control")
        ui_group.setFixedHeight(120)  # Increased height
        ui_layout = QVBoxLayout(ui_group)
        
        # Restore UI button
        self.restore_ui_button = QPushButton("🔄 Restore Original UI Layout")
        self.restore_ui_button.setFixedHeight(30)  # Fixed height
        self.restore_ui_button.clicked.connect(self.restore_original_ui)
        self.restore_ui_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        ui_layout.addWidget(self.restore_ui_button)
        
        # Theme controls (example of additional UI controls)
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setFixedHeight(25)
        theme_layout.addWidget(theme_label)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText("Light")
        self.theme_combo.setFixedHeight(25)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        ui_layout.addLayout(theme_layout)
        
        layout.addWidget(ui_group)
        
        # Auto-detect everything on tab creation
        QTimer.singleShot(100, self.auto_detect_all)
        
        layout.addStretch()
        return tab
    
    def create_basic_options_tab(self):
        """Create the basic options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Process mode group
        process_group = QGroupBox("Process Mode")
        process_group.setFixedHeight(80)  # Fixed height
        process_layout = QVBoxLayout(process_group)
        
        self.blur_radio = QRadioButton("Apply Blur/Mask")
        self.blur_radio.setFixedHeight(25)  # Fixed height
        self.boxes_radio = QRadioButton("Show Detection Boxes")
        self.boxes_radio.setFixedHeight(25)  # Fixed height
        self.blur_radio.setChecked(True)
        
        process_layout.addWidget(self.blur_radio)
        process_layout.addWidget(self.boxes_radio)
        layout.addWidget(process_group)
        
        # Blur options
        blur_group = QGroupBox("Blur Options")
        blur_group.setFixedHeight(190)  # Fixed height
        blur_layout = QVBoxLayout(blur_group)
        
        self.enhanced_blur_toggle = ModernToggle("Enhanced Blur (Stronger)")
        self.enhanced_blur_toggle.setFixedHeight(30)  # Fixed height
        blur_layout.addWidget(self.enhanced_blur_toggle)
        
        self.solid_color_toggle = ModernToggle("Use Solid Color Instead of Blur")
        self.solid_color_toggle.setFixedHeight(30)  # Fixed height
        blur_layout.addWidget(self.solid_color_toggle)
        
        color_layout = QHBoxLayout()
        color_label = QLabel("Mask Color:")
        color_label.setFixedHeight(25)  # Fixed height
        color_layout.addWidget(color_label)
        self.color_button = ColorPickerButton([0, 0, 0])  # Black default
        self.color_button.setFixedHeight(25)  # Fixed height
        color_layout.addWidget(self.color_button)
        color_layout.addStretch()
        blur_layout.addLayout(color_layout)

        shape_layout = QHBoxLayout()
        shape_label = QLabel("Mask Shape:")
        shape_label.setFixedHeight(25)
        shape_layout.addWidget(shape_label)
        self.mask_shape_combo = QComboBox()
        self.mask_shape_combo.addItems(["rectangle", "ellipse"])
        self.mask_shape_combo.setCurrentText(self.settings.get("mask_shape", "rectangle"))
        self.mask_shape_combo.setFixedHeight(25)
        shape_layout.addWidget(self.mask_shape_combo)
        shape_layout.addStretch()
        blur_layout.addLayout(shape_layout)

        strength_layout = QHBoxLayout()
        strength_label = QLabel("Blur Strength:")
        strength_label.setFixedHeight(25)
        strength_layout.addWidget(strength_label)
        self.blur_strength_spin = QSpinBox()
        self.blur_strength_spin.setRange(3, 151)
        self.blur_strength_spin.setSingleStep(2)
        self.blur_strength_spin.setValue(int(self.settings.get("blur_strength", 23)))
        self.blur_strength_spin.setFixedHeight(25)
        strength_layout.addWidget(self.blur_strength_spin)
        strength_layout.addStretch()
        blur_layout.addLayout(strength_layout)
        
        layout.addWidget(blur_group)
        
        # Video options
        video_group = QGroupBox("Video Options")
        video_group.setFixedHeight(120)  # Fixed height
        video_layout = QVBoxLayout(video_group)
        
        self.include_audio_toggle = ModernToggle("Include Audio in Output")
        self.include_audio_toggle.setFixedHeight(30)  # Fixed height
        video_layout.addWidget(self.include_audio_toggle)
        
        codec_layout = QHBoxLayout()
        codec_label = QLabel("Codec:")
        codec_label.setFixedHeight(25)  # Fixed height
        codec_layout.addWidget(codec_label)
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["mp4v", "avc1", "xvid", "mjpg"])
        self.codec_combo.setFixedHeight(25)  # Fixed height
        codec_layout.addWidget(self.codec_combo)
        codec_layout.addStretch()
        video_layout.addLayout(codec_layout)
        
        self.delete_frames_toggle = ModernToggle("Delete Frames After Processing")
        self.delete_frames_toggle.setFixedHeight(30)  # Fixed height
        self.delete_frames_toggle.setChecked(True)
        video_layout.addWidget(self.delete_frames_toggle)
        
        layout.addWidget(video_group)

        # Analysis and marker export options
        analysis_group = QGroupBox("Analysis / Editor Markers")
        analysis_group.setFixedHeight(170)
        analysis_layout = QVBoxLayout(analysis_group)

        self.analyze_only_toggle = ModernToggle("Analyze Only (No Rendered Video)")
        self.analyze_only_toggle.setFixedHeight(28)
        self.analyze_only_toggle.setChecked(self.settings.get("analyze_only", False))
        analysis_layout.addWidget(self.analyze_only_toggle)

        self.save_report_toggle = ModernToggle("Write JSON/CSV Detection Reports")
        self.save_report_toggle.setFixedHeight(28)
        self.save_report_toggle.setChecked(self.settings.get("save_report", False))
        analysis_layout.addWidget(self.save_report_toggle)

        report_layout = QHBoxLayout()
        report_layout.addWidget(QLabel("Report:"))
        self.report_formats_combo = QComboBox()
        self.report_formats_combo.addItems(["json,csv", "json", "csv"])
        self.report_formats_combo.setCurrentText(self.settings.get("report_formats", "json,csv"))
        report_layout.addWidget(self.report_formats_combo)
        analysis_layout.addLayout(report_layout)

        marker_layout = QHBoxLayout()
        marker_layout.addWidget(QLabel("Markers:"))
        self.marker_export_combo = QComboBox()
        self.marker_export_combo.addItems(["none", "edl", "fcpxml", "both"])
        self.marker_export_combo.setCurrentText(self.settings.get("export_markers", "none") or "none")
        marker_layout.addWidget(self.marker_export_combo)
        analysis_layout.addLayout(marker_layout)

        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Marker Gap:"))
        self.marker_gap_spin = QDoubleSpinBox()
        self.marker_gap_spin.setRange(0.1, 30.0)
        self.marker_gap_spin.setSingleStep(0.25)
        self.marker_gap_spin.setValue(float(self.settings.get("marker_gap", 1.0)))
        self.marker_gap_spin.setSuffix(" sec")
        gap_layout.addWidget(self.marker_gap_spin)
        analysis_layout.addLayout(gap_layout)

        layout.addWidget(analysis_group)
        
        # Output directory
        output_group = QGroupBox("Output Directory")
        output_group.setFixedHeight(80)  # Fixed height
        output_layout = QVBoxLayout(output_group)
        
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Leave empty for default output directory")
        self.output_dir_edit.setFixedHeight(25)  # Fixed height
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.setFixedHeight(25)  # Fixed height
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        output_layout.addLayout(output_dir_layout)
        
        layout.addWidget(output_group)
        
        layout.addStretch()
        return tab
        
    def on_preview_type_changed(self, preview_type):
        """Handle changes to media preview type."""
        # Get the tab widget by finding the parent container
        current_widget = self.preview_type_combo.parent()
        while current_widget and not isinstance(current_widget, QTabWidget):
            current_widget = current_widget.parent()
        
        if current_widget and preview_type == "Included":
            # Show the media preview tab
            current_widget.setTabVisible(self.media_preview_tab_index, True)
            # Connect tree view to embedded preview
            self.tree_view.file_selected_for_preview = self.show_embedded_preview
        elif current_widget:
            # Hide the media preview tab
            current_widget.setTabVisible(self.media_preview_tab_index, False)
            # Reset tree view to dialog preview
            self.tree_view.file_selected_for_preview = None
    
    def show_embedded_preview(self, file_path):
        """Show media file in embedded preview tab."""
        try:
            # Clear previous viewer
            if self.embedded_viewer:
                self.embedded_viewer.setParent(None)
                self.embedded_viewer = None
            
            # Hide the default label
            self.media_preview_label.hide()
            
            # Create new embedded viewer
            self.embedded_viewer = ImageVideoViewer(file_path, embedded=True)
            self.media_preview_tab.layout().addWidget(self.embedded_viewer)
            
            # Switch to media preview tab
            current_widget = self.preview_type_combo.parent()
            while current_widget and not isinstance(current_widget, QTabWidget):
                current_widget = current_widget.parent()
            if current_widget:
                current_widget.setCurrentIndex(self.media_preview_tab_index)
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Failed to preview file:\n{str(e)}")
            # Show the default label again
            self.media_preview_label.show()

    def on_global_preview_type_changed(self, preview_type):
        """Handle global preview type change - applies to all tabs."""
        self.input_preview_type_combo.setCurrentText(preview_type)
        self.output_preview_type_combo.setCurrentText(preview_type)
        self.preview_type_combo.setCurrentText(preview_type)
        
        # Save settings immediately
        self._save_settings()
        
    def on_input_preview_type_changed(self, preview_type):
        """Handle input preview type change."""
        # Update input preview behavior based on selection
        # Save settings immediately
        self._save_settings()
        
    def on_output_preview_type_changed(self, preview_type):
        """Handle output preview type change."""
        # Update output preview behavior based on selection  
        # Save settings immediately
        self._save_settings()

    def on_preview_type_changed(self, preview_type):
        """Handle changes to media preview type."""
        # Get the tab widget by finding the parent container
        current_widget = self.preview_type_combo.parent()
        while current_widget and not isinstance(current_widget, (QTabWidget, SplittableTabWidget)):
            current_widget = current_widget.parent()
        
        if current_widget and preview_type == "Included":
            # Show the media preview tab
            current_widget.setTabVisible(self.media_preview_tab_index, True)
            # Connect tree view to embedded preview
            self.tree_view.file_selected_for_preview = self.show_embedded_preview
        elif current_widget:
            # Hide the media preview tab
            current_widget.setTabVisible(self.media_preview_tab_index, False)
            # Reset tree view to dialog preview
            self.tree_view.file_selected_for_preview = None
            
        # Save settings immediately
        self._save_settings()
    
    def show_media_preview(self, file_path):
        """Show media file in embedded preview tab. Used by output tab and tree view."""
        # Clear previous viewer
        if hasattr(self, 'embedded_viewer') and self.embedded_viewer:
            self.embedded_viewer.setParent(None)
            self.embedded_viewer.deleteLater()
            self.embedded_viewer = None
        
        try:
            # Hide the default label
            self.media_preview_label.hide()
            
            # Create new embedded viewer
            self.embedded_viewer = ImageVideoViewer(file_path, embedded=True)
            self.media_preview_tab.layout().addWidget(self.embedded_viewer)
            
            # Switch to media preview tab - handles both regular and split views
            current_widget = self.preview_tabs
            if self.preview_tabs.is_split:
                # Handle split view - use the active split
                if self.preview_tabs.active_split == "left":
                    current_widget = self.preview_tabs
                elif hasattr(self.preview_tabs, 'split_widget') and self.preview_tabs.split_widget:
                    current_widget = self.preview_tabs.split_widget
            
            # Ensure the media preview tab is visible
            if current_widget:
                current_widget.setTabVisible(self.media_preview_tab_index, True)
                current_widget.setCurrentIndex(self.media_preview_tab_index)
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Failed to preview file:\n{str(e)}")
            # Show the default label again
            self.media_preview_label.show()
            
    def show_embedded_preview(self, file_path):
        """Show media file in embedded preview tab."""
        # Use the common media preview method
        self.show_media_preview(file_path)
    
    def restore_original_ui(self):
        """Restore the original UI layout."""
        try:
            # Restore splitter sizes
            if hasattr(self, 'main_splitter'):
                self.main_splitter.setSizes([400, 900])  # Original left/right ratio
            if hasattr(self, 'right_splitter'):
                self.right_splitter.setSizes([600, 400])  # Original preview/log ratio
            
            # Restore any split views
            if hasattr(self, 'preview_tabs') and self.preview_tabs.is_split:
                self.preview_tabs.restore_single_view()
            
            # Reset preview type to loaded setting (don't force to Dialog)
            saved_global_type = self.settings.get("global_preview_type", "Included")
            if hasattr(self, 'global_preview_type_combo'):
                self.global_preview_type_combo.setCurrentText(saved_global_type)
            
            QMessageBox.information(self, "UI Restored", "Original UI layout has been restored!")
            
        except Exception as e:
            QMessageBox.warning(self, "Restore Error", f"Failed to restore UI: {str(e)}")

    def create_advanced_options_tab(self):
        """Create the advanced options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Monitoring rules
        monitor_group = QGroupBox("📊 Full Blur Monitoring Rules")
        monitor_group.setFixedHeight(160)  # Fixed height
        monitor_layout = QVBoxLayout(monitor_group)
        
        # Add header with info button
        monitor_header_layout = QHBoxLayout()
        monitor_label = QLabel("Configure automatic full blur activation")
        monitor_label.setFixedHeight(20)  # Fixed height
        monitor_header_layout.addWidget(monitor_label)
        monitor_header_layout.addStretch()
        monitor_info = self.create_info_button(
            "Full Blur Monitoring Rules:\n\n"
            "• Percentage Threshold: Minimum percentage of content to trigger full blur\n"
            "• Frame Count Threshold: Number of consecutive frames needed\n"
            "• Full Blur Rule: Labels per frames ratio (e.g., 2/10 means 2 labels in any 10 frames)\n"
            "• Used with -r parameter for video processing"
        )
        monitor_header_layout.addWidget(monitor_info)
        monitor_layout.addLayout(monitor_header_layout)
        
        # Percentage rule
        percent_layout = QHBoxLayout()
        percent_label = QLabel("Percentage Threshold:")
        percent_label.setFixedHeight(25)  # Fixed height
        percent_label.setFixedWidth(150)  # Fixed width
        percent_layout.addWidget(percent_label)
        self.percent_spin = QDoubleSpinBox()
        self.percent_spin.setRange(0, 100)
        self.percent_spin.setSingleStep(0.5)
        self.percent_spin.setValue(10.0)
        self.percent_spin.setSuffix("%")
        self.percent_spin.setFixedHeight(25)  # Fixed height
        self.percent_spin.setMinimumWidth(80)  # Fixed minimum width
        percent_layout.addWidget(self.percent_spin)
        percent_layout.addStretch()
        monitor_layout.addLayout(percent_layout)
        
        # Count rule
        count_layout = QHBoxLayout()
        count_label = QLabel("Frame Count Threshold:")
        count_label.setFixedHeight(25)  # Fixed height
        count_label.setFixedWidth(150)  # Fixed width
        count_layout.addWidget(count_label)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(0, 10000)
        self.count_spin.setValue(5)
        self.count_spin.setFixedHeight(25)  # Fixed height
        self.count_spin.setMinimumWidth(80)  # Fixed minimum width
        count_layout.addWidget(self.count_spin)
        count_layout.addStretch()
        monitor_layout.addLayout(count_layout)
        
        # Full blur labels rule
        fbr_layout = QHBoxLayout()
        fbr_label = QLabel("Full Blur Rule (labels/frames):")
        fbr_label.setFixedHeight(25)  # Fixed height
        fbr_label.setFixedWidth(150)  # Fixed width
        fbr_layout.addWidget(fbr_label)
        self.fbr_labels_spin = QSpinBox()
        self.fbr_labels_spin.setRange(1, 1000)
        self.fbr_labels_spin.setValue(2)
        self.fbr_labels_spin.setFixedHeight(25)  # Fixed height
        self.fbr_labels_spin.setMinimumWidth(60)  # Fixed minimum width
        fbr_layout.addWidget(self.fbr_labels_spin)
        slash_label = QLabel("/")
        slash_label.setFixedHeight(25)  # Fixed height
        fbr_layout.addWidget(slash_label)
        self.fbr_frames_spin = QSpinBox()
        self.fbr_frames_spin.setRange(1, 10000)
        self.fbr_frames_spin.setValue(10)
        self.fbr_frames_spin.setFixedHeight(25)  # Fixed height
        self.fbr_frames_spin.setMinimumWidth(60)  # Fixed minimum width
        fbr_layout.addWidget(self.fbr_frames_spin)
        fbr_layout.addStretch()
        monitor_layout.addLayout(fbr_layout)
        
        layout.addWidget(monitor_group)
        
        # Rule File Detection Group
        rule_group = QGroupBox("📄 Rule File Detection")
        rule_group.setFixedHeight(120)  # Fixed height
        rule_layout = QVBoxLayout(rule_group)
        
        # Add header with info button
        rule_header_layout = QHBoxLayout()
        rule_label = QLabel("Automatic .rule file detection")
        rule_label.setFixedHeight(20)  # Fixed height
        rule_header_layout.addWidget(rule_label)
        rule_header_layout.addStretch()
        rule_info = self.create_info_button(
            "Rule File Detection:\n\n"
            "• Automatically detects .rule files in script directory\n"
            "• Used for custom blur/detection rules\n"
            "• Files should have .rule extension\n"
            "• Applied automatically when found"
        )
        rule_header_layout.addWidget(rule_info)
        rule_layout.addLayout(rule_header_layout)
        
        # Rule file status
        self.rule_status_label = QLabel("🔄 Detecting rule files...")
        self.rule_status_label.setFixedHeight(20)  # Fixed height
        self.rule_status_label.setStyleSheet("color: #6c757d; font-style: italic;")
        rule_layout.addWidget(self.rule_status_label)
        
        # Rule file path
        rule_file_layout = QHBoxLayout()
        rule_file_label = QLabel("Rule File:")
        rule_file_label.setFixedHeight(25)  # Fixed height
        rule_file_label.setFixedWidth(100)  # Fixed width
        rule_file_layout.addWidget(rule_file_label)
        self.rule_file_edit = QLineEdit()
        self.rule_file_edit.setPlaceholderText("Auto-detected or browse for .rule file")
        self.rule_file_edit.setFixedHeight(25)  # Fixed height
        self.rule_file_browse = QPushButton("📁")
        self.rule_file_browse.setFixedSize(30, 25)
        self.rule_file_browse.clicked.connect(self.browse_rule_file)
        rule_file_layout.addWidget(self.rule_file_edit)
        rule_file_layout.addWidget(self.rule_file_browse)
        rule_layout.addLayout(rule_file_layout)
        
        layout.addWidget(rule_group)
        
        # Parameter Enable/Disable Group
        param_group = QGroupBox("⚙️ Parameter Control")
        param_group.setFixedHeight(120)  # Fixed height
        param_layout = QVBoxLayout(param_group)
        
        # Add header with info button
        param_header_layout = QHBoxLayout()
        param_label = QLabel("Enable/disable specific processing parameters")
        param_label.setFixedHeight(20)  # Fixed height
        param_header_layout.addWidget(param_label)
        param_header_layout.addStretch()
        param_info = self.create_info_button(
            "Parameter Control:\n\n"
            "• Enable/disable specific command line parameters\n"
            "• -r: Monitoring rules parameter\n"
            "• -fbr: Full blur rule parameter\n"
            "• -a: Audio inclusion parameter\n"
            "• -df: Delete frames parameter\n"
            "• All enabled by default"
        )
        param_header_layout.addWidget(param_info)
        param_layout.addLayout(param_header_layout)
        
        # Parameter checkboxes
        param_grid = QGridLayout()
        
        self.enable_r_param = QCheckBox("Enable -r (Monitoring Rules)")
        self.enable_r_param.setFixedHeight(25)  # Fixed height
        self.enable_r_param.setChecked(True)
        param_grid.addWidget(self.enable_r_param, 0, 0)
        
        self.enable_fbr_param = QCheckBox("Enable -fbr (Full Blur Rule)")
        self.enable_fbr_param.setFixedHeight(25)  # Fixed height
        self.enable_fbr_param.setChecked(True)
        param_grid.addWidget(self.enable_fbr_param, 0, 1)
        
        self.enable_a_param = QCheckBox("Enable -a (Audio)")
        self.enable_a_param.setFixedHeight(25)  # Fixed height
        self.enable_a_param.setChecked(True)
        param_grid.addWidget(self.enable_a_param, 1, 0)
        
        self.enable_df_param = QCheckBox("Enable -df (Delete Frames)")
        self.enable_df_param.setFixedHeight(25)  # Fixed height
        self.enable_df_param.setChecked(True)
        param_grid.addWidget(self.enable_df_param, 1, 1)
        
        param_layout.addLayout(param_grid)
        layout.addWidget(param_group)
        
        layout.addStretch()
        return tab
        
    def browse_file(self):
        """Browse for input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Media File", "", 
            "Media Files (*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if file_path:
            self.load_file(file_path)
            
    def load_file(self, file_path):
        """Load a file for processing."""
        if os.path.exists(file_path):
            self.current_file = file_path
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            self.file_info_label.setText(f"Selected: {filename}\nSize: {file_size:.1f} MB")
            self.process_button.setEnabled(True)
            
            # Show input preview
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Show image preview
                self.input_preview.setImage(file_path)
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                # Show first frame of video
                try:
                    cap = open_video_capture(file_path)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        self.input_preview.setQImage(qt_image)
                    cap.release()
                except Exception as e:
                    self.input_preview.setText("Video preview failed")
            
            self.output_log.append_info(f"Loaded file: {file_path}")
        else:
            QMessageBox.warning(self, "Error", "File does not exist!")
            
    def browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            
    def browse_ffmpeg(self):
        """Browse for FFmpeg executable."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select FFmpeg Executable", "", 
            "Executable Files (*.exe);;All Files (*)"
        )
        if file_path:
            self.ffmpeg_path_edit.setText(file_path)
            
    def build_command(self):
        """Build the command line arguments based on GUI settings."""
        if not self.current_file:
            return None
            
        file_ext = Path(self.current_file).suffix.lower()
        
        # Determine which script to use
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Use main.py for images
            script_path = self.get_main_script_path()
            if script_path.endswith('.exe'):
                command = [script_path]
            else:
                command = [sys.executable, script_path]
            
            # Input file
            command.extend(["-i", self.current_file])
            
            # Output directory
            if self.output_dir_edit.text().strip():
                output_path = os.path.join(self.output_dir_edit.text().strip(), 
                                         f"processed_{os.path.basename(self.current_file)}")
                command.extend(["-o", output_path])
            
            # Blur option
            if self.blur_radio.isChecked():
                command.append("-b")

            if self.solid_color_toggle.isChecked():
                command.append("--color")
                color = self.color_button.get_color()
                command.extend(["--mask-color", f"{color[0]},{color[1]},{color[2]}"])

            command.extend(["--mask-shape", self.mask_shape_combo.currentText()])
            command.extend(["--blur-strength", str(self.blur_strength_spin.value())])
                
            # Full blur rule for images
            if self.fbr_labels_spin.value() > 0 and self.enable_fbr_param.isChecked():
                command.extend(["-fbr", str(self.fbr_labels_spin.value())])
                
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            # Use video.py for videos
            script_path = self.get_video_script_path()
            if script_path.endswith('.exe'):
                command = [script_path]
            else:
                command = [sys.executable, script_path]
            
            # Input file
            command.extend(["-i", self.current_file])
            
            # Output directory
            if self.output_dir_edit.text().strip():
                command.extend(["-vo", self.output_dir_edit.text().strip()])
            
            # Process mode
            if self.analyze_only_toggle.isChecked():
                command.append("--analyze-only")
            elif self.boxes_radio.isChecked():
                command.append("-b")
                if self.blur_radio.isChecked():
                    command.append("--blur")
            
            # Audio
            if self.include_audio_toggle.isChecked() and self.enable_a_param.isChecked():
                command.append("-a")
                
            # Codec
            command.extend(["-c", self.codec_combo.currentText()])
            
            # Delete frames
            if self.delete_frames_toggle.isChecked() and self.enable_df_param.isChecked():
                command.append("-df")
                
            # Enhanced blur
            if self.enhanced_blur_toggle.isChecked():
                command.append("--enhanced-blur")
                
            # Solid color
            if self.solid_color_toggle.isChecked():
                command.append("--color")
                color = self.color_button.get_color()
                command.extend(["--mask-color", f"{color[0]},{color[1]},{color[2]}"])

            command.extend(["--mask-shape", self.mask_shape_combo.currentText()])
            command.extend(["--blur-strength", str(self.blur_strength_spin.value())])

            if self.analyze_only_toggle.isChecked() or self.save_report_toggle.isChecked():
                command.append("--save-report")
            command.extend(["--report-formats", self.report_formats_combo.currentText()])

            marker_format = self.marker_export_combo.currentText()
            if marker_format != "none":
                command.extend(["--export-markers", marker_format])
                command.extend(["--marker-gap", str(self.marker_gap_spin.value())])
                
            # Monitoring rules
            if (self.percent_spin.value() > 0 or self.count_spin.value() > 0) and self.enable_r_param.isChecked():
                rule = f"{self.percent_spin.value()}/{self.count_spin.value()}"
                command.extend(["-r", rule])
                
            # Full blur rule
            if (self.fbr_labels_spin.value() > 0 and self.fbr_frames_spin.value() > 0) and self.enable_fbr_param.isChecked():
                fbr_rule = f"{self.fbr_labels_spin.value()}/{self.fbr_frames_spin.value()}"
                command.extend(["-fbr", fbr_rule])
                
            # FFmpeg path
            ffmpeg_path = self.ffmpeg_path_edit.text().strip()
            if ffmpeg_path and ffmpeg_path != 'ffmpeg':
                command.extend(["--ffmpeg-path", ffmpeg_path])
        else:
            return None
            
        return command
        
    def process_file(self):
        """Start processing the file."""
        if not self.current_file:
            QMessageBox.warning(self, "Error", "No file selected!")
            return
            
        command = self.build_command()
        if not command:
            QMessageBox.warning(self, "Error", "Unsupported file format!")
            return
            
        # Clear output log and reset previews
        self.output_log.clear()
        self.output_preview.clear_previews()
        
        # Update UI for processing state
        self.process_button.setEnabled(False)
        self.stop_button.show()
        self.progress_bar.start_animation("Processing file...")
        self.status_label.setText("🔄 Processing...")
        
        # Log the command
        self.output_log.append_info("Starting processing...")
        if not self.hideCommand:
            self.output_log.append_colored(f"Command: {' '.join(command)}", "#6c757d")
        
        # Start worker thread
        working_dir = os.path.dirname(os.path.abspath(__file__))
        self.worker = CommandWorker(command, working_dir)
        self.worker.signals.output_line.connect(self.log_output)
        self.worker.signals.status.connect(self.update_status)
        self.worker.signals.finished.connect(self.processing_finished)
        self.worker.signals.frame_detected.connect(self.update_preview_frame)
        self.worker.start()
        
    def log_output(self, text):
        """Add text to the output log with smart coloring."""
        text = text.strip()
        if not text:
            return
            
        # Update progress bar text for meaningful messages
        if "Processing" in text and ("%" in text or "frames/s" in text):
            self.progress_bar.update_text(text[:50] + "..." if len(text) > 50 else text)
            
        # Categorize log messages and apply colors
        if "error" in text.lower() or "traceback" in text.lower() or "failed" in text.lower():
            self.output_log.append_error(text)
        elif "warning" in text.lower() or "warn" in text.lower():
            self.output_log.append_warning(text)
        elif "processing" in text.lower() and ("%" in text or "frames/s" in text):
            self.output_log.append_progress(text)
        elif "completed" in text.lower() or "saved" in text.lower() or "created" in text.lower():
            self.output_log.append_success(text)
        elif "frame" in text.lower() and "exposed" in text.lower():
            self.output_log.append_warning(text)
        elif text.startswith("Using") or text.startswith("Original") or text.startswith("Processing"):
            self.output_log.append_info(text)
        else:
            self.output_log.append_colored(text, "#ffffff")
            
    def update_preview_frame(self, frame_path):
        """Update the preview with the latest processed frame."""
        if os.path.exists(frame_path):
            # For real-time preview, we could add it to output preview
            # but for now, we'll just log it
            self.output_log.append_progress(f"Frame processed: {os.path.basename(frame_path)}")
        
    def update_status(self, status):
        """Update the status label."""
        self.status_label.setText(f"🔄 {status}")
        
    def processing_finished(self, success, message):
        """Handle processing completion."""
        self.process_button.setEnabled(True)
        self.stop_button.hide()
        self.progress_bar.stop_animation()
        
        if success:
            self.status_label.setText("✅ Processing completed successfully!")
            self.output_log.append_success("Processing completed successfully!")
            
            # Look for and display output files
            self.scan_and_display_outputs()
            
            QMessageBox.information(self, "Success", "Processing completed successfully!")
        else:
            self.status_label.setText("❌ Processing failed!")
            self.output_log.append_error(f"Processing failed: {message}")
            QMessageBox.warning(self, "Error", f"Processing failed: {message}")
            
        # Clean up worker
        if self.worker:
            self.worker.stop()
            self.worker = None
            
    def scan_and_display_outputs(self):
        """Scan for output files and display them in the preview."""
        try:
            if not self.current_file:
                return
                
            # Get input filename without extension
            input_name = Path(self.current_file).stem
            
            # Determine output directories based on log messages
            output_dirs = []
            
            # Default output directories
            if self.current_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                output_dirs = ["video_output", "video_outputs"]
            else:
                output_dirs = ["output", "Blur", "Prosses"]
            
            # Add custom output directory if specified
            custom_dir = self.output_dir_edit.text().strip()
            if custom_dir and os.path.exists(custom_dir):
                output_dirs.insert(0, custom_dir)
            
            found_files = []
            
            # Search for output files in all directories
            for output_dir in output_dirs:
                if not os.path.exists(output_dir):
                    continue
                    
                # Look for files that start with input name or contain it
                for ext in ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.json', '.csv', '.edl', '.fcpxml']:
                    # Pattern 1: input_name + something + extension (e.g., i1_Output.png)
                    pattern1 = os.path.join(output_dir, f"{input_name}*{ext}")
                    files1 = glob.glob(pattern1)
                    
                    # Pattern 2: files containing input name anywhere
                    pattern2 = os.path.join(output_dir, f"*{input_name}*{ext}")
                    files2 = glob.glob(pattern2)
                    
                    # Combine and deduplicate
                    all_files = list(set(files1 + files2))
                    
                    for file_path in all_files:
                        # Priority scoring based on directory and filename patterns
                        score = 0
                        filename = os.path.basename(file_path).lower()
                        
                        # Higher priority for certain directories
                        if "output" in output_dir.lower():
                            score += 10
                        elif "blur" in output_dir.lower():
                            score += 8
                        elif "prosses" in output_dir.lower():
                            score += 6
                        
                        # Higher priority for files with _Output pattern
                        if "_output" in filename:
                            score += 5
                        elif "_blur" in filename:
                            score += 4
                        elif "_detect" in filename:
                            score += 3
                        
                        # Avoid old detection files that aren't related
                        if filename.endswith(('_wm_detected.jpg', '_ocr_detected.jpg', '_detect.jpg')):
                            # Only include if they actually contain the input name
                            if input_name.lower() not in filename:
                                continue
                        
                        found_files.append((score, file_path, output_dir))
            
            if not found_files:
                self.output_log.append_warning("No output files found matching input name")
                return
            
            # Sort by score (highest first) and modification time
            found_files.sort(key=lambda x: (x[0], os.path.getmtime(x[1])), reverse=True)
            
            # Display up to 9 most relevant outputs (3x3 grid)
            displayed_count = 0
            for score, file_path, output_dir in found_files[:9]:
                if displayed_count >= 9:
                    break
                    
                relative_path = os.path.relpath(file_path)
                title = f"{os.path.basename(output_dir)}: {os.path.basename(file_path)}"
                
                self.output_preview.add_preview(file_path, title)
                self.output_log.append_success(f"Output file: {relative_path}")
                displayed_count += 1
            
            if displayed_count > 0:
                self.output_log.append_info(f"Displaying {displayed_count} output files in preview")
                    
        except Exception as e:
            self.output_log.append_error(f"Error scanning outputs: {str(e)}")
            
    def load_file(self, file_path):
        """Load a file for processing."""
        if os.path.exists(file_path):
            self.current_file = file_path
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            # Update file info in header
            self.file_info_label.setText(f"📁 {filename} ({file_size:.1f} MB)")
            self.process_button.setEnabled(True)
            
            # Display input preview
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Show image preview
                self.input_preview.setImage(file_path)
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                # Show first frame of video
                try:
                    cap = open_video_capture(file_path)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Convert BGR to RGB
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w, ch = rgb_frame.shape
                            bytes_per_line = ch * w
                            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                            self.input_preview.setQImage(qt_image)
                        else:
                            self.input_preview.setText("Could not read video frame")
                    else:
                        self.input_preview.setText("Could not open video file")
                    cap.release()
                except Exception as e:
                    self.input_preview.setText(f"Video preview failed:\n{str(e)}")
            else:
                self.input_preview.setText("Unsupported file format")
            
            # Clear previous outputs
            self.output_preview.clear_previews()
            self.output_log.clear()
            
            # Log file info
            self.output_log.append_info(f"Loaded file: {filename}")
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    img = cv2_imread(file_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        self.output_log.append_colored(f"Image dimensions: {w}x{h}", "#6c757d")
                except Exception:
                    pass
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                try:
                    cap = open_video_capture(file_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.output_log.append_colored(f"Video: {w}x{h}, {fps:.1f} fps, {duration:.1f}s", "#6c757d")
                    cap.release()
                except Exception:
                    pass
        else:
            QMessageBox.warning(self, "Error", "File does not exist!")
            
    def clear_output(self):
        """Clear the output log and previews."""
        self.output_log.clear()
        self.output_preview.clear_previews()
        self.output_log.append_info("Output log and previews cleared.")
        
    def stop_processing(self):
        """Stop the current processing."""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("⏹️ Processing stopped by user")
            self.output_log.append_warning("Processing stopped by user")
            self.process_button.setEnabled(True)
            self.stop_button.hide()
            self.progress_bar.stop_animation()
            
    def _load_settings(self):
        """Load settings from file."""
        settings_file = "safevision_settings.json"
        default_settings = {
            "output_dir": "",
            "ffmpeg_path": "",
            "codec": "mp4v",
            "delete_frames": True,
            "enhanced_blur": False,
            "solid_color": False,
            "mask_shape": "rectangle",
            "mask_color": [0, 0, 0],
            "blur_strength": 23,
            "analyze_only": False,
            "save_report": False,
            "report_formats": "json,csv",
            "export_markers": "none",
            "marker_gap": 1.0,
            "include_audio": False,
            "percent_threshold": 10.0,
            "count_threshold": 5,
            "fbr_labels": 2,
            "fbr_frames": 10,
            "global_preview_type": "Included",
            "input_preview_type": "Included", 
            "output_preview_type": "Included",
            "tree_preview_type": "Included"
        }
        
        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    for key, value in default_settings.items():
                        if key not in settings:
                            settings[key] = value
                    return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            
        return default_settings
        
    def _save_settings(self):
        """Save current settings to file."""
        settings = {
            "output_dir": self.output_dir_edit.text(),
            "ffmpeg_path": self.ffmpeg_path_edit.text(),
            "codec": self.codec_combo.currentText(),
            "delete_frames": self.delete_frames_toggle.isChecked(),
            "enhanced_blur": self.enhanced_blur_toggle.isChecked(),
            "solid_color": self.solid_color_toggle.isChecked(),
            "mask_shape": self.mask_shape_combo.currentText(),
            "mask_color": self.color_button.get_color(),
            "blur_strength": self.blur_strength_spin.value(),
            "analyze_only": self.analyze_only_toggle.isChecked(),
            "save_report": self.save_report_toggle.isChecked(),
            "report_formats": self.report_formats_combo.currentText(),
            "export_markers": self.marker_export_combo.currentText(),
            "marker_gap": self.marker_gap_spin.value(),
            "include_audio": self.include_audio_toggle.isChecked(),
            "percent_threshold": self.percent_spin.value(),
            "count_threshold": self.count_spin.value(),
            "fbr_labels": self.fbr_labels_spin.value(),
            "fbr_frames": self.fbr_frames_spin.value(),
            "global_preview_type": self.global_preview_type_combo.currentText(),
            "input_preview_type": self.input_preview_type_combo.currentText(),
            "output_preview_type": self.output_preview_type_combo.currentText(),
            "tree_preview_type": self.preview_type_combo.currentText()
        }
        
        try:
            with open("safevision_settings.json", 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
            
    def closeEvent(self, event):
        """Handle window close event."""
        self._save_settings()
        
        # Stop worker if running
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)  # Wait up to 3 seconds
            
        event.accept()
    
    def auto_detect_all(self):
        """Auto-detect all scripts and FFmpeg."""
        self.auto_detect_scripts()
        self.auto_detect_ffmpeg()
        self.auto_detect_rule_files()
        
    def auto_detect_rule_files(self):
        """Auto-detect .rule files in script directory."""
        self.rule_status_label.setText("🔄 Detecting rule files...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for .rule files
        rule_candidates = []
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if file.lower().endswith('.rule'):
                    rule_candidates.append(os.path.join(root, file))
        
        # Update UI
        if rule_candidates:
            # Use the first found rule file
            self.rule_file_edit.setText(rule_candidates[0])
            self.rule_status_label.setText(f"✅ Found {len(rule_candidates)} rule file(s)")
            self.rule_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.rule_status_label.setText("❌ No rule files found")
            self.rule_status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
    
    def browse_rule_file(self):
        """Browse for rule file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Rule File", "", 
            "Rule Files (*.rule);;All Files (*)"
        )
        if file_path:
            self.rule_file_edit.setText(file_path)
        
    def auto_detect_scripts(self):
        """Auto-detect image and video processing scripts."""
        self.detection_status_label.setText("🔄 Detecting scripts...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for main.py or safevision_image.exe
        main_candidates = [
            os.path.join(current_dir, "main.py"),
            os.path.join(current_dir, "safevision_image.exe"),
            os.path.join(current_dir, "detect", "main.py"),
            os.path.join(current_dir, "detect", "safevision_image.exe")
        ]
        
        main_script = None
        for candidate in main_candidates:
            if os.path.exists(candidate):
                main_script = candidate
                break
                
        # Look for video.py or safevision_video.exe
        video_candidates = [
            os.path.join(current_dir, "video.py"),
            os.path.join(current_dir, "safevision_video.exe"),
            os.path.join(current_dir, "detect", "video.py"),
            os.path.join(current_dir, "detect", "safevision_video.exe")
        ]
        
        video_script = None
        for candidate in video_candidates:
            if os.path.exists(candidate):
                video_script = candidate
                break
        
        # Update UI
        if main_script:
            self.main_script_edit.setText(main_script)
        if video_script:
            self.video_script_edit.setText(video_script)
            
        # Update status
        detected_count = sum([1 for x in [main_script, video_script] if x])
        if detected_count == 2:
            self.detection_status_label.setText("✅ All scripts detected successfully")
            self.detection_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        elif detected_count == 1:
            self.detection_status_label.setText("⚠️ Partial detection - some scripts found")
            self.detection_status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        else:
            self.detection_status_label.setText("❌ No scripts found - manual configuration required")
            self.detection_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
    
    def auto_detect_ffmpeg(self):
        """Auto-detect FFmpeg installation."""
        self.ffmpeg_status_label.setText("🔄 Detecting FFmpeg...")
        
        ffmpeg_path = None
        
        # First, check if ffmpeg is in PATH
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                ffmpeg_path = 'ffmpeg'  # Available in PATH
        except:
            pass
        
        # If not in PATH, search common locations
        if not ffmpeg_path:
            common_paths = [
                r'C:\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg.exe'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'ffmpeg.exe')
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
        
        # Update UI
        if ffmpeg_path:
            self.ffmpeg_path_edit.setText(ffmpeg_path)
            if ffmpeg_path == 'ffmpeg':
                self.ffmpeg_status_label.setText("✅ FFmpeg found in system PATH")
            else:
                self.ffmpeg_status_label.setText(f"✅ FFmpeg found: {os.path.basename(ffmpeg_path)}")
            self.ffmpeg_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.ffmpeg_status_label.setText("❌ FFmpeg not found - install FFmpeg or specify path")
            self.ffmpeg_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
    
    def browse_main_script(self):
        """Browse for main script file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image Processing Script", "", 
            "Python/Executable Files (*.py *.exe);;All Files (*)"
        )
        if file_path:
            self.main_script_edit.setText(file_path)
    
    def browse_video_script(self):
        """Browse for video script file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video Processing Script", "", 
            "Python/Executable Files (*.py *.exe);;All Files (*)"
        )
        if file_path:
            self.video_script_edit.setText(file_path)
    
    def get_main_script_path(self):
        """Get the path to the main processing script."""
        custom_path = self.main_script_edit.text().strip()
        if custom_path and os.path.exists(custom_path):
            return custom_path
        # Fallback to current logic
        return os.path.join(os.path.dirname(__file__), "main.py")
    
    def get_video_script_path(self):
        """Get the path to the video processing script."""
        custom_path = self.video_script_edit.text().strip()
        if custom_path and os.path.exists(custom_path):
            return custom_path
        # Fallback to current logic
        return os.path.join(os.path.dirname(__file__), "video.py")


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("SafeVision")
    app.setApplicationVersion("2.0")
    
    # Create and show main window
    window = SafeVisionGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
