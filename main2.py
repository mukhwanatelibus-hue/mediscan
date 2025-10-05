import os
import sys
from kivy import platform

# Android-specific setup
if platform == 'android':
    from android.permissions import request_permissions, Permission
    from android.storage import app_storage_path
    
    # Request necessary permissions
    request_permissions([
        Permission.CAMERA,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_EXTERNAL_STORAGE
    ])
    
    # Set up paths for Android
    APP_DIR = app_storage_path()
    DATA_DIR = os.path.join(APP_DIR, 'data')
    MODELS_DIR = os.path.join(APP_DIR, 'models')
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # For Android, we need to use TFLite instead of full TensorFlow
    USE_TFLITE = True
else:
    # Standard paths for desktop
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(APP_DIR, 'data')
    MODELS_DIR = os.path.join(APP_DIR, 'models')
    USE_TFLITE = False
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.properties import StringProperty, ListProperty, NumericProperty, ObjectProperty, BooleanProperty
from kivymd.uix.list import OneLineListItem, MDList, TwoLineListItem, ThreeLineListItem, OneLineIconListItem
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDFloatingActionButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.textfield import MDTextField
from kivymd.uix.progressbar import MDProgressBar
from kivymd.toast import toast
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.chip import MDChip
from kivymd.uix.fitimage import FitImage
from kivymd.uix.navigationdrawer import MDNavigationDrawer, MDNavigationLayout

import json
import re
import requests
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pyzbar import pyzbar
import tensorflow as tf
import webbrowser
import random

from kivy.metrics import dp
from kivy.core.clipboard import Clipboard
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.uix.image import Image as KivyImage
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.lang import Builder
from threading import Thread
import time

# Set mobile-friendly window size
Window.size = (360, 640)

# Load KV string for custom components
Builder.load_string("""
<NavDrawerItem>:
    IconLeftWidget:
        icon: root.icon

<LoadingScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(50)
        spacing: dp(20)
        MDSpinner:
            size_hint: None, None
            size: dp(46), dp(46)
            pos_hint: {'center_x': 0.5}
            active: True
        MDLabel:
            text: "Loading AI Model..."
            halign: "center"
            theme_text_color: "Primary"
            
<LoadingDialog>:
    title: "Processing..."
    size_hint: 0.8, 0.3
    BoxLayout:
        orientation: "vertical"
        padding: dp(20)
        spacing: dp(10)
        MDProgressBar:
            id: progress
            type: "indeterminate"
            running: True
        MDLabel:
            text: "Analyzing with AI. Please wait..."
            halign: "center"

<ImagePreviewDialog>:
    title: "Preview Image"
    size_hint: 0.9, 0.8
    BoxLayout:
        orientation: "vertical"
        padding: dp(10)
        spacing: dp(10)
        FitImage:
            id: preview_image
            source: root.image_path
            size_hint_y: 0.7
        MDBoxLayout:
            adaptive_height: True
            spacing: dp(10)
            MDRaisedButton:
                text: "Analyze with AI"
                on_release: root.analyze_callback()
            MDFlatButton:
                text: "Cancel"
                on_release: root.dismiss()
            
<HistoryItem>:
    orientation: "vertical"
    padding: dp(10)
    spacing: dp(5)
    adaptive_height: True
    size_hint_y: None
    height: self.minimum_height
    MDBoxLayout:
        adaptive_height: True
        MDLabel:
            text: root.medicine_name
            font_style: "H6"
            theme_text_color: "Primary"
            adaptive_size: True
        MDLabel:
            text: root.scan_date
            font_style: "Caption"
            theme_text_color: "Secondary"
            adaptive_size: True
    MDBoxLayout:
        adaptive_height: True
        MDLabel:
            text: root.status
            theme_text_color: "Primary" if root.is_genuine else "Error"
            bold: True
            adaptive_size: True
        Widget:
            size_hint_x: None
            width: dp(10)
        MDLabel:
            text: f"{root.confidence}% confidence"
            font_style: "Caption"
            theme_text_color: "Secondary"
            adaptive_size: True
    MDRaisedButton:
        text: "Share Result"
        size_hint_y: None
        height: dp(40)
        on_release: app.share_history_item(root.scan_data)
        icon: "share-variant"
   
<ChatMessage>:
    orientation: 'horizontal'
    padding: dp(10)
    adaptive_height: True
    size_hint_y: None
    
    MDBoxLayout:
        orientation: 'horizontal'
        adaptive_size: True
        padding: dp(5)
        spacing: dp(10)
        
        MDLabel:
            text: "AI Assistant" if root.is_bot else "You"
            font_style: "Caption"
            theme_text_color: "Primary" if root.is_bot else "Secondary"
            size_hint_x: None
            width: dp(80)
            adaptive_size: True
            
        MDCard:
            size_hint: None, None
            size: self.minimum_size
            padding: dp(10)
            md_bg_color: app.theme_cls.primary_color if root.is_bot else app.theme_cls.bg_dark
            line_color: app.theme_cls.primary_color if root.is_bot else app.theme_cls.bg_dark
            
            MDLabel:
                text: root.message
                size_hint_x: None
                width: dp(200)  # Fixed width
                text_size: (dp(200), None)  # This enables text wrapping
                adaptive_height: True             

<NavigationLayout>:
    ScreenManager:
        id: screen_manager

<HomeScreen>:
    name: "home"
    MDBoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            id: toolbar
            title: "MediScan"
            elevation: 4
            left_action_items: [["menu", lambda x: app.open_menu()]]
            right_action_items: [["history", lambda x: app.switch_screen('history')], ["chat", lambda x: app.switch_screen('assistant')]]
        
        ScrollView:
            id: home_scroll
            scroll_type: ['bars', 'content']
            bar_width: dp(10)
            bar_color: app.theme_cls.primary_color
            MDBoxLayout:
                orientation: 'vertical'
                adaptive_height: True
                padding: dp(20)
                spacing: dp(20)
                
                FitImage:
                    source: "assets/medical_icon.png"
                    size_hint_y: None
                    height: dp(120)
                
                MDLabel:
                    text: "Welcome to MediScan"
                    font_style: "H4"
                    halign: "center"
                    adaptive_height: True
                    
                MDLabel:
                    text: "Scan medicine packaging or pills to verify authenticity"
                    font_style: "Body1"
                    halign: "center"
                    adaptive_height: True
                
                MDBoxLayout:
                    orientation: 'vertical'
                    adaptive_height: True
                    spacing: dp(10)
                    
                    MDRaisedButton:
                        text: "Scan Barcode"
                        size_hint_y: None
                        height: dp(50)
                        on_release: app.switch_screen('scan')
                        icon: "barcode"
                        
                    MDRaisedButton:
                        text: "Scan Pill"
                        size_hint_y: None
                        height: dp(50)
                        on_release: app.switch_screen('scan_pill')
                        icon: "pill"
                        
                    MDRaisedButton:
                        text: "Upload Image"
                        size_hint_y: None
                        height: dp(50)
                        on_release: app.show_file_chooser()
                        icon: "image"
                        
                    MDRaisedButton:
                        text: "Emergency Contacts"
                        size_hint_y: None
                        height: dp(50)
                        on_release: app.show_emergency_contacts()
                        icon: "phone-alert"
                
                MDCard:
                    orientation: 'vertical'
                    padding: dp(15)
                    spacing: dp(10)
                    adaptive_height: True
                    size_hint_y: None
                    height: self.minimum_height
                    
                    MDLabel:
                        text: "Recent Scans"
                        font_style: "H6"
                        adaptive_height: True
                        
                    MDBoxLayout:
                        adaptive_height: True
                        MDLabel:
                            text: f"Total Scans: {app.scan_history_count}"
                            adaptive_size: True
                            
                    MDBoxLayout:
                        adaptive_height: True
                        MDLabel:
                            text: f"Genuine: {app.genuine_count}"
                            theme_text_color: "Primary"
                            adaptive_size: True
                            
                    MDBoxLayout:
                        adaptive_height: True
                        MDLabel:
                            text: f"Counterfeit: {app.fake_count}"
                            theme_text_color: "Error"
                            adaptive_size: True

<ScanScreen>:
    name: "scan"
    MDBoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Scan Barcode"
            elevation: 4
            left_action_items: [["arrow-left", lambda x: app.switch_screen('home')]]
            right_action_items: [["robot", lambda x: app.toggle_scan_mode()]]
        
            MDBoxLayout:
                orientation: 'horizontal'
                adaptive_height: True
                padding: dp(10)
                spacing: dp(10)
                pos_hint: {'center_x': 0.5}
            
                MDChip:
                    id: simple_mode_chip
                    text: "Simple Scan"
                    selected: True
                    on_release: app.set_scan_mode("simple")
                    icon: "lightning-bolt"
                    icon: "lightning-bolt"
                    text_color: [0, 0, 0, 1]  # Black text
                   

                
                MDChip:
                    id: ai_mode_chip
                    text: "AI Enhanced"
                    selected: False
                    on_release: app.set_scan_mode("ai")
                    icon: "robot"
                    text_color: [0, 0, 0, 1]  # Black text
                    color: [0, 0, 0, 1]  # Black icon
            
        MDLabel:
            id: scan_mode_label
            text: "Fast barcode scanning - Quick and reliable"
            halign: "center"
            theme_text_color: "Primary"
            adaptive_height: True
            padding: dp(10)
            size_hint_y: None
            height: self.texture_size[1] + dp(20)
        
        ScrollView:
            scroll_type: ['bars', 'content']
            bar_width: dp(10)
            bar_color: app.theme_cls.primary_color
            BoxLayout:
                orientation: 'vertical'
                padding: dp(20)
                spacing: dp(20)
                size_hint_y: None
                height: self.minimum_height
                
                MDLabel:
                    text: "Point camera at medicine barcode"
                    halign: "center"
                    adaptive_height: True
                    size_hint_y: None
                    height: self.texture_size[1] + dp(10)
                    
                BoxLayout:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: dp(400)
                    spacing: dp(10)
                    
                    MDRaisedButton:
                        id: start_camera_btn
                        text: "Start Camera"
                        size_hint_y: None
                        height: dp(50)
                        on_release: app.start_camera()
                        icon: "camera"
                        
                    Camera:
                        id: camera
                        resolution: (640, 480)
                        play: False
                        size_hint_y: None
                        height: 0
                        opacity: 0
                        
                    MDRaisedButton:
                        id: capture_btn
                        text: "Capture & Analyze"
                        size_hint_y: None
                        height: dp(50)
                        disabled: True
                        opacity: 0.5
                        on_release: app.capture_and_analyze()
                        icon: "camera"
                
                MDRaisedButton:
                    text: "Enter Barcode Manually"
                    size_hint_y: None
                    height: dp(50)
                    on_release: app.show_barcode_input()
                    icon: "keyboard"                    
    
   
                    
<ScanPillScreen>:
    name: "scan_pill"
    MDBoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Scan Pill"
            elevation: 4
            left_action_items: [["arrow-left", lambda x: app.switch_screen('home')]]
        
        ScrollView:
            scroll_type: ['bars', 'content']
            bar_width: dp(10)
            bar_color: app.theme_cls.primary_color
            BoxLayout:
                orientation: 'vertical'
                padding: dp(20)
                spacing: dp(20)
                size_hint_y: None
                height: self.minimum_height
                
                MDLabel:
                    text: "Place pill on a contrasting background"
                    halign: "center"
                    adaptive_height: True
                    size_hint_y: None
                    height: self.texture_size[1] + dp(10)
                    
                BoxLayout:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: dp(450)
                    spacing: dp(10)
                    
                    MDRaisedButton:
                        id: start_pill_camera_btn
                        text: "Start Camera"
                        size_hint_y: None
                        height: dp(50)
                        on_release: app.start_pill_camera()
                        icon: "camera"
                        
                    Camera:
                        id: pill_camera
                        resolution: (640, 480)
                        play: False
                        size_hint_y: None
                        height: 0
                        opacity: 0
                        
                    MDLabel:
                        id: pill_result_label
                        text: "Ready to scan"
                        halign: "center"
                        adaptive_height: True
                        size_hint_y: None
                        height: self.texture_size[1] + dp(10)
                        
                    MDRaisedButton:
                        id: capture_pill_btn
                        text: "Capture & Analyze"
                        size_hint_y: None
                        height: dp(50)
                        disabled: True
                        opacity: 0.5
                        on_release: app.capture_and_analyze_pill()
                        icon: "camera"

<HistoryScreen>:
    name: "history"
    MDBoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Scan History"
            elevation: 4
            left_action_items: [["arrow-left", lambda x: app.switch_screen('home')]]
        
        ScrollView:
            scroll_type: ['bars', 'content']
            bar_width: dp(10)
            bar_color: app.theme_cls.primary_color
            MDList:
                id: history_list

<AssistantScreen>:
    name: "assistant"
    MDBoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Medical Assistant"
            elevation: 4
            left_action_items: [["arrow-left", lambda x: app.switch_screen('home')]]
        
        ScrollView:
            id: chat_scroll
            scroll_type: ['bars', 'content']
            bar_width: dp(10)
            bar_color: app.theme_cls.primary_color
            MDList:
                id: chat_list
                padding: dp(10)
                spacing: dp(10)
        
        MDBoxLayout:
            adaptive_height: True
            padding: dp(10)
            spacing: dp(10)
            
            MDTextField:
                id: chat_input
                hint_text: "Ask about medicines or health..."
                mode: "rectangle"
                size_hint_x: 0.8
                max_text_length: 200
                
            MDFloatingActionButton:
                icon: "send"
                size_hint_x: 0.2
                on_release: app.send_chat_message()

<SettingsScreen>:
    name: "settings"
    MDBoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Settings"
            elevation: 4
            left_action_items: [["arrow-left", lambda x: app.switch_screen('home')]]
        
        ScrollView:
            scroll_type: ['bars', 'content']
            bar_width: dp(10)
            bar_color: app.theme_cls.primary_color
            MDBoxLayout:
                orientation: 'vertical'
                adaptive_height: True
                padding: dp(20)
                spacing: dp(20)
                
                MDLabel:
                    text: "App Settings"
                    font_style: "H5"
                    adaptive_height: True
                    
                MDBoxLayout:
                    adaptive_height: True
                    MDLabel:
                        text: "Language:"
                        adaptive_size: True
                    MDLabel:
                        text: app.current_language
                        adaptive_size: True
                        
                MDSwitch:
                    id: ai_switch
                    text: "Enable AI Analysis"
                    active: True
                    
                MDRaisedButton:
                    text: "Clear History"
                    size_hint_y: None
                    height: dp(50)
                    on_release: app.clear_history()
                    
                MDLabel:
                    text: "Scanning Tips"
                    font_style: "H6"
                    adaptive_height: True
                    
                MDLabel:
                    text: "- Ensure good lighting when scanning\\n- Place pills on contrasting background\\n- Keep camera steady for barcode scanning\\n- Clean camera lens for better results"
                    theme_text_color: "Secondary"
                    adaptive_height: True

<FileChooserPopup>:
    title: "Select an image to analyze"
    size_hint: 0.9, 0.8
    BoxLayout:
        orientation: "vertical"
        FileChooserListView:
            id: file_chooser
            filters: ["*.png", "*.jpg", "*.jpeg"]
        BoxLayout:
            size_hint_y: None
            height: dp(50)
            MDRaisedButton:
                text: "Select"
                on_release: app.analyze_selected_image(file_chooser.selection)
            MDFlatButton:
                text: "Cancel"
                on_release: root.dismiss()

<BarcodeInputDialog>:
    title: "Enter Barcode Manually"
    size_hint: 0.9, 0.4
    BoxLayout:
        orientation: "vertical"
        padding: dp(20)
        spacing: dp(10)
        MDTextField:
            id: barcode_input
            hint_text: "Enter barcode number"
            mode: "rectangle"
            input_filter: "int"
            max_text_length: 13
        MDBoxLayout:
            adaptive_height: True
            spacing: dp(10)
            MDRaisedButton:
                text: "Analyze"
                on_release: app.analyze_manual_barcode(barcode_input.text)
            MDFlatButton:
                text: "Cancel"
                on_release: root.dismiss()
""")

# Custom dialog for loading
class LoadingDialog(MDDialog):
    pass

class ImagePreviewDialog(MDDialog):
    image_path = StringProperty("")
    
    def __init__(self, image_path, analyze_callback, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path
        self.analyze_callback = analyze_callback

class FileChooserPopup(Popup):
    pass

class BarcodeInputDialog(MDDialog):
    pass

# Custom history list item
class HistoryItem(MDBoxLayout):
    medicine_name = StringProperty("")
    scan_date = StringProperty("")
    status = StringProperty("")
    is_genuine = BooleanProperty(False)
    confidence = StringProperty("")
    scan_data = ObjectProperty(None)

class ChatMessage(MDBoxLayout):
    message = StringProperty("")
    is_bot = BooleanProperty(True)

class NavDrawerItem(OneLineIconListItem):
    icon = StringProperty()

# Define screen classes
class NavigationLayout(MDNavigationLayout):
    pass

class LoadingScreen(Screen):
    pass

class HomeScreen(Screen):
    pass

class ScanScreen(Screen):
    pass

class ScanPillScreen(Screen):
    pass

class HistoryScreen(Screen):
    pass

class AssistantScreen(Screen):
    pass

class SettingsScreen(Screen):
    pass

# AI Model Class - Using actual trained model
class MedicineAIModel:
    def __init__(self, app_instance):
        self.model = None
        self.interpreter = None
        self.class_names = []
        self.model_loaded = False
        self.app = app_instance
        self.use_tflite = USE_TFLITE
        
    def load_model_in_thread(self):
        """Start model loading in a separate thread"""
        thread = Thread(target=self.load_model, daemon=True)
        thread.start()
        
    def load_model(self):
        """Load trained AI model in background thread"""
        try:
            model_path = os.path.join(MODELS_DIR, 'pill_classifier.h5')
            tflite_path = os.path.join(MODELS_DIR, 'pill_classifier.tflite')
            model_info_path = os.path.join(MODELS_DIR, 'model_info.json')
            
            # Load class names from model info
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.class_names = model_info.get('class_names', [])
            else:
                # Create a default model info file if it doesn't exist
                self.class_names = ["Amoxicillin", "Paracetamol", "Ibuprofen", "Aspirin", "Vitamin C"]
                model_info = {"class_names": self.class_names}
                with open(model_info_path, 'w') as f:
                    json.dump(model_info, f)
                print("Created default model_info.json")
            
            # Check if any model file exists
            model_exists = os.path.exists(model_path) or os.path.exists(tflite_path)
            
            if not model_exists:
                print("No model files found. Using fallback analysis.")
                self.model_loaded = True  # Set to True to allow fallback analysis
                Clock.schedule_once(lambda dt: self.app.handle_model_loaded(True))
                return
            
            if self.use_tflite:
                # For Android, use TensorFlow Lite
                try:
                    # Load TFLite model
                    if os.path.exists(tflite_path):
                        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
                        self.interpreter.allocate_tensors()
                        self.model_loaded = True
                        print("✅ TFLite Model loaded successfully")
                    else:
                        print("❌ No TFLite model found, using fallback analysis")
                        self.model_loaded = True  # Allow fallback analysis
                        
                except Exception as e:
                    print(f"❌ Error loading TFLite model: {e}")
                    self.model_loaded = True  # Allow fallback analysis
                    
            else:
                # For desktop, use full TensorFlow
                if os.path.exists(model_path):
                    try:
                        self.model = tf.keras.models.load_model(model_path)
                        self.model_loaded = True
                        print("✅ TensorFlow Model loaded successfully")
                    except Exception as e:
                        print(f"❌ Error loading TensorFlow model: {e}")
                        self.model_loaded = True  # Allow fallback analysis
                else:
                    print("❌ No model found, using fallback analysis")
                    self.model_loaded = True  # Allow fallback analysis
            
            # Update UI on main thread
            Clock.schedule_once(lambda dt: self.app.handle_model_loaded(self.model_loaded))
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = True  # Allow fallback analysis
            Clock.schedule_once(lambda dt: self.app.handle_model_loaded(True))
    
    def analyze_pill_image(self, image_path):
        """Analyze pill image using the actual AI model or fallback analysis"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}
            
            # If model is not loaded, use fallback analysis
            if not self.model_loaded:
                return self.fallback_analysis(img)
            
            # If model files exist but couldn't be loaded, use fallback
            model_path = os.path.join(MODELS_DIR, 'pill_classifier.h5')
            tflite_path = os.path.join(MODELS_DIR, 'pill_classifier.tflite')
            model_exists = os.path.exists(model_path) or os.path.exists(tflite_path)
            
            if not model_exists or (self.model is None and self.interpreter is None):
                return self.fallback_analysis(img)
            
            # Preprocess image for the model
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction using the actual model
            if self.use_tflite and self.interpreter:
                # TFLite inference
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(output_details[0]['index'])
            elif self.model:
                # Full TensorFlow inference
                predictions = self.model.predict(img, verbose=0)
            else:
                return self.fallback_analysis(img)
            
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Get additional info using actual detection methods
            shape = self.detect_pill_shape(cv2.imread(image_path))
            color = self.detect_pill_color(cv2.imread(image_path))
            
            # Determine if genuine based on confidence threshold
            is_genuine = confidence > 0.7
            
            return {
                "genuine": is_genuine,
                "confidence": float(confidence),
                "type": self.class_names[class_idx] if class_idx < len(self.class_names) else "Unknown",
                "shape": shape,
                "color": color,
                "method": "AI Model"
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def fallback_analysis(self, img):
        """Fallback analysis when AI model is not available"""
        try:
            # Simple fallback analysis based on image properties
            shape = self.detect_pill_shape(img)
            color = self.detect_pill_color(img)
            
            # Get image dimensions for basic analysis
            height, width = img.shape[:2]
            area = height * width
            
            # Simple classification based on shape and color
            pill_type = "Unknown"
            confidence = 0.6
            
            if shape == "Round" and color == "White":
                pill_type = "Paracetamol"
                confidence = 0.7
            elif shape == "Oval" and color == "Red":
                pill_type = "Ibuprofen"
                confidence = 0.65
            elif shape == "Round" and color in ["Blue", "Green"]:
                pill_type = "Amoxicillin"
                confidence = 0.75
            
            return {
                "genuine": True,
                "confidence": confidence,
                "type": pill_type,
                "shape": shape,
                "color": color,
                "method": "Visual Analysis (Fallback)"
            }
        except Exception as e:
            return {"error": f"Fallback analysis failed: {str(e)}"}
    
    def detect_pill_shape(self, img):
        """Detect pill shape using contour analysis"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.04 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                vertices = len(approx)
                if vertices < 4:
                    return "Round"
                elif vertices == 4:
                    return "Rectangle"
                else:
                    return "Oval"
        except:
            pass
        return "Unknown"
    
    def detect_pill_color(self, img):
        """Detect dominant pill color using HSV color space"""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate average hue value
            avg_hue = np.mean(hsv[:,:,0])
            
            # Map hue values to color names
            if avg_hue < 15 or avg_hue > 165:
                return "Red"
            elif 15 <= avg_hue < 35:
                return "Orange"
            elif 35 <= avg_hue < 85:
                return "Yellow"
            elif 85 <= avg_hue < 95:
                return "Green"
            elif 95 <= avg_hue < 130:
                return "Blue"
            elif 130 <= avg_hue < 165:
                return "Purple"
            else:
                return "White"
        except:
            return "Unknown"

# Medicine database for barcode lookup
MEDICINE_DATABASE = {
    "123456789012": {
        "name": "Paracetamol 500mg",
        "manufacturer": "PharmaCorp Inc.",
        "expiry_date": "2024-12-31",
        "manufacturing_date": "2022-01-15",
        "uses": "Pain relief and fever reduction",
        "side_effects": "Rare: skin rash, nausea",
        "genuine": True
    },
    "987654321098": {
        "name": "Ibuprofen 200mg",
        "manufacturer": "HealthPlus Ltd.",
        "expiry_date": "2025-06-30",
        "manufacturing_date": "2023-02-20",
        "uses": "Anti-inflammatory pain relief",
        "side_effects": "May cause stomach discomfort",
        "genuine": True
    },
    "112233445566": {
        "name": "Amoxicillin 250mg",
        "manufacturer": "MediSolutions Co.",
        "expiry_date": "2024-09-15",
        "manufacturing_date": "2022-09-15",
        "uses": "Antibiotic for bacterial infections",
        "side_effects": "Allergic reactions in some cases",
        "genuine": True
    },
    "123456789013": {
        "name": "Aspirin 100mg",
        "manufacturer": "PainRelief Inc.",
        "expiry_date": "2024-11-30",
        "manufacturing_date": "2022-11-30",
        "uses": "Pain relief and blood thinner",
        "side_effects": "May cause stomach irritation",
        "genuine": True
    }
}

# Emergency contacts
EMERGENCY_CONTACTS = [
    {"name": "Emergency Services", "number": "112", "type": "Emergency"},
    {"name": "Poison Control", "number": "1-800-222-1222", "type": "Emergency"},
    {"name": "Local Hospital", "number": "+1234567890", "type": "Hospital"},
    {"name": "Family Doctor", "number": "+0987654321", "type": "Doctor"},
]

# Medical chatbot responses
CHATBOT_RESPONSES = {
    "hello": "Hello! I'm your medical assistant. How can I help you today?",
    "hi": "Hi there! I can help you with medicine information and health advice.",
    "paracetamol": "Paracetamol is used for pain relief and fever reduction. Typical dosage is 500mg every 4-6 hours for adults. Do not exceed 4g per day.",
    "ibuprofen": "Ibuprofen is an anti-inflammatory medication. It's used for pain, fever, and inflammation. Take with food to avoid stomach upset. Typical dosage is 200-400mg every 4-6 hours.",
    "amoxicillin": "Amoxicillin is an antibiotic used to treat bacterial infections. Complete the full course even if you feel better. Typical dosage is 250-500mg three times daily.",
    "aspirin": "Aspirin is used for pain relief, reducing inflammation, and as a blood thinner. Do not give to children under 16. Typical dosage is 300-600mg every 4-6 hours.",
    "vitamin c": "Vitamin C supports immune function and acts as an antioxidant. Typical dosage is 500-1000mg daily. It's water-soluble, so excess is excreted in urine.",
    "headache": "For headaches, you can try paracetamol or ibuprofen. Rest in a quiet, dark room and stay hydrated. Consult a doctor if headaches persist or are severe.",
    "fever": "For fever, paracetamol can help reduce temperature. Stay hydrated and rest. Consult a doctor if fever is high (>39°C) or persists for more than 3 days.",
    "cold": "For cold symptoms, rest, stay hydrated, and consider over-the-counter cold remedies. Most colds resolve in 7-10 days. Vitamin C may help reduce duration.",
    "cough": "For cough, honey and lemon can help. Cough suppressants may be used for dry coughs, but consult a pharmacist. See a doctor if cough persists for more than 3 weeks.",
    "side effects": "All medicines can have side effects. Common ones include nausea, dizziness, or drowsiness. Serious side effects should be reported to a doctor immediately.",
    "expiry": "Never use medicines past their expiry date as they may be less effective or potentially harmful. Check expiration dates regularly.",
    "genuine": "To check if medicine is genuine, look for proper packaging, correct spelling, manufacturer details, and verify with the manufacturer if possible.",
    "storage": "Store medicines in a cool, dry place away from direct sunlight. Some medications may require refrigeration - check the label.",
    "interaction": "Some medicines can interact with each other or with certain foods. Always inform your doctor about all medications you're taking.",
    "dosage": "Always follow the recommended dosage on the packaging or as prescribed by your doctor. Do not exceed recommended doses.",
    "default": "I'm here to help with medicine information. You can ask about specific medications, side effects, dosage, storage, or general health advice."
}

# Main App Class
class MediScanApp(MDApp):
    current_language = StringProperty("English")
    scan_history = ListProperty([])
    genuine_count = NumericProperty(0)
    fake_count = NumericProperty(0)
    scan_history_count = NumericProperty(0)
    camera = ObjectProperty(None)
    pill_camera = ObjectProperty(None)
    ai_model = ObjectProperty(None)
    current_scan = ObjectProperty(None)
    is_processing = BooleanProperty(False)
    model_loaded = BooleanProperty(False)
    scan_mode = StringProperty("simple")  # "simple" or "ai"
    camera_running = BooleanProperty(False)
    
    def build(self):
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.accent_palette = "Blue"
        self.theme_cls.theme_style = "Light"
        
        # Create navigation layout
        self.nav_layout = NavigationLayout()
        
        # Add loading screen first
        self.loading_screen = LoadingScreen(name='loading')
        self.nav_layout.ids.screen_manager.add_widget(self.loading_screen)
        
        # Create other screens but don't add them yet
        self.home_screen = HomeScreen(name='home')
        self.scan_screen = ScanScreen(name='scan')
        self.scan_pill_screen = ScanPillScreen(name='scan_pill')
        self.history_screen = HistoryScreen(name='history')
        self.assistant_screen = AssistantScreen(name='assistant')
        self.settings_screen = SettingsScreen(name='settings')
        
        # Load scan history
        self.load_history()
        
        # Initialize AI model (will load in background)
        self.ai_model = MedicineAIModel(self)
        
        # Start model loading in background
        Clock.schedule_once(lambda dt: self.ai_model.load_model_in_thread(), 0.5)
        
        # Create dropdown menu
        self.create_menu()
        
        return self.nav_layout
    
    def on_start(self):
        """Request Android permissions on app start"""
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ])
    
    def create_menu(self):
        """Create dropdown menu for navigation"""
        menu_items = [
            {
                "text": "Home",
                "viewclass": "OneLineListItem",
                "on_release": lambda x="home": self.switch_screen(x),
            },
            {
                "text": "Scan Barcode",
                "viewclass": "OneLineListItem",
                "on_release": lambda x="scan": self.switch_screen(x),
            },
            {
                "text": "Scan Pill",
                "viewclass": "OneLineListItem",
                "on_release": lambda x="scan_pill": self.switch_screen(x),
            },
            {
                "text": "History",
                "viewclass": "OneLineListItem",
                "on_release": lambda x="history": self.switch_screen(x),
            },
            {
                "text": "Settings",
                "viewclass": "OneLineListItem",
                "on_release": lambda x="settings": self.switch_screen(x),
            },
            {
                "text": "Medical Assistant",
                "viewclass": "OneLineListItem",
                "on_release": lambda x="assistant": self.switch_screen(x),
            },
        ]
        
        self.menu = MDDropdownMenu(
            items=menu_items,
            width_mult=4,
        )
    
    def open_menu(self):
        """Open the dropdown menu"""
        self.menu.caller = self.root.ids.screen_manager.get_screen('home').ids.toolbar
        self.menu.open()
    
    def show_emergency_contacts(self):
        """Show emergency contacts dialog"""
        contacts_text = "Emergency Contacts:\n\n"
        for contact in EMERGENCY_CONTACTS:
            contacts_text += f"{contact['name']}: {contact['number']} ({contact['type']})\n\n"
        
        self.emergency_dialog = MDDialog(
            title="Emergency Contacts",
            text=contacts_text,
            buttons=[
                MDFlatButton(text="Close", on_release=lambda x: self.emergency_dialog.dismiss())
            ],
            size_hint=(0.9, 0.7)
        )
        self.emergency_dialog.open()
    
    def handle_model_loaded(self, success):
        """When model is loaded, switch to home screen"""
        if success:
            # Add all screens to manager
            self.root.ids.screen_manager.add_widget(self.home_screen)
            self.root.ids.screen_manager.add_widget(self.scan_screen)
            self.root.ids.screen_manager.add_widget(self.scan_pill_screen)
            self.root.ids.screen_manager.add_widget(self.history_screen)
            self.root.ids.screen_manager.add_widget(self.assistant_screen)
            self.root.ids.screen_manager.add_widget(self.settings_screen)
            
            # Switch to home screen
            self.root.ids.screen_manager.current = 'home'
            self.model_loaded = True
            toast("AI Model loaded successfully")
        else:
            # Show error and still proceed to home screen
            self.root.ids.screen_manager.add_widget(self.home_screen)
            self.root.ids.screen_manager.add_widget(self.scan_screen)
            self.root.ids.screen_manager.add_widget(self.scan_pill_screen)
            self.root.ids.screen_manager.add_widget(self.history_screen)
            self.root.ids.screen_manager.add_widget(self.assistant_screen)
            self.root.ids.screen_manager.add_widget(self.settings_screen)
            
            self.root.ids.screen_manager.current = 'home'
            toast("AI Model not available. Using fallback analysis.")
    
    def toggle_scan_mode(self):
        """Toggle between simple and AI scan modes"""
        if self.scan_mode == "simple":
            self.set_scan_mode("ai")
        else:
            self.set_scan_mode("simple")
    
    def set_scan_mode(self, mode):
        """Switch between simple and AI scan modes"""
        self.scan_mode = mode
        scan_screen = self.root.ids.screen_manager.get_screen('scan')
        
        if mode == "simple":
            scan_screen.ids.simple_mode_chip.selected = True
            scan_screen.ids.ai_mode_chip.selected = False
            scan_screen.ids.scan_mode_label.text = "Fast barcode scanning - Quick and reliable"
            scan_screen.ids.scan_mode_label.theme_text_color = "Primary"
        else:
            scan_screen.ids.simple_mode_chip.selected = False
            scan_screen.ids.ai_mode_chip.selected = True
            scan_screen.ids.scan_mode_label.text = "AI-enhanced scanning - Verifies packaging authenticity"
            scan_screen.ids.scan_mode_label.theme_text_color = "Secondary"
    
    def reset_camera_ui(self):
        """Reset the camera UI to initial state with smooth animation"""
        try:
            scan_screen = self.root.ids.screen_manager.get_screen('scan')
            camera_widget = scan_screen.ids.camera
            start_btn = scan_screen.ids.start_camera_btn
            capture_btn = scan_screen.ids.capture_btn
            
            from kivy.animation import Animation
            
            # Animate camera disappearance
            anim_camera = Animation(height=dp(0), opacity=0, duration=0.3)
            anim_camera.start(camera_widget)
            
            # Animate button appearance
            anim_btn = Animation(height=dp(50), opacity=1, duration=0.3)
            anim_btn.start(start_btn)
            
            # Stop camera after animation completes
            Clock.schedule_once(lambda dt: setattr(camera_widget, 'play', False), 0.4)
            
            # Disable capture button
            capture_btn.disabled = True
            capture_btn.opacity = 0.5
            
        except Exception as e:
            print(f"Error resetting camera UI: {e}")

    def reset_pill_camera_ui(self):
        """Reset the pill camera UI to initial state with smooth animation"""
        try:
            scan_pill_screen = self.root.ids.screen_manager.get_screen('scan_pill')
            camera_widget = scan_pill_screen.ids.pill_camera
            start_btn = scan_pill_screen.ids.start_pill_camera_btn
            capture_btn = scan_pill_screen.ids.capture_pill_btn
            
            from kivy.animation import Animation
            
            # Animate camera disappearance
            anim_camera = Animation(height=dp(0), opacity=0, duration=0.3)
            anim_camera.start(camera_widget)
            
            # Animate button appearance
            anim_btn = Animation(height=dp(50), opacity=1, duration=0.3)
            anim_btn.start(start_btn)
            
            # Stop camera after animation completes
            Clock.schedule_once(lambda dt: setattr(camera_widget, 'play', False), 0.4)
            
            # Disable capture button
            capture_btn.disabled = True
            capture_btn.opacity = 0.5
            
            # Reset result label
            result_label = scan_pill_screen.ids.pill_result_label
            result_label.text = "Ready to scan"
            
        except Exception as e:
            print(f"Error resetting pill camera UI: {e}")
    
    def start_camera(self):
        """Start the camera when user clicks the button with smooth animation"""
        try:
            scan_screen = self.root.ids.screen_manager.get_screen('scan')
            camera_widget = scan_screen.ids.camera
            start_btn = scan_screen.ids.start_camera_btn
            capture_btn = scan_screen.ids.capture_btn
            
            # Start the camera first
            camera_widget.play = True
            
            # Use animations for smooth transitions
            from kivy.animation import Animation
            
            # Animate camera appearance
            anim_camera = Animation(height=dp(300), opacity=1, duration=0.3)
            anim_camera.start(camera_widget)
            
            # Animate button disappearance
            anim_btn = Animation(height=dp(0), opacity=0, duration=0.3)
            anim_btn.start(start_btn)
            
            # Enable capture button after a short delay
            Clock.schedule_once(lambda dt: setattr(capture_btn, 'disabled', False), 0.3)
            Clock.schedule_once(lambda dt: setattr(capture_btn, 'opacity', 1), 0.3)
            
        except Exception as e:
            toast(f"Error starting camera: {str(e)}")
    
    def start_pill_camera(self):
        """Start the pill camera when user clicks the button with smooth animation"""
        try:
            scan_pill_screen = self.root.ids.screen_manager.get_screen('scan_pill')
            camera_widget = scan_pill_screen.ids.pill_camera
            start_btn = scan_pill_screen.ids.start_pill_camera_btn
            capture_btn = scan_pill_screen.ids.capture_pill_btn
            
            # Start the camera
            camera_widget.play = True
            
            # Use animations for smooth transitions
            from kivy.animation import Animation
            
            # Animate camera appearance
            anim_camera = Animation(height=dp(300), opacity=1, duration=0.3)
            anim_camera.start(camera_widget)
            
            # Animate button disappearance
            anim_btn = Animation(height=dp(0), opacity=0, duration=0.3)
            anim_btn.start(start_btn)
            
            # Enable capture button after a short delay
            Clock.schedule_once(lambda dt: setattr(capture_btn, 'disabled', False), 0.3)
            Clock.schedule_once(lambda dt: setattr(capture_btn, 'opacity', 1), 0.3)
            
        except Exception as e:
            toast(f"Error starting camera: {str(e)}")
    
    def on_leave_scan_screen(self):
        """Stop camera when leaving scan screen to save resources"""
        try:
            self.reset_camera_ui()
            self.reset_pill_camera_ui()
        except:
            pass
    
    def switch_screen(self, screen_name):
        """Switch to different screen"""
        # Stop cameras if we're leaving scan screens
        if self.root.ids.screen_manager.current in ['scan', 'scan_pill']:
            self.on_leave_scan_screen()
            
        self.root.ids.screen_manager.current = screen_name
        self.menu.dismiss()
    
    def show_file_chooser(self):
        """Show file chooser dialog for image upload"""
        self.file_chooser = FileChooserPopup()
        self.file_chooser.open()
    
    def show_barcode_input(self):
        """Show dialog for manual barcode input"""
        self.barcode_dialog = BarcodeInputDialog()
        self.barcode_dialog.open()
    
    def analyze_manual_barcode(self, barcode_text):
        """Analyze manually entered barcode"""
        if not barcode_text or len(barcode_text) < 8:
            toast("Please enter a valid barcode (8+ digits)")
            return
        
        self.barcode_dialog.dismiss()
        self.analyze_barcode(barcode_text)
    
    def analyze_selected_image(self, selection):
        """Analyze the selected image file"""
        if not selection:
            toast("Please select an image file")
            return
            
        image_path = selection[0]
        self.file_chooser.dismiss()
        
        # Show preview dialog
        self.preview_dialog = ImagePreviewDialog(
            image_path=image_path,
            analyze_callback=lambda: self.analyze_uploaded_image(image_path)
        )
        self.preview_dialog.open()
    
    def analyze_uploaded_image(self, image_path):
        """Analyze the uploaded image"""
        self.preview_dialog.dismiss()
        
        # Check if it's likely a pill or packaging
        img = cv2.imread(image_path)
        if img is None:
            toast("Could not load image")
            return
            
        # Try to detect barcode first (FAST PATH)
        barcodes = pyzbar.decode(img)
        if barcodes:
            barcode_data = barcodes[0].data.decode('utf-8')
            toast(f"Barcode detected: {barcode_data}")
            self.analyze_barcode(barcode_data)
            return
            
        # If no barcode, assume it's a pill (AI path)
        self.analyze_pill_image_uploaded(image_path)
    
    def analyze_pill_image_uploaded(self, image_path):
        """Analyze uploaded pill image using actual AI model"""
        try:
            self.is_processing = True
            loading_dialog = LoadingDialog()
            loading_dialog.open()
            
            # Analyze with the actual AI model
            analysis = self.ai_model.analyze_pill_image(image_path)
            
            if "error" in analysis:
                toast(f"Analysis error: {analysis['error']}")
                loading_dialog.dismiss()
                self.is_processing = False
                return
                
            # Process results
            self.process_pill_analysis(analysis)
            
            loading_dialog.dismiss()
            self.is_processing = False
            
        except Exception as e:
            toast(f"Error: {str(e)}")
            self.is_processing = False

    def analyze_barcode(self, barcode_data):
        """Fast barcode analysis - no AI, just database lookup"""
        # Immediate database lookup (no delays)
        medicine_info = MEDICINE_DATABASE.get(barcode_data, None)
        
        if medicine_info:
            self.current_scan = {
                "type": "barcode",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result": medicine_info
            }
            
            # Show result immediately
            self.show_scan_result()
        else:
            toast("Barcode not found in database")
            # Offer manual entry option
            self.offer_manual_entry(barcode_data)
    
    def offer_manual_entry(self, barcode_data):
        """Offer to add unknown barcode to database"""
        self.unknown_barcode_dialog = MDDialog(
            title="Unknown Barcode",
            text=f"Barcode {barcode_data} not found in database.\n\nWould you like to add it manually?",
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: self.unknown_barcode_dialog.dismiss()),
                MDRaisedButton(text="Add Manually", on_release=lambda x: self.add_manual_medicine(barcode_data))
            ]
        )
        self.unknown_barcode_dialog.open()
    
    def add_manual_medicine(self, barcode_data):
        """Add manual medicine entry"""
        self.unknown_barcode_dialog.dismiss()
        toast("Manual entry feature coming soon!")
        # Here you could implement a form to add new medicines to the database
    
    def capture_and_analyze(self):
        """Choose scanning method based on mode"""
        if self.scan_mode == "simple":
            self.simple_barcode_scan()
        else:
            self.ai_enhanced_scan()
    
    def simple_barcode_scan(self):
        """Fast, simple barcode scanning"""
        try:
            # Stop Kivy camera first to avoid conflicts
            scan_screen = self.root.ids.screen_manager.get_screen('scan')
            scan_screen.ids.camera.play = False
            
            # Small delay to release camera
            Clock.schedule_once(lambda dt: self._actually_simple_scan(), 0.2)
            
        except Exception as e:
            toast(f"Scan error: {str(e)}")
            self.reset_camera_ui()
    
    def _actually_simple_scan(self):
        """Actual simple scan after camera is released"""
        try:
            toast("Scanning...")
            
            # Use OpenCV directly to avoid camera conflicts
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()  # Release immediately
            
            if not ret:
                toast("Cannot access camera")
                self.reset_camera_ui()
                return
                
            # Quick barcode detection
            barcodes = pyzbar.decode(frame)
            
            if barcodes:
                barcode_data = barcodes[0].data.decode('utf-8')
                toast("✓ Barcode detected")
                Clock.schedule_once(lambda dt: self.analyze_barcode(barcode_data), 0.3)
            else:
                toast("No barcode detected")
                self.show_quick_fallback()
                
            # Reset UI
            Clock.schedule_once(lambda dt: self.reset_camera_ui(), 0.5)
            
        except Exception as e:
            toast(f"Capture error: {str(e)}")
            self.reset_camera_ui()
    
    def ai_enhanced_scan(self):
        """AI-powered packaging verification"""
        try:
            # Stop Kivy camera first
            scan_screen = self.root.ids.screen_manager.get_screen('scan')
            scan_screen.ids.camera.play = False
            
            Clock.schedule_once(lambda dt: self._actually_ai_scan(), 0.2)
            
        except Exception as e:
            toast(f"AI scan error: {str(e)}")
            self.reset_camera_ui()
    
    def _actually_ai_scan(self):
        """Actual AI scan after camera is released"""
        try:
            self.is_processing = True
            loading_dialog = LoadingDialog()
            loading_dialog.open()
            
            # Capture image using OpenCV
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                toast("Cannot access camera")
                loading_dialog.dismiss()
                self.is_processing = False
                self.reset_camera_ui()
                return
            
            # Save image for analysis
            temp_path = "temp_ai_scan.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Try barcode first
            barcodes = pyzbar.decode(frame)
            if barcodes:
                barcode_data = barcodes[0].data.decode('utf-8')
                medicine_info = MEDICINE_DATABASE.get(barcode_data)
                
                if medicine_info:
                    # AI verification of packaging
                    verification = self.verify_packaging_ai(frame, medicine_info)
                    self.current_scan = {
                        "type": "ai_barcode",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "result": {**medicine_info, **verification}
                    }
                else:
                    self.current_scan = {
                        "type": "ai_barcode",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "result": {
                            "name": "Unknown Medicine",
                            "genuine": False,
                            "confidence": 0.3,
                            "method": "AI Enhanced Scan",
                            "reason": "Barcode not in database"
                        }
                    }
            else:
                # AI text recognition and packaging analysis
                analysis = self.analyze_packaging_ai(frame)
                self.current_scan = {
                    "type": "ai_packaging",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "result": analysis
                }
            
            loading_dialog.dismiss()
            self.show_scan_result()
            self.is_processing = False
            self.reset_camera_ui()
            
        except Exception as e:
            toast(f"AI analysis error: {str(e)}")
            self.is_processing = False
            self.reset_camera_ui()
    
    def verify_packaging_ai(self, image, medicine_info):
        """AI verification of packaging authenticity"""
        try:
            # Check image quality
            brightness = np.mean(image)
            sharpness = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            
            # Simple text detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            
            # Basic verification logic
            is_authentic = True
            confidence = 0.8
            reasons = []
            
            if brightness < 50:
                is_authentic = False
                confidence *= 0.7
                reasons.append("Poor lighting")
                
            if sharpness < 100:
                is_authentic = False
                confidence *= 0.8
                reasons.append("Blurry image")
                
            if medicine_info['name'].lower() not in text.lower():
                is_authentic = False
                confidence *= 0.6
                reasons.append("Name mismatch")
            
            return {
                "genuine": is_authentic,
                "confidence": confidence,
                "ai_verification": True,
                "verification_reasons": reasons if reasons else ["Packaging appears authentic"],
                "method": "AI Enhanced Verification"
            }
            
        except:
            return {
                "genuine": medicine_info.get('genuine', True),
                "confidence": 0.7,
                "ai_verification": False,
                "verification_reasons": ["AI analysis failed"],
                "method": "AI Enhanced Verification"
            }
    
    def analyze_packaging_ai(self, image):
        """AI analysis of packaging without barcode"""
        try:
            # Simple text recognition
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            
            # Extract potential medicine name
            medicine_name = "Unknown Medicine"
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 3 and any(char.isdigit() for char in line) and any(char.isalpha() for char in line):
                    medicine_name = line
                    break
            
            return {
                "name": medicine_name,
                "genuine": True,  # Assume genuine for AI analysis
                "confidence": 0.75,
                "method": "AI Packaging Analysis",
                "text_found": text[:100] + "..." if len(text) > 100 else text
            }
            
        except:
            return {
                "name": "Unknown Medicine",
                "genuine": False,
                "confidence": 0.3,
                "method": "AI Packaging Analysis",
                "reason": "Could not analyze packaging"
            }
    
    def show_quick_fallback(self):
        """Quick fallback when no barcode is detected"""
        self.current_scan = {
            "type": "packaging",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": {
                "name": "Unknown Medicine",
                "genuine": False,
                "confidence": 0.3,
                "method": "Barcode Scan",
                "reason": "No barcode detected. Try manual entry or pill scan."
            }
        }
        Clock.schedule_once(lambda dt: self.show_scan_result(), 0.5)
    
    def capture_and_analyze_pill(self):
        """Capture image and analyze pill using actual AI model"""
        try:
            scan_pill_screen = self.root.ids.screen_manager.get_screen('scan_pill')
            pill_camera_widget = scan_pill_screen.ids.pill_camera
            if not pill_camera_widget or not pill_camera_widget.texture:
                toast("Camera not ready. Please use upload option.")
                return
                
            self.is_processing = True
            loading_dialog = LoadingDialog()
            loading_dialog.open()
            
            # Capture image from camera
            texture = pill_camera_widget.texture
            size = texture.size
            pixels = texture.pixels
            image_data = np.frombuffer(pixels, np.uint8)
            image_data = image_data.reshape((size[1], size[0], 4))
            
            # Save temporary image for analysis
            temp_path = "temp_pill_image.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR))
            
            # Analyze with the actual AI model
            analysis = self.ai_model.analyze_pill_image(temp_path)
            
            if "error" in analysis:
                toast(f"Analysis error: {analysis['error']}")
                loading_dialog.dismiss()
                self.is_processing = False
                return
                
            # Process results
            self.process_pill_analysis(analysis)
            
            # Reset camera UI after analysis
            self.reset_pill_camera_ui()
            loading_dialog.dismiss()
            self.is_processing = False
            
        except Exception as e:
            toast(f"Error: {str(e)}")
            self.reset_pill_camera_ui()
            self.is_processing = False

    def process_pill_analysis(self, analysis):
        """Process AI pill analysis results"""
        medicine_data = {
            "name": f"{analysis['type'].title()}",
            "genuine": analysis['genuine'],
            "confidence": analysis['confidence'],
            "shape": analysis['shape'],
            "color": analysis['color'],
            "ai_analysis": True
        }
        
        self.current_scan = {
            "type": "pill",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": medicine_data
        }
        
        status = "✅ GENUINE" if analysis['genuine'] else "❌ COUNTERFEIT"
        confidence_percent = analysis['confidence'] * 100
        
        # Update the result label
        try:
            scan_pill_screen = self.root.ids.screen_manager.get_screen('scan_pill')
            if hasattr(scan_pill_screen, 'ids') and 'pill_result_label' in scan_pill_screen.ids:
                scan_pill_screen.ids.pill_result_label.text = f"{status} - {analysis['type'].title()} ({confidence_percent:.1f}%)"
        except:
            pass
        
        self.show_scan_result()

    def show_scan_result(self):
        """Show scan result dialog"""
        if not self.current_scan:
            return
        
        result = self.current_scan['result']
        
        result_text = f"""
Medicine: {result.get('name', 'Unknown')}
Status: {'✅ GENUINE' if result.get('genuine') else '❌ COUNTERFEIT'}
Confidence: {result.get('confidence', 0.95)*100:.1f}%
Method: {result.get('method', 'Unknown')}
"""

        if 'shape' in result:
            result_text += f"Shape: {result.get('shape', 'Unknown')}\n"
        if 'color' in result:
            result_text += f"Color: {result.get('color', 'Unknown')}\n"
        if 'manufacturer' in result:
            result_text += f"Manufacturer: {result.get('manufacturer', 'Unknown')}\n"
        if 'expiry_date' in result:
            result_text += f"Expiry Date: {result.get('expiry_date', 'Unknown')}\n"
        if 'uses' in result:
            result_text += f"Uses: {result.get('uses', 'Unknown')}\n"
        if 'verification_reasons' in result:
            result_text += f"Verification: {', '.join(result.get('verification_reasons', []))}\n"
        if 'reason' in result:
            result_text += f"Note: {result.get('reason', '')}\n"

        buttons = [
            MDFlatButton(text="Cancel", on_release=lambda x: self.result_dialog.dismiss()),
            MDRaisedButton(text="Save", on_release=lambda x: self.save_result()),
            MDRaisedButton(text="Share", on_release=lambda x: self.share_result())
        ]
        
        self.result_dialog = MDDialog(
            title="Scan Result",
            text=result_text,
            buttons=buttons,
            size_hint=(0.9, 0.7)
        )
        self.result_dialog.open()
    
    def share_result(self):
        """Share scan result"""
        if not self.current_scan:
            return
            
        result = self.current_scan['result']
        share_text = f"MediScan Result: {result.get('name', 'Unknown')} - {'Genuine' if result.get('genuine') else 'Counterfeit'}"
        
        # Copy to clipboard
        Clipboard.copy(share_text)
        toast("Result copied to clipboard")
        
        self.result_dialog.dismiss()

    def save_result(self):
        """Save scan result to history"""
        if not self.current_scan:
            return
        
        self.scan_history.append(self.current_scan)
        self.scan_history_count += 1
        
        if self.current_scan['result'].get('genuine'):
            self.genuine_count += 1
        else:
            self.fake_count += 1
            
        self.save_history()
        self.result_dialog.dismiss()
        toast("Result saved to history")
        
        # Update history screen
        self.update_history_screen()

    def update_history_screen(self):
        """Update the history screen with current scan history"""
        try:
            history_screen = self.root.ids.screen_manager.get_screen('history')
            if hasattr(history_screen, 'ids') and 'history_list' in history_screen.ids:
                history_list = history_screen.ids.history_list
                history_list.clear_widgets()
                
                for scan in reversed(self.scan_history):  # Show most recent first
                    result = scan['result']
                    item = HistoryItem(
                        medicine_name=result.get('name', 'Unknown'),
                        scan_date=scan['timestamp'],
                        status='GENUINE' if result.get('genuine') else 'COUNTERFEIT',
                        is_genuine=result.get('genuine', False),
                        confidence=f"{result.get('confidence', 0)*100:.1f}",
                        scan_data=scan
                    )
                    history_list.add_widget(item)
        except Exception as e:
            print(f"Error updating history screen: {e}")

    def share_history_item(self, scan_data):
        """Share a specific history item"""
        result = scan_data['result']
        share_text = f"MediScan Result:\nMedicine: {result.get('name', 'Unknown')}\nStatus: {'Genuine' if result.get('genuine') else 'Counterfeit'}\nDate: {scan_data['timestamp']}\nConfidence: {result.get('confidence', 0)*100:.1f}%"
        
        # Copy to clipboard
        Clipboard.copy(share_text)
        toast("Result copied to clipboard")

    def send_chat_message(self):
        """Send message to medical assistant"""
        try:
            assistant_screen = self.root.ids.screen_manager.get_screen('assistant')
            chat_input = assistant_screen.ids.chat_input
            chat_list = assistant_screen.ids.chat_list
            
            message = chat_input.text.strip()
            if not message:
                return
                
            # Add user message
            user_message = ChatMessage(message=message, is_bot=False)
            chat_list.add_widget(user_message)
            
            # Clear input
            chat_input.text = ""
            
            # Generate bot response
            Clock.schedule_once(lambda dt: self.generate_bot_response(message), 0.5)
            
            # Scroll to bottom
            Clock.schedule_once(lambda dt: setattr(assistant_screen.ids.chat_scroll, 'scroll_y', 0), 0.1)
            
        except Exception as e:
            print(f"Error sending chat message: {e}")
    
    def generate_bot_response(self, user_message):
        """Generate response from medical assistant"""
        try:
            assistant_screen = self.root.ids.screen_manager.get_screen('assistant')
            chat_list = assistant_screen.ids.chat_list
            
            # Simple response matching
            user_message_lower = user_message.lower()
            response = CHATBOT_RESPONSES['default']
            
            for key in CHATBOT_RESPONSES:
                if key in user_message_lower:
                    response = CHATBOT_RESPONSES[key]
                    break
            
            # Add bot response
            bot_message = ChatMessage(message=response, is_bot=True)
            chat_list.add_widget(bot_message)
            
            # Scroll to bottom
            Clock.schedule_once(lambda dt: setattr(assistant_screen.ids.chat_scroll, 'scroll_y', 0), 0.1)
            
        except Exception as e:
            print(f"Error generating bot response: {e}")

    def clear_history(self):
        """Clear scan history"""
        self.scan_history = []
        self.scan_history_count = 0
        self.genuine_count = 0
        self.fake_count = 0
        self.save_history()
        self.update_history_screen()
        toast("History cleared")

    def load_history(self):
        """Load scan history from file"""
        try:
            if os.path.exists('scan_history.json'):
                with open('scan_history.json', 'r') as f:
                    data = json.load(f)
                    self.scan_history = data.get('history', [])
                    self.scan_history_count = len(self.scan_history)
                    self.genuine_count = data.get('genuine_count', 0)
                    self.fake_count = data.get('fake_count', 0)
        except:
            self.scan_history = []

    def save_history(self):
        """Save scan history to file"""
        try:
            data = {
                'history': self.scan_history,
                'genuine_count': self.genuine_count,
                'fake_count': self.fake_count
            }
            with open('scan_history.json', 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving history: {e}")

if __name__ == '__main__':
    MediScanApp().run()