import customtkinter as ctk
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image, ImageTk
import time
import keyboard
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests
import base64

# Create directories for saving data
os.makedirs('training_data', exist_ok=True)
os.makedirs('models', exist_ok=True)

GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-2.0-flash"

INSTRUCTIONS_TEXT = (
    "Welcome to GameAble!\n\n"
    "1. Enter the game name and get suggested controls/poses using the chatbot.\n"
    "2. Set the number of poses and assign keys for each pose.\n"
    "3. Capture images for each pose using the camera feed.\n"
    "4. Train the model with your captured data.\n"
    "5. Use 'Smart AI Model' for an AI-generated model and preprocessing.\n"
    "6. Start prediction to control your game with poses!\n\n"
    "Tips:\n- Use clear, distinct poses.\n- Ensure good lighting and a clean background.\n- Use the recommended hyperparameters for best results.\n"
)

class PoseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Get all pose directories
        pose_dirs = [d for d in os.listdir(data_dir) if d.startswith('pose_')]
        pose_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by pose number
        
        for pose_idx, pose_dir in enumerate(pose_dirs):
            pose_path = os.path.join(data_dir, pose_dir)
            if os.path.isdir(pose_path):
                image_files = [f for f in os.listdir(pose_path) if f.endswith('.jpg')]
                if not image_files:
                    print(f"Warning: No images found in {pose_path}")
                    continue
                    
                for img_name in image_files:
                    self.images.append(os.path.join(pose_path, img_name))
                    self.labels.append(pose_idx)
        
        if not self.images:
            raise ValueError("No images found in training data")
            
        # Convert labels to tensor for max operation
        self.labels = torch.tensor(self.labels)
        print(f"Loaded {len(self.images)} images")
        print(f"Pose distribution: {torch.bincount(self.labels).tolist()}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SetupWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("GameAble - Pose Configuration")
        self.geometry("500x600")
        self.num_poses = 0
        self.key_mappings = {}
        self.create_widgets()

    def create_widgets(self):
        # Title and logo
        title_label = ctk.CTkLabel(self, text="GameAble", font=("Arial", 28, "bold"))
        title_label.pack(pady=(10, 0))
        try:
            from PIL import Image
            logo_img = ctk.CTkImage(light_image=Image.open("logo.png"), dark_image=Image.open("logo.png"), size=(80, 80))
            logo_label = ctk.CTkLabel(self, image=logo_img, text="")
            logo_label.pack(pady=(0, 10))
        except Exception as e:
            print(f"[UI] Could not load logo: {e}")
        # Instructions button
        instr_btn = ctk.CTkButton(self, text="Instructions", command=self.show_instructions)
        instr_btn.pack(pady=(0, 10))
        # Game suggestion chatbot UI
        self.game_frame = ctk.CTkFrame(self)
        self.game_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(self.game_frame, text="Game Name:").pack(side="left", padx=5)
        self.game_entry = ctk.CTkEntry(self.game_frame, width=120)
        self.game_entry.pack(side="left", padx=5)
        self.send_btn = ctk.CTkButton(self.game_frame, text="Send", command=self.ask_game_suggestions)
        self.send_btn.pack(side="left", padx=5)
        self.suggestion_box = ctk.CTkTextbox(self, height=100)
        self.suggestion_box.pack(pady=5, padx=10, fill="x")
        # Number of poses input
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.input_frame, text="Number of Poses:").pack(side="left", padx=5)
        self.num_poses_entry = ctk.CTkEntry(self.input_frame, width=50)
        self.num_poses_entry.pack(side="left", padx=5)
        self.setup_btn = ctk.CTkButton(self.input_frame, text="Setup", command=self.setup_poses)
        self.setup_btn.pack(side="left", padx=5)
        # Pose list
        self.list_frame = ctk.CTkFrame(self)
        self.list_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.pose_list = ctk.CTkTextbox(self.list_frame, height=200)
        self.pose_list.pack(pady=5, padx=5, fill="both", expand=True)
        # Done button
        self.done_btn = ctk.CTkButton(self, text="Done", command=self.finish_setup)
        self.done_btn.pack(pady=10)

    def ask_game_suggestions(self):
        game_name = self.game_entry.get().strip()
        if not game_name:
            self.suggestion_box.delete("1.0", "end")
            self.suggestion_box.insert("end", "Please enter a game name.")
            return
        self.suggestion_box.delete("1.0", "end")
        self.suggestion_box.insert("end", "Thinking...\n")
        self.update()
        # Compose Gemini prompt
        prompt = (
            f"this is the pc game {game_name} and my project aims to play it without keyboard or mouse with pose detection with cnn. "
            "Give me ONLY the following output, with NO extra notes, rationale, tips, or explanations: \n"
            "game name:\nsuggested controls\nkeybaord key: pose\nkeybaord key 2:pose 2....\n"
            "The controls should NOT involve usage of fingers."
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        data = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            output = None
            for candidate in result.get('candidates', []):
                parts = candidate.get('content', {}).get('parts', [])
                for part in parts:
                    if 'text' in part:
                        output = part['text']
                        break
                if output:
                    break
            self.suggestion_box.delete("1.0", "end")
            if output:
                output = output.replace('*', '')
                self.suggestion_box.insert("end", output)
            else:
                self.suggestion_box.insert("end", "No suggestion received.")
        except Exception as e:
            self.suggestion_box.delete("1.0", "end")
            self.suggestion_box.insert("end", f"Error: {e}")

    def setup_poses(self):
        try:
            num_poses = int(self.num_poses_entry.get())
            if num_poses <= 0:
                self.show_error("Please enter a positive number")
                return
                
            self.num_poses = num_poses
            self.key_mappings = {}
            
            # Add normal pose (always index 0)
            self.key_mappings[0] = None
            
            # Get keys for other poses
            for i in range(1, num_poses + 1):
                key = self.ask_key(f"Enter key for pose {i} (or 'none' for no action):")
                if key.lower() == 'none':
                    key = None
                self.key_mappings[i] = key
            
            # Update pose list display
            self.pose_list.delete("1.0", "end")
            self.pose_list.insert("end", f"Total Poses: {num_poses + 1} (including normal)\n\n")
            self.pose_list.insert("end", "Key mappings:\n")
            self.pose_list.insert("end", "  Normal Pose: None\n")
            for pose, key in self.key_mappings.items():
                if pose > 0:
                    self.pose_list.insert("end", f"  Pose {pose}: {key if key else 'None'}\n")
            
        except ValueError:
            self.show_error("Please enter valid numbers")
            
    def ask_key(self, message):
        dialog = ctk.CTkInputDialog(text=message, title="Key Mapping")
        return dialog.get_input()
        
    def show_error(self, message):
        dialog = ctk.CTkInputDialog(text=message, title="Error")
        dialog.get_input()
        
    def finish_setup(self):
        if not self.key_mappings:
            self.show_error("Please setup poses first")
            return
        self.destroy()

    def show_instructions(self):
        popup = ctk.CTkToplevel(self)
        popup.title("Instructions")
        popup.geometry("400x400")
        text_box = ctk.CTkTextbox(popup, height=30, width=380)
        text_box.pack(pady=10, padx=10, fill="both", expand=True)
        text_box.insert("end", INSTRUCTIONS_TEXT)
        close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=10)

class KeyControllerApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("GameAble - Key Controller")
        self.root.geometry("800x700")
        
        # Initialize variables
        self.cap = None
        self.is_capturing = False
        self.model = None
        self.key_mappings = {}
        self.current_pose = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Start with setup
        self.setup_poses()
        
    def setup_poses(self):
        setup_window = SetupWindow(self.root)
        self.root.wait_window(setup_window)
        self.key_mappings = setup_window.key_mappings
        self.create_widgets()
        
    def create_widgets(self):
        # Title and logo
        title_label = ctk.CTkLabel(self.root, text="GameAble", font=("Arial", 28, "bold"))
        title_label.pack(pady=(10, 0))
        try:
            from PIL import Image
            logo_img = ctk.CTkImage(light_image=Image.open("logo.png"), dark_image=Image.open("logo.png"), size=(80, 80))
            logo_label = ctk.CTkLabel(self.root, image=logo_img, text="")
            logo_label.pack(pady=(0, 10))
        except Exception as e:
            print(f"[UI] Could not load logo: {e}")
        # Instructions button
        instr_btn = ctk.CTkButton(self.root, text="Instructions", command=self.show_instructions)
        instr_btn.pack(pady=(0, 10))
        # Camera feed
        self.camera_frame = ctk.CTkFrame(self.root)
        self.camera_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.camera_width = 320
        self.camera_height = 240
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Camera Feed", width=self.camera_width, height=self.camera_height)
        self.camera_label.pack(pady=5)

        # --- Model Parameter Controls ---
        self.param_frame = ctk.CTkFrame(self.root)
        self.param_frame.pack(pady=5, padx=10, fill="x")

        # Epochs
        ctk.CTkLabel(self.param_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=2)
        self.epochs_slider = ctk.CTkSlider(self.param_frame, from_=1, to=50, number_of_steps=49)
        self.epochs_slider.set(12)
        self.epochs_slider.grid(row=0, column=1, padx=5, pady=2)
        self.epochs_value = ctk.CTkLabel(self.param_frame, text="12")
        self.epochs_value.grid(row=0, column=2, padx=5, pady=2)
        self.epochs_slider.configure(command=lambda v: self.epochs_value.configure(text=str(int(float(v)))))

        # Dataset size (images per pose)
        ctk.CTkLabel(self.param_frame, text="Images/Pose:").grid(row=1, column=0, padx=5, pady=2)
        self.dataset_slider = ctk.CTkSlider(self.param_frame, from_=10, to=200, number_of_steps=19)
        self.dataset_slider.set(50)
        self.dataset_slider.grid(row=1, column=1, padx=5, pady=2)
        self.dataset_value = ctk.CTkLabel(self.param_frame, text="50")
        self.dataset_value.grid(row=1, column=2, padx=5, pady=2)
        self.dataset_slider.configure(command=lambda v: self.dataset_value.configure(text=str(int(float(v)))))

        # Batch size
        ctk.CTkLabel(self.param_frame, text="Batch Size:").grid(row=2, column=0, padx=5, pady=2)
        self.batch_slider = ctk.CTkSlider(self.param_frame, from_=1, to=64, number_of_steps=63)
        self.batch_slider.set(32)
        self.batch_slider.grid(row=2, column=1, padx=5, pady=2)
        self.batch_value = ctk.CTkLabel(self.param_frame, text="32")
        self.batch_value.grid(row=2, column=2, padx=5, pady=2)
        self.batch_slider.configure(command=lambda v: self.batch_value.configure(text=str(int(float(v)))))

        # Learning rate
        ctk.CTkLabel(self.param_frame, text="Learning Rate:").grid(row=3, column=0, padx=5, pady=2)
        self.lr_entry = ctk.CTkEntry(self.param_frame, width=60)
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=3, column=1, padx=5, pady=2)

        # Progress frame
        self.progress_frame = ctk.CTkFrame(self.root)
        self.progress_frame.pack(pady=5, padx=10, fill="x")
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(pady=5, padx=5, fill="x")
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Ready")
        self.progress_label.pack(pady=5)

        # Pose selection
        self.pose_frame = ctk.CTkFrame(self.root)
        self.pose_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.pose_frame, text="Select Pose:").pack(side="left", padx=5)
        self.pose_var = ctk.StringVar()
        self.pose_menu = ctk.CTkOptionMenu(
            self.pose_frame,
            values=["Normal Pose"] + [f"Pose {i}" for i in range(1, len(self.key_mappings))],
            variable=self.pose_var,
            command=self.pose_selected
        )
        self.pose_menu.pack(side="left", padx=5)

        # Buttons
        self.button_frame = ctk.CTkFrame(self.root)
        self.button_frame.pack(pady=10, padx=10, fill="x")
        self.capture_btn = ctk.CTkButton(
            self.button_frame,
            text="Start Capture",
            command=self.start_capture
        )
        self.capture_btn.pack(side="left", padx=5)
        self.stop_btn = ctk.CTkButton(
            self.button_frame,
            text="Stop Capture",
            command=self.stop_capture,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)
        self.train_btn = ctk.CTkButton(
            self.button_frame,
            text="Train Model",
            command=self.train_model
        )
        self.train_btn.pack(side="left", padx=5)
        self.start_btn = ctk.CTkButton(
            self.button_frame,
            text="Start Prediction",
            command=self.start_prediction
        )
        self.start_btn.pack(side="left", padx=5)
        self.smart_ai_btn = ctk.CTkButton(
            self.button_frame,
            text="Smart AI Model",
            command=self.generate_smart_ai_model
        )
        self.smart_ai_btn.pack(side="left", padx=5)
        # Status label
        self.status_label = ctk.CTkLabel(self.root, text="Status: Ready")
        self.status_label.pack(pady=5)
        
    def pose_selected(self, pose_name):
        if pose_name == "Normal Pose":
            self.current_pose = 0
        else:
            self.current_pose = int(pose_name.split()[1])
            
    def start_capture(self):
        if self.current_pose is None:
            self.status_label.configure(text="Please select a pose")
            return
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        self.is_capturing = True
        self.count = 0
        self.max_images = int(float(self.dataset_slider.get()))
        # Create directory for this pose
        self.pose_dir = os.path.join('training_data', f'pose_{self.current_pose}')
        os.makedirs(self.pose_dir, exist_ok=True)
        # Update UI
        self.capture_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0)
        pose_name = "Normal" if self.current_pose == 0 else f"Pose {self.current_pose}"
        self.progress_label.configure(text=f"Capturing {pose_name}: 0/{self.max_images}")
        # Start capture loop
        self.capture_loop()
        
    def stop_capture(self):
        self.is_capturing = False
        self.capture_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_label.configure(text=f"Captured {self.count} images")
        self.status_label.configure(text=f"Captured {self.count} images for pose {self.current_pose}")
        
    def capture_loop(self):
        if not self.is_capturing:
            return
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save image
            img_path = os.path.join(self.pose_dir, f'{self.count}.jpg')
            cv2.imwrite(img_path, frame)
            # Resize for display
            img = Image.fromarray(frame_rgb).resize((self.camera_width, self.camera_height))
            img = ctk.CTkImage(light_image=img, dark_image=img, size=(self.camera_width, self.camera_height))
            self.camera_label.configure(image=img)
            # Update progress
            self.count += 1
            progress = self.count / self.max_images
            self.progress_bar.set(progress)
            pose_name = "Normal" if self.current_pose == 0 else f"Pose {self.current_pose}"
            self.progress_label.configure(
                text=f"Capturing {pose_name}: {self.count}/{self.max_images}"
            )
            if self.count >= self.max_images:
                self.stop_capture()
            else:
                self.root.after(100, self.capture_loop)
                
    def show_smart_ai_popup(self, code, preprocessing_code, hyperparams, reason):
        popup = ctk.CTkToplevel(self.root)
        popup.title("Smart AI Model Details")
        popup.geometry("600x600")
        text_box = ctk.CTkTextbox(popup, height=30, width=580)
        text_box.pack(pady=10, padx=10, fill="both", expand=True)
        text_box.insert("end", "SmartCNN Code:\n")
        text_box.insert("end", code + "\n\n")
        if preprocessing_code:
            text_box.insert("end", "Preprocessing Code:\n")
            text_box.insert("end", preprocessing_code + "\n\n")
        if hyperparams:
            text_box.insert("end", f"Hyperparameters: {hyperparams}\n\n")
        if reason:
            text_box.insert("end", f"Reason: {reason}\n\n")
        close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=10)

    def generate_smart_ai_model(self):
        print("[SmartAI] Starting Gemini AI model generation...")
        self.status_label.configure(text="Generating AI model with Gemini...")
        self.root.update()
        # Collect one sample image from each pose class
        print("[SmartAI] Collecting sample images from each pose class...")
        sample_images = []
        for i in range(len(self.key_mappings)):
            pose_dir = os.path.join('training_data', f'pose_{i}')
            if not os.path.isdir(pose_dir):
                print(f"[SmartAI] Skipping missing directory: {pose_dir}")
                continue
            files = [f for f in os.listdir(pose_dir) if f.endswith('.jpg')]
            if not files:
                print(f"[SmartAI] No images found in: {pose_dir}")
                continue
            img_path = os.path.join(pose_dir, files[0])
            print(f"[SmartAI] Using sample image: {img_path}")
            with open(img_path, 'rb') as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            sample_images.append((i, img_b64))
        # Compose strict prompt
        print("[SmartAI] Composing prompt for Gemini...")
        prompt = (
            "Generate a PyTorch CNN model class for image classification. "
            f"The model must take input images of shape (3, 224, 224) and output logits for {len(self.key_mappings)} classes. "
            "Only output the class definition (no training loop, no extra text). "
            "The class must be named 'SmartCNN' and inherit from nn.Module. "
            "The __init__ method of SmartCNN must take a num_classes argument and use it for the output layer (e.g., self.fc2 = nn.Linear(..., num_classes)). "
            "Also, provide the recommended preprocessing pipeline as a PyTorch torchvision.transforms.Compose code snippet, using the variable name 'self.transform', compatible with the model and dataset. "
            "Do not include any code except the class definition and the preprocessing code.\n"
            "Additionally, recommend the best hyperparameters (epochs, batch size, learning rate) for this model and dataset as a JSON object in the format: {\"epochs\": int, \"batch_size\": int, \"learning_rate\": float}. "
            "Finally, provide a short reason for why this model architecture and preprocessing are suitable for the provided images. "
            "Format your response as:\nCODE:\n<model code>\nPREPROCESSING:\n<preprocessing code>\nHYPERPARAMETERS:\n<json>\nREASON:\n<reason>"
        )
        # Prepare Gemini API request
        print("[SmartAI] Preparing Gemini API request...")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        data = {
            "contents": [
                {"role": "user", "parts": [
                    {"text": prompt},
                    *[{"inlineData": {"mimeType": "image/jpeg", "data": img_b64}} for _, img_b64 in sample_images]
                ]}
            ]
        }
        try:
            print("[SmartAI] Sending request to Gemini API...")
            response = requests.post(url, json=data)
            print(f"[SmartAI] Gemini API response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            print("[SmartAI] Parsing Gemini response...")
            # Extract code, preprocessing, hyperparameters, and reason from Gemini response
            code = None
            preprocessing_code = None
            hyperparams = None
            reason = None
            for candidate in result.get('candidates', []):
                parts = candidate.get('content', {}).get('parts', [])
                for part in parts:
                    if 'text' in part:
                        text = part['text']
                        print("[SmartAI] Raw Gemini output:")
                        print(text)
                        import re, json
                        # Extract code
                        code_match = re.search(r'CODE:\s*([\s\S]+?)PREPROCESSING:', text)
                        if code_match:
                            code_block = code_match.group(1)
                            code_block = re.sub(r'```+\s*python', '', code_block, flags=re.IGNORECASE)
                            code_block = re.sub(r'```+', '', code_block)
                            code = code_block.strip()
                        # Extract preprocessing
                        preproc_match = re.search(r'PREPROCESSING:\s*([\s\S]+?)HYPERPARAMETERS:', text)
                        if preproc_match:
                            preproc_block = preproc_match.group(1)
                            preproc_block = re.sub(r'```+\s*python', '', preproc_block, flags=re.IGNORECASE)
                            preproc_block = re.sub(r'```+', '', preproc_block)
                            preprocessing_code = preproc_block.strip()
                        # Extract hyperparameters
                        hyper_match = re.search(r'HYPERPARAMETERS:\s*([\{\[].*?[\}\]])', text, re.DOTALL)
                        if hyper_match:
                            try:
                                hyperparams = json.loads(hyper_match.group(1))
                            except Exception as e:
                                print(f"[SmartAI] Failed to parse hyperparameters: {e}")
                        # Extract reason
                        reason_match = re.search(r'REASON:\s*([\s\S]+)', text)
                        if reason_match:
                            reason = reason_match.group(1).strip()
                        break
                if code:
                    break
            if not code:
                print("[SmartAI] Gemini did not return a valid model class.")
                self.status_label.configure(text="Gemini did not return a valid model class.")
                return
            print("[SmartAI] Extracted SmartCNN code:")
            print(code)
            if preprocessing_code:
                print("[SmartAI] Extracted preprocessing code:")
                print(preprocessing_code)
                # Dynamically set self.transform
                try:
                    exec(preprocessing_code, globals(), locals())
                    self.transform = locals()['self'].transform if 'self' in locals() and hasattr(locals()['self'], 'transform') else locals()['transform']
                except Exception as e:
                    print(f"[SmartAI] Failed to set preprocessing: {e}")
            # Show everything in a popup window
            self.show_smart_ai_popup(code, preprocessing_code, hyperparams, reason)
            # Save code to file
            print("[SmartAI] Saving SmartCNN code to smart_cnn.py...")
            with open('smart_cnn.py', 'w', encoding='utf-8') as f:
                f.write(code)
            # Dynamically import SmartCNN
            print("[SmartAI] Importing SmartCNN class from smart_cnn.py...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("smart_cnn", "smart_cnn.py")
            smart_cnn = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(smart_cnn)
            self.SmartCNN = smart_cnn.SmartCNN
            print("[SmartAI] SmartCNN is ready for use!")
            # Apply recommended hyperparameters if available
            if hyperparams:
                print(f"[SmartAI] Applying recommended hyperparameters: {hyperparams}")
                self.epochs_slider.set(hyperparams.get('epochs', 12))
                self.batch_slider.set(hyperparams.get('batch_size', 32))
                self.lr_entry.delete(0, 'end')
                self.lr_entry.insert(0, str(hyperparams.get('learning_rate', 0.001)))
            # Display reason in the UI
            if reason:
                print(f"[SmartAI] Reason for model: {reason}")
                self.status_label.configure(text=f"AI model ready! Reason: {reason}")
            else:
                self.status_label.configure(text="AI model generated! Using SmartCNN for training.")
            # Set flag to use SmartCNN for prediction after training
            self.use_smartcnn_for_prediction = True
            print("[SmartAI] Starting training with SmartCNN and recommended hyperparameters...")
            self.train_model()
        except Exception as e:
            print(f"[SmartAI] Gemini error: {e}")
            self.status_label.configure(text=f"Gemini error: {e}")

    def train_model(self):
        # Clean up pose directories to match current number of poses
        num_classes = len(self.key_mappings)
        valid_pose_dirs = {f'pose_{i}' for i in range(num_classes)}
        data_dir = 'training_data'
        for d in os.listdir(data_dir):
            if d.startswith('pose_') and d not in valid_pose_dirs:
                pose_path = os.path.join(data_dir, d)
                import shutil
                shutil.rmtree(pose_path)
        # Update UI for training
        self.capture_btn.configure(state="disabled")
        self.train_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Training model...")
        self.status_label.configure(text="Preparing dataset...")
        self.root.update()
        try:
            # Get user-set parameters
            num_epochs = int(float(self.epochs_slider.get()))
            batch_size = int(float(self.batch_slider.get()))
            learning_rate = float(self.lr_entry.get())
            # Create dataset and dataloader
            dataset = PoseDataset('training_data', transform=self.transform)
            # Verify number of classes matches our key mappings
            if dataset.labels.max() >= num_classes:
                raise ValueError(f"Dataset contains labels up to {dataset.labels.max()}, but model expects {num_classes} classes")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # Use SmartCNN if available, else default CNN
            model_class = getattr(self, 'SmartCNN', None) or CNN
            self.model = model_class(num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            total_batches = len(dataloader)
            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                self.progress_label.configure(
                    text=f"Training - Epoch {epoch+1}/{num_epochs}"
                )
                self.root.update()
                for batch_idx, (inputs, labels) in enumerate(dataloader):
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # Update progress
                    batch_progress = (batch_idx + 1) / total_batches
                    epoch_progress = (epoch + batch_progress) / num_epochs
                    self.progress_bar.set(epoch_progress)
                    # Update status with current loss and accuracy
                    accuracy = 100 * correct / total
                    self.status_label.configure(
                        text=f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{total_batches}, "
                        f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%"
                    )
                    self.root.update()
                # Update status after each epoch
                avg_loss = running_loss / total_batches
                epoch_accuracy = 100 * correct / total
                self.status_label.configure(
                    text=f"Epoch {epoch+1}/{num_epochs} completed. "
                    f"Average Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
                )
                self.root.update()
            # Save model
            torch.save(self.model.state_dict(), 'models/pose_model.pth')
            # Update UI for completion
            self.progress_bar.set(1.0)
            self.progress_label.configure(text="Training completed!")
            self.status_label.configure(
                text=f"Model trained and saved successfully! Final Accuracy: {epoch_accuracy:.2f}%"
            )
            # If SmartCNN was used and flagged, start prediction automatically
            if getattr(self, 'use_smartcnn_for_prediction', False):
                print("[SmartAI] Starting prediction with SmartCNN after training...")
                self.use_smartcnn_for_prediction = False
                self.start_prediction()
        except Exception as e:
            print(f"[SmartAI] Gemini error: {e}")
            self.status_label.configure(text=f"Gemini error: {e}")
        finally:
            # Re-enable buttons
            self.capture_btn.configure(state="normal")
            self.train_btn.configure(state="normal")
            self.start_btn.configure(state="normal")
            self.root.update()
            
    def start_prediction(self):
        if not self.model:
            self.status_label.configure(text="Please train the model first!")
            return
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        self.is_capturing = True
        last_pose = None
        held_key = None
        while self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                # Preprocess frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                img = Image.fromarray(frame_rgb).resize((self.camera_width, self.camera_height))
                img_tensor = self.transform(img).unsqueeze(0)
                # Make prediction
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    _, predicted = torch.max(outputs, 1)
                    pose_idx = predicted.item()
                    # Hold key if not normal pose
                    if pose_idx > 0 and self.key_mappings[pose_idx] is not None:
                        key = self.key_mappings[pose_idx]
                        # Convert number keys to arrow keys
                        if key == '1':
                            key_to_hold = 'left'
                        elif key == '2':
                            key_to_hold = 'right'
                        elif key == '3':
                            key_to_hold = 'up'
                        elif key == '4':
                            key_to_hold = 'down'
                        else:
                            key_to_hold = key
                        if held_key != key_to_hold:
                            if held_key is not None:
                                keyboard.release(held_key)
                            keyboard.press(key_to_hold)
                            held_key = key_to_hold
                    else:
                        if held_key is not None:
                            keyboard.release(held_key)
                            held_key = None
                    last_pose = pose_idx
                    # Add predictions to overlay
                    frame_with_overlay = frame.copy()
                    y_offset = 30
                    # Show normal pose first
                    normal_prob = probabilities[0].item() * 100
                    cv2.putText(
                        frame_with_overlay,
                        f"Normal: {normal_prob:.1f}%",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    y_offset += 40
                    # Then show other poses
                    for i in range(1, len(self.key_mappings)):
                        prob = probabilities[i].item() * 100
                        key = self.key_mappings[i] if self.key_mappings[i] else "None"
                        if key == '1':
                            display_key = "<- (1)"
                        elif key == '2':
                            display_key = "-> (2)"
                        elif key == '3':
                            display_key = "Up (3)"
                        elif key == '4':
                            display_key = "Down (4)"
                        else:
                            display_key = key
                        cv2.putText(
                            frame_with_overlay,
                            f"Pose {i} ({display_key}): {prob:.1f}%",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        y_offset += 40
                # Update camera feed
                frame_with_overlay_rgb = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_with_overlay_rgb).resize((self.camera_width, self.camera_height))
                img = ctk.CTkImage(light_image=img, dark_image=img, size=(self.camera_width, self.camera_height))
                self.camera_label.configure(image=img)
                self.root.update()
                
    def show_instructions(self):
        popup = ctk.CTkToplevel(self.root)
        popup.title("Instructions")
        popup.geometry("400x400")
        text_box = ctk.CTkTextbox(popup, height=30, width=380)
        text_box.pack(pady=10, padx=10, fill="both", expand=True)
        text_box.insert("end", INSTRUCTIONS_TEXT)
        close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=10)

    def run(self):
        self.root.mainloop()
        
    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    app = KeyControllerApp()
    app.run() 