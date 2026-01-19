import torch
import cv2
import os
import sys
import numpy as np
import importlib

def letterbox_resize(img, target_size=640):
    """
    Resize RGB image using letterbox (maintains aspect ratio + padding).
    
    Args:
        img: numpy array [H, W, C] in RGB format
        target_size: target size (default 640)
    
    Returns:
        resized_img: letterboxed image [target_size, target_size, C]
        scale_ratio: scale factor used
        (pad_w, pad_h): padding applied (left/right, top/bottom)
    """
    h, w = img.shape[:2]
    
    # Calculate scale to fit within target_size
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize maintaining aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    # Create padded image (gray padding)
    padded = np.full((target_size, target_size, img.shape[2]), 114, dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return padded, scale, (pad_w, pad_h)

# Try importing VanillaCNN components
# Assuming the script is run from project root, we need to make sure we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Store paths but don't add to sys.path yet - we'll load the right one per model
SPIKEYOLO_PATH = os.path.join(project_root, 'SpikeYOLO')
SPIKEYOLO_ENCODED_PATH = os.path.join(project_root, 'SpikeYOLO_Encoded')

# We'll import YOLO dynamically in ModelWrapper based on model type
YOLO = None


try:
    from VanillaCNN.model import SimpleYoloCNN
    from VanillaCNN.yolo_decoder import decode_predictions
    from VanillaCNN.nms import nms
    VANILLA_AVAILABLE = True
except ImportError as e:
    VANILLA_AVAILABLE = False
    print(f"Warning: VanillaCNN not found or import failed: {e}")

# Encoding functions (latency and poisson). Poisson replaces previous deterministic rate encoder when used.
try:
    from data_encoding.latency_encoding import latency_encode
except ImportError as e:
    latency_encode = None
    print(f"Warning: latency encoder not found: {e}")

try:
    from data_encoding.poisson_encoding import poisson_encode
except ImportError as e:
    poisson_encode = None
    print(f"Warning: poisson encoder not found: {e}")

ENCODING_AVAILABLE = any([latency_encode is not None, poisson_encode is not None])

DEFAULT_WEIGHTS = {
    'SpikeYOLO': os.path.join(project_root, 'Results_train_final', 'SpikeYOLO', 'weights', 'best.pt'),
    'SpikeYOLO_latency': os.path.join(project_root, 'Results_train_final', 'SpikeYOLO_latency', 'weights', 'best.pt'),
    'SpikeYOLO_poisson': os.path.join(project_root, 'Results_train_final', 'SpikeYOLO_poisson', 'weights', 'best.pt'),
    'YOLO': os.path.join(project_root, 'Results_train_final', 'Yolov8', 'weights', 'best.pt'),
    'VanillaCNN': os.path.join(project_root, 'Results_train_final', 'VanillaCNN', 'best_simple_yolo_cnn.pt'),
} 

# Map user-facing model names to internal normalized identifiers
MODEL_ALIAS = {
    'SpikeYOLO': 'spikeyolo_standard',
    'SpikeYOLO_latency': 'spikeyolo_latency',
    'SpikeYOLO_poisson': 'spikeyolo_poisson',
    'YOLO': 'yolov8',
    'VanillaCNN': 'vanillacnn',
} 

class ModelWrapper:
    def __init__(self, model_type, weights_path, time_steps=4, device='cuda', encoding=None, imgsz=640):
        # Normalize user-facing model names to internal identifiers
        normalized = MODEL_ALIAS.get(model_type, model_type)
        if normalized != model_type:
            print(f"Info: mapping model type '{model_type}' to '{normalized}'")
        self.model_type = normalized
        self.weights_path = weights_path
        self.time_steps = time_steps
        self.imgsz = imgsz
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Determine encoding: User override > Model inferred
        self.encoding_type = encoding
        if not self.encoding_type:
            if 'latency' in self.model_type:
                self.encoding_type = 'latency'
            elif 'poisson' in self.model_type:
                self.encoding_type = 'poisson'
            else:
                self.encoding_type = None
        
        # Set is_spike flag (True if model uses spike encoding)
        self.is_spike = self.encoding_type is not None
        
        # Initialize temporal mode attribute (set by test scripts)
        self.temporal_mode = 'repeat'  # default
        
        # Load the correct YOLO class based on model type
        self._load_yolo_module()
        
        self.model = None
        self.load_model()
    
    def _load_yolo_module(self):
        """Dynamically load the correct ultralytics YOLO based on model type."""
        global YOLO
        
        if self.model_type in ['spikeyolo_latency', 'spikeyolo_poisson']:
            # Encoded models: use SpikeYOLO_Encoded/ultralytics
            if SPIKEYOLO_ENCODED_PATH not in sys.path:
                sys.path.insert(0, SPIKEYOLO_ENCODED_PATH)
            print(f"Loading YOLO from SpikeYOLO_Encoded for {self.model_type}")
        elif self.model_type == 'spikeyolo_standard':
            # Standard SpikeYOLO: use SpikeYOLO/ultralytics
            if SPIKEYOLO_PATH not in sys.path:
                sys.path.insert(0, SPIKEYOLO_PATH)
            print(f"Loading YOLO from SpikeYOLO for {self.model_type}")
        else:
            # YOLOv8: use standard ultralytics (should already be available)
            pass
        
        # Import or reload YOLO module
        if 'ultralytics' in sys.modules:
            # Reload to get the correct version
            import ultralytics
            importlib.reload(ultralytics)
            from ultralytics import YOLO as YOLO_class
            YOLO = YOLO_class
        else:
            from ultralytics import YOLO as YOLO_class
            YOLO = YOLO_class

    def load_model(self):
        print(f"Loading {self.model_type} from {self.weights_path}...")
        
        if self.model_type == 'vanillacnn':
            if not VANILLA_AVAILABLE:
                raise ImportError("VanillaCNN modules not available")
            self.model = SimpleYoloCNN(num_classes=2, S=20).to(self.device)
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        elif self.model_type == 'yolov8':
            self.model = YOLO(self.weights_path) # Ultralytics handles device internally usually, but we can force it
            # Ensure model is on the desired device
            try:
                self.model.to(self.device)
                print(f"Moved yolov8 model to {self.device}")
            except Exception:
                try:
                    self.model.model.to(self.device)
                    print(f"Moved yolov8 underlying module to {self.device}")
                except Exception as e:
                    print(f"Warning: Could not move yolov8 model to device: {e}")
            
        elif self.model_type in ['spikeyolo_standard', 'spikeyolo_latency', 'spikeyolo_poisson']:
            self.model = YOLO(self.weights_path)
            # Move spiking model to device as well
            try:
                self.model.to(self.device)
                print(f"Moved SpikeYOLO model to {self.device}")
            except Exception:
                try:
                    self.model.model.to(self.device)
                    print(f"Moved SpikeYOLO underlying module to {self.device}")
                except Exception as e:
                    print(f"Warning: Could not move SpikeYOLO model to device: {e}")
            
            # Patch for spike inputs if needed
            if self.model_type in ['spikeyolo_latency', 'spikeyolo_poisson']:
                 # Patch logic from predict_spikeyolo_latency.py
                 net = self.model.model
                 # Assuming first layer handles T
                 if hasattr(net, 'model') and len(net.model) > 0:
                     ms_get_t_layer = net.model[0]
                     original_forward = ms_get_t_layer.forward
                     
                     def patched_forward(x):
                         if x.dim() == 5: 
                             return x
                         return original_forward(x)
                     
                     ms_get_t_layer.forward = patched_forward
                     print("Patched SpikeYOLO first layer for 5D input.")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def encode_image(self, img_rgb):
        """
        Encodes an RGB image into spikes if applicable.
        Returns: tensor suitable for model input, or None if no encoding needed.

        This returns a multi-step encoding: shape [T, 1, C, H, W].
        """
        if self.encoding_type == 'latency':
            # Resize using letterbox (maintains aspect ratio)
            img_resized, _, _ = letterbox_resize(img_rgb, target_size=self.imgsz)
            spikes = latency_encode(img_resized, self.time_steps) # returns [T, C, H, W]
            # Make batch-first shape: [1, T, C, H, W]
            spikes = spikes.unsqueeze(0)
            return spikes.to(self.device)
            
        elif self.encoding_type == 'poisson':
            # Resize using letterbox (maintains aspect ratio)
            img_resized, _, _ = letterbox_resize(img_rgb, target_size=self.imgsz)
            if poisson_encode is None:
                raise ImportError("Poisson encoder not available (data_encoding.poisson_encoding)")
            spikes = poisson_encode(img_resized, self.time_steps)
            spikes = spikes.unsqueeze(0) # [1, T, C, H, W]
            return spikes.to(self.device)

        return None # No encoding

    def encode_frame_single(self, img_rgb):
        """
        Encodes a single RGB frame as a single time-step spike tensor.
        Returns: Tensor [C, H, W] (on CPU) or None if not applicable.
        This helper is intended for building sliding windows of T frames.
        """
        if self.encoding_type == 'latency':
            img_resized, _, _ = letterbox_resize(img_rgb, target_size=self.imgsz)
            # Use the full T to get a proper TTFS slice; take the earliest slice (t=0) to avoid all-ones
            spikes = latency_encode(img_resized, self.time_steps)  # [T, C, H, W]
            return spikes[0].to(torch.uint8).cpu()
        elif self.encoding_type == 'poisson':
            img_resized, _, _ = letterbox_resize(img_rgb, target_size=self.imgsz)
            if poisson_encode is None:
                raise ImportError("Poisson encoder not available (data_encoding.poisson_encoding)")
            spikes = poisson_encode(img_resized, self.time_steps)  # [T, C, H, W]
            return spikes[0].to(torch.uint8).cpu()

        return None  # No encoding for non-spiking models

    def predict_tensor(self, input_tensor, conf_thres=0.25, iou_thres=0.45, original_shape=None):
        """
        Runs inference on pre-processed tensor.
        original_shape: (h, w) of original image for scaling boxes.
        """
        with torch.no_grad():
            if self.model_type == 'yolov8':
                results = self.model.predict(input_tensor, verbose=False, conf=conf_thres, iou=iou_thres)
                det = []
                if len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        cls = int(box.cls)
                        det.append([*xyxy, conf, float(cls)])
                return det

            elif self.model_type == 'spikeyolo_standard':
                 # Standard expects list of images or similar usually, assuming input_tensor is standard yolo input
                 # But YOLOv8 predict() usually takes path or component. 
                 # If input_tensor is pixel tensor, predict() might handle it or we use model directly
                 # For consistency with SpikeYOLO, we might use model.model(input_tensor)
                 # BUT standar YOLOv8 usually handles sizing internally.
                 results = self.model.predict(input_tensor, verbose=False, conf=conf_thres, iou=iou_thres)
                 det = []
                 if len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        cls = int(box.cls)
                        det.append([*xyxy, conf, float(cls)])
                 return det
                 
            elif self.model_type == 'vanillacnn':
                 # input_tensor expected to be [1, 3, 640, 640]
                 pred = self.model(input_tensor)
                 batch_dets = decode_predictions(pred, conf_thres=conf_thres, S=20, img_size=640)
                 dets = batch_dets[0]
                 final_dets = nms(dets, iou_thres=iou_thres)
                 return final_dets

            elif self.model_type in ['spikeyolo_latency', 'spikeyolo_poisson']:
                 net = self.model.model
                 # Ensure the input tensor is float and on the model device
                 if input_tensor.device != self.device:
                     input_tensor = input_tensor.to(self.device)
                 if input_tensor.dtype != torch.float32:
                     input_tensor = input_tensor.float()
                 
                 # Debug: print input shape and model type
                 import os as _os_debug
                 debug_spike = _os_debug.environ.get('DEBUG_SPIKE', '').lower() in ('1', 'true', 'yes')
                 if debug_spike:
                     print(f"[SPIKE DEBUG] input_tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}, device: {input_tensor.device}")
                     print(f"[SPIKE DEBUG] model type: {self.model_type}, net: {type(net)}")
                 
                 preds = net(input_tensor)
                 
                 if debug_spike:
                     print(f"[SPIKE DEBUG] raw preds shape: {preds.shape if hasattr(preds, 'shape') else 'N/A'}, type: {type(preds)}")
                 
                 from ultralytics.utils.ops import non_max_suppression, scale_boxes
                 
                 preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)
                 
                 if debug_spike:
                     print(f"[SPIKE DEBUG] after NMS: {len(preds)} batches, first batch len: {len(preds[0]) if len(preds) > 0 else 0}")
                 
                 det_result = preds[0]
                 
                 det = []
                 if len(det_result):
                    if original_shape:
                        det_result[:, :4] = scale_boxes((640, 640), det_result[:, :4], original_shape).round()
                        
                    for *xyxy, conf, cls in det_result:
                        det.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf), float(cls)])
                 
                 if debug_spike:
                     print(f"[SPIKE DEBUG] final det count: {len(det)}")
                 
                 return det
                 
        return []

    def predict(self, image_source, conf_thres=0.25, iou_thres=0.45):
        """
        Runs inference on a single image.
        image_source: path to image (str) or numpy array (BGR).
        Returns list of detections: [[x1, y1, x2, y2, conf, cls], ...]
        """
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Could not read image: {image_source}")
        else:
            img = image_source
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 1. Encode if needed
        if self.encoding_type:
            input_tensor = self.encode_image(img_rgb)
            return self.predict_tensor(input_tensor, conf_thres, iou_thres, original_shape=(h, w))
            
        # 2. Others
        elif self.model_type == 'vanillacnn':
            img_resized, _, _ = letterbox_resize(img_rgb, target_size=self.imgsz)
            img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            return self.predict_tensor(img_tensor, conf_thres, iou_thres)
            
        elif self.model_type in ['yolov8', 'spikeyolo_standard']:
            # YOLOv8 handles raw images well
            return self.predict_tensor(img_rgb, conf_thres, iou_thres) # YOLOv8 predict accepts numpy
            
        return []

    def predict_batched(self, frames_list, conf_thres=0.25, iou_thres=0.45):
        """
        Run inference on a batch of frames for batched temporal mode.
        frames_list: list of numpy arrays (BGR frames)
        Returns list of detections for the last frame in the batch.
        
        This method is used in 'batched' temporal mode where T consecutive frames
        are encoded and processed together without overlap.
        """
        if not self.encoding_type:
            # For non-encoded models, just predict the last frame
            return self.predict(frames_list[-1], conf_thres, iou_thres)
        
        # Encode each frame individually and stack them
        encoded_frames = []
        for frame in frames_list:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized, _, _ = letterbox_resize(img_rgb, target_size=self.imgsz)
            
            if self.encoding_type == 'latency':
                # Encode as single timestep (take first slice of TTFS encoding)
                spike = latency_encode(img_resized, self.time_steps)[0]  # [C, H, W]
                encoded_frames.append(spike.to(torch.uint8).cpu())
            elif self.encoding_type == 'poisson':
                if poisson_encode is None:
                    raise ImportError("Poisson encoder not available")
                spike = poisson_encode(img_resized, self.time_steps)[0]  # [C, H, W]
                encoded_frames.append(spike.to(torch.uint8).cpu())
        
        # Stack to create [T, C, H, W] tensor
        input_tensor = torch.stack(encoded_frames, dim=0)  # [T, C, H, W]
        input_tensor = input_tensor.unsqueeze(0)  # [1, T, C, H, W]
        
        # Get original shape from last frame for bbox scaling
        h, w = frames_list[-1].shape[:2]
        
        return self.predict_tensor(input_tensor, conf_thres, iou_thres, original_shape=(h, w))

    def predict_batched_inference_only(self, frames_list, conf_thres=0.25, iou_thres=0.45):
        """
        Optimized version that pre-encodes frames to exclude encoding overhead from energy measurements.
        Only the model inference is timed/measured, not the preprocessing.
        
        frames_list: list of numpy arrays (BGR frames)
        Returns list of detections for the last frame in the batch.
        """
        if not self.encoding_type:
            # For non-encoded models, just predict the last frame
            return self.predict(frames_list[-1], conf_thres, iou_thres)
        
        # PRE-ENCODE (not measured by tracker)
        encoded_frames = []
        for frame in frames_list:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized, _, _ = letterbox_resize(img_rgb, target_size=self.imgsz)
            
            if self.encoding_type == 'latency':
                spike = latency_encode(img_resized, self.time_steps)[0]  # [C, H, W]
                encoded_frames.append(spike.to(torch.uint8).cpu())
            elif self.encoding_type == 'poisson':
                if poisson_encode is None:
                    raise ImportError("Poisson encoder not available")
                spike = poisson_encode(img_resized, self.time_steps)[0]  # [C, H, W]
                encoded_frames.append(spike.to(torch.uint8).cpu())
        
        # Stack tensor
        input_tensor = torch.stack(encoded_frames, dim=0).unsqueeze(0)  # [1, T, C, H, W]
        h, w = frames_list[-1].shape[:2]
        
        # ONLY THIS IS MEASURED: actual model inference
        return self.predict_tensor(input_tensor, conf_thres, iou_thres, original_shape=(h, w))

    def preprocess_to_tensor(self, frame):
        """
        Preprocess a BGR frame to tensor format for RGB models (YOLO/VanillaCNN).
        Returns preprocessed tensor and metadata needed for inference.
        This does NOT run inference - only preprocessing.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        if self.model_type == 'vanillacnn':
            img_resized, _, _ = letterbox_resize(img_rgb, target_size=self.imgsz)
            img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            return {'tensor': img_tensor, 'orig_shape': (h, w), 'type': 'vanillacnn'}
        
        elif self.model_type in ['yolov8', 'spikeyolo_standard']:
            # For YOLO models, just return the RGB array - YOLO handles preprocessing
            return {'rgb': img_rgb, 'orig_shape': (h, w), 'type': 'yolo'}
        
        return None

    def predict_tensor_only(self, tensor_data, conf_thres=0.25, iou_thres=0.45):
        """
        Run inference on preprocessed tensor (for RGB models).
        This is ONLY the model forward pass - no preprocessing.
        """
        if tensor_data['type'] == 'vanillacnn':
            return self.predict_tensor(tensor_data['tensor'], conf_thres, iou_thres)
        
        elif tensor_data['type'] == 'yolo':
            # For YOLO, use predict_tensor with the RGB array
            return self.predict_tensor(tensor_data['rgb'], conf_thres, iou_thres)
        
        return []

    def get_class_names(self):
        if self.model_type == 'vanillacnn':
            return {0: 'smoke', 1: 'fire'} # Hardcoded for D-Fire usually
        elif hasattr(self.model, 'names'):
            return self.model.names
        return {0: 'smoke', 1: 'fire'}

