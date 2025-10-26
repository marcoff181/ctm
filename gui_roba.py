import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import ctypes  # For DPI awareness

# --- Import Attack Functions ---
try:
    from attack_functions import (
        awgn,
        blur,
        sharpening,
        median,
        resizing,
        jpeg_compression,
    )

    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False

# --- Attack Configuration ---
param_converters = {
    "JPEG": lambda x: int(round((1 - x) * 95) + 5),  # Quality from 5 to 100
    "Blur": lambda x: x * 5.0 + 0.1,  # Sigma from 0.1 to 5.1
    "AWGN": lambda x: x * 50.0,  # Dev. std from 0 to 50
    "Resize": lambda x: 1.0 - (x * 0.9 + 0.05),  # Scale from 0.95 to 0.05
    "Median": lambda x: [1, 3, 5, 7][int(round(x * 3))],  # Kernel size 1, 3, 5, 7
    "Sharp": lambda x: x * 5.0,  # Alpha from 0 to 5.0
}

attack_config = {
    "JPEG": lambda img, x: jpeg_compression(img, quality=param_converters["JPEG"](x)),
    "Blur": lambda img, x: blur(img, sigma=param_converters["Blur"](x)),
    "AWGN": lambda img, x: awgn(img, std=param_converters["AWGN"](x)),
    "Resize": lambda img, x: resizing(img, scale=param_converters["Resize"](x)),
    "Median": lambda img, x: median(img, kernel_size=param_converters["Median"](x)),
    "Sharp": lambda img, x: sharpening(
        img, sigma=1.0, alpha=param_converters["Sharp"](x)
    ),
}


try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS


class AttackGUI:
    """Professional GUI application for applying image attacks."""

    def __init__(self, root):
        self.root = root
        self.root.title("Professional Image Attack Tool (Zoom/Pan)")

        # Set correct DPI scaling (Windows only)
        self._setup_dpi_awareness()

        # State variables
        self.cv_image_original = None
        self.cv_image_modified = None
        self.tk_image = None  # PhotoImage (ALWAYS FULL RESOLUTION)
        self.image_on_canvas = None  # Reference to the image object on the canvas
        self.image_path = None

        # Selection variables
        self.selection_rect = None  # Reference to the selection rectangle
        self.start_x = 0
        self.start_y = 0
        # Store selection in image coordinates (x, y, w, h) to preserve during zoom
        self.selection_image_coords = None

        # Zoom state
        self.zoom = 1.0
        self.ZOOM_MIN = 0.05
        self.ZOOM_MAX = 8.0

        # Control variables
        # --- FIX: Set a default value for the attack variable ---
        self.attack_var = tk.StringVar(value=list(attack_config.keys())[0])
        # Single stepped slider: 0..100 mapped to 0.00..1.00
        self.step_strength_var = tk.IntVar(value=50)
        self.strength_label_var = tk.StringVar()

        # Detection state
        self.detection_ctx = None  # dict with keys: base_name, image_number, original_path, watermarked_path, project_root, detection_function

        # Style for widgets
        style = ttk.Style()
        style.configure("TLabel", padding=5)
        style.configure("TButton", padding=5)
        style.configure("TFrame", padding=10)
        style.configure("TLabelframe.Label", font="-weight bold")

        # --- Main Layout ---
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Control Panel (left)
        self._create_control_panel()

        # Vertical separator
        ttk.Separator(self.main_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=5
        )

        # Image Canvas (right)
        self._create_image_canvas()

        # --- FIX: Update the strength label on startup ---
        self.update_strength_label()

    def _setup_dpi_awareness(self):
        """Set the application to be DPI-aware on Windows."""
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass  # Fails quietly on non-Windows

    def _create_control_panel(self):
        """Create the side frame with all controls."""
        control_frame = ttk.Frame(self.main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        control_frame.pack_propagate(False)  # Prevent frame from shrinking

        # --- File Section ---
        file_labelframe = ttk.LabelFrame(control_frame, text="File")
        file_labelframe.pack(fill=tk.X, pady=5)
        ttk.Button(file_labelframe, text="Load Image", command=self.load_image).pack(
            fill=tk.X, expand=True, padx=10, pady=5
        )
        ttk.Button(file_labelframe, text="Save Image", command=self.save_image).pack(
            fill=tk.X, expand=True, padx=10, pady=5
        )
        ttk.Button(file_labelframe, text="Reset Image", command=self.reset_image).pack(
            fill=tk.X, expand=True, padx=10, pady=(5, 10)
        )

        # --- View Section ---
        view_labelframe = ttk.LabelFrame(control_frame, text="View")
        view_labelframe.pack(fill=tk.X, pady=5)
        ttk.Button(
            view_labelframe, text="Reset View (Fit)", command=self.fit_to_window
        ).pack(fill=tk.X, padx=10, pady=5)

        help_text = (
            "Keys +/-: Zoom (under cursor)\n"
            "Key 0: Reset view (fit)\n"
            "Arrow keys: Pan image\n"
            "Right-Click + Drag: Pan\n"
            "Left-Click + Drag: Select"
        )
        ttk.Label(view_labelframe, text=help_text, justify=tk.LEFT).pack(
            padx=10, pady=5
        )

        # --- Attack Section ---
        attack_labelframe = ttk.LabelFrame(control_frame, text="Attack Parameters")
        attack_labelframe.pack(fill=tk.X, pady=5)

        ttk.Label(attack_labelframe, text="Attack Type:").pack(
            anchor=tk.W, padx=10, pady=(5, 0)
        )
        attack_menu = ttk.Combobox(
            attack_labelframe,
            textvariable=self.attack_var,
            values=list(attack_config.keys()),
            state="readonly",
        )
        attack_menu.pack(fill=tk.X, padx=10, pady=5)
        attack_menu.bind("<<ComboboxSelected>>", self.update_strength_label)

        ttk.Label(attack_labelframe, text="Strength (0.0 - 1.0):").pack(
            anchor=tk.W, padx=10, pady=(5, 0)
        )

        # --- FIX: Removed problematic font property, added foreground color ---
        strength_feedback_label = ttk.Label(
            attack_labelframe,
            textvariable=self.strength_label_var,
            foreground="gray",  # Use a dimmer color
        )
        strength_feedback_label.pack(anchor=tk.W, padx=10)

        # Stepped slider (0..100) -> 0.00..1.00
        slider_frame = ttk.Frame(attack_labelframe)
        slider_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        tk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=True,
            variable=self.step_strength_var,
            command=lambda v: self.update_strength_label(),
        ).pack(fill=tk.X)

        # --- Actions Section ---
        self.action_labelframe = ttk.LabelFrame(control_frame, text="Actions")
        self.action_labelframe.pack(fill=tk.X, pady=5)
        ttk.Button(
            self.action_labelframe, text="Apply Attack", command=self.apply_attack
        ).pack(fill=tk.X, padx=10, pady=10)

        # Detection button (shown only when detection context is available)
        self.detect_button = ttk.Button(
            self.action_labelframe, text="Run Detection", command=self.run_detection
        )

    def _create_image_canvas(self):
        """Create the central canvas for the image."""
        canvas_frame = ttk.Frame(self.main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="gray", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- Selection (Left-Click) ---
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # --- Pan (Right Click) ---
        self.canvas.bind("<ButtonPress-3>", self.on_pan_press)
        self.canvas.bind("<B3-Motion>", self.on_pan_motion)

        # --- Zoom (Keyboard) ---
        self.root.bind("<KeyPress-plus>", self.on_key_zoom)  # + key
        self.root.bind("<KeyPress-equal>", self.on_key_zoom)  # = key (often shares +)
        self.root.bind("<KeyPress-minus>", self.on_key_zoom)  # - key

        self.root.bind_all("<KeyPress-0>", lambda e: self.fit_to_window())
        self.root.bind_all("<KeyPress-a>", lambda e: self.apply_attack())
        self.root.bind_all("<KeyPress-o>", lambda e: self.load_image())
        self.root.bind_all("<KeyPress-c>", lambda e: self.clear_selection())
        self.root.bind_all("<KeyPress-s>", lambda e: self.save_image())

        # Arrow keys for panning
        self.root.bind("<Left>", self.on_arrow_pan)
        self.root.bind("<Right>", self.on_arrow_pan)
        self.root.bind("<Up>", self.on_arrow_pan)
        self.root.bind("<Down>", self.on_arrow_pan)

        # Window resize event
        self.canvas.bind("<Configure>", self.on_resize_window)

        self.last_resize_id = None

    def update_strength_label(self, *args):
        """Update the label showing the real attack parameter."""
        try:
            strength = self.get_strength_value()
            attack_name = self.attack_var.get()

            if attack_name in param_converters:
                actual_param = param_converters[attack_name](strength)

                if isinstance(actual_param, (float, np.floating)):
                    param_display = f"{actual_param:.2f}"
                else:
                    param_display = f"{actual_param}"

                text = f"Actual value -> {attack_name}: {param_display}"
                self.strength_label_var.set(text)
            else:
                self.strength_label_var.set("Select an attack")
        except Exception:
            self.strength_label_var.set("Invalid parameter")

    def get_strength_value(self):
        """Return current strength in [0.0, 1.0], computed from selected input mode."""
        val = int(self.step_strength_var.get())
        val = max(0, min(100, val))
        return round(val / 100.0, 2)

    def load_image(self):
        """Open a file dialog to load an image."""
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.bmp *.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            # Force grayscale workflow
            self.cv_image_original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.cv_image_original is None:
                raise ValueError("Could not read image file.")

            self.cv_image_modified = self.cv_image_original.copy()
            self.image_path = path
            self.display_image(self.cv_image_modified)

            # Check if image filename matches watermarked pattern (name_number.ext)
            self._check_and_run_detection(path)

        except Exception as e:
            messagebox.showerror("Loading Error", f"Could not load image file:\n{e}")

    def _check_and_run_detection(self, image_path):
        """Check if the loaded image follows the watermarked pattern and run detection."""
        import re
        import importlib.util
        import sys

        # Extract filename without path and extension
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]

        # Check if filename contains underscore (e.g., luigi_lucia, crispymcmark_0000)
        # Split on last underscore to get base_suffix pattern
        if "_" not in name_without_ext:
            # Filename doesn't match watermarked pattern, clear detection context and skip
            self.detection_ctx = None
            self._update_detection_button_visibility()
            return

        # Split on the last underscore
        parts = name_without_ext.rsplit("_", 1)
        if len(parts) != 2:
            self.detection_ctx = None
            self._update_detection_button_visibility()
            return

        base_name = parts[0]  # e.g., "crispymcmark" or "luigi"
        suffix = parts[1]  # e.g., "0000" or "lucia"

        # Construct detection module filename and locate it (current dir or parent project dir)
        detection_module_name = f"detection_{base_name}"
        dir_path = os.path.dirname(image_path)
        project_root = os.path.dirname(dir_path)
        candidate1 = os.path.join(dir_path, f"{detection_module_name}.py")
        candidate2 = os.path.join(project_root, f"{detection_module_name}.py")
        detection_file = None
        if os.path.exists(candidate1):
            detection_file = candidate1
        elif os.path.exists(candidate2):
            detection_file = candidate2

        # Check if detection file exists
        if not detection_file:
            print(
                f"[DETECTION] No detection module found in {dir_path} or {project_root} for {detection_module_name}.py"
            )
            self.detection_ctx = None
            self._update_detection_button_visibility()
            return

        try:
            # Dynamically import the detection module
            spec = importlib.util.spec_from_file_location(
                detection_module_name, detection_file
            )
            if spec is None or spec.loader is None:
                print(f"[DETECTION] Could not load module spec from {detection_file}")
                return

            detection_module = importlib.util.module_from_spec(spec)
            sys.modules[detection_module_name] = detection_module
            spec.loader.exec_module(detection_module)

            # Check if detection function exists
            if not hasattr(detection_module, "detection"):
                print(
                    f"[DETECTION] Module {detection_module_name} does not have 'detection' function"
                )
                return

            detection_function = detection_module.detection

            # Prepare paths for detection
            # Extract base name and suffix from filename (e.g., crispymcmark_0005 or luigi_lucia)
            ext = os.path.splitext(filename)[1]

            # Build paths using project structure the user described
            import glob

            # 1) ORIGINAL (no watermark): in challenge_images/, arbitrary filename (letters/numbers) but containing the suffix
            #    Strategy: find any file in challenge_images that ends with the target suffix before extension
            original_dir = os.path.join(project_root, "challenge_images")
            original_path = None
            if os.path.isdir(original_dir):
                exts = ["bmp", "png", "jpg", "jpeg", "tif", "tiff"]
                candidates = []
                for e in exts:
                    candidates.extend(
                        glob.glob(os.path.join(original_dir, f"*{suffix}.{e}"))
                    )
                # Prefer exact basename == suffix (e.g., lucia.bmp or 0005.bmp) if present, otherwise first candidate
                exact = [
                    p
                    for p in candidates
                    if os.path.splitext(os.path.basename(p))[0] == suffix
                ]
                if exact:
                    original_path = exact[0]
                elif candidates:
                    original_path = candidates[0]
                else:
                    original_path = None

            # 2) WATERMARKED (no attack): in watermarked_groups_images/{base_name}_{suffix}.*
            wm_dir = os.path.join(project_root, "watermarked_groups_images")
            watermarked_path = os.path.join(wm_dir, f"{base_name}_{suffix}{ext}")
            if not os.path.exists(watermarked_path):
                # Fallback to any known extension
                exts = ["bmp", "png", "jpg", "jpeg"]
                found = None
                for e in exts:
                    cand = os.path.join(wm_dir, f"{base_name}_{suffix}.{e}")
                    if os.path.exists(cand):
                        found = cand
                        break
                watermarked_path = found or watermarked_path

            # 3. Attacked: current loaded image
            attacked_path = image_path

            # Verify that required files exist
            if not original_path or not os.path.exists(original_path):
                print(f"[DETECTION] Original image not found: {original_path}")
                self.detection_ctx = None
                self._update_detection_button_visibility()
                return

            if not watermarked_path or not os.path.exists(watermarked_path):
                print(f"[DETECTION] Watermarked image not found: {watermarked_path}")
                self.detection_ctx = None
                self._update_detection_button_visibility()
                return

            # Save detection context for later manual invocation
            self.detection_ctx = {
                "base_name": base_name,
                "image_suffix": suffix,
                "original_path": original_path,
                "watermarked_path": watermarked_path,
                "project_root": project_root,
                "detection_function": detection_function,
            }
            self._update_detection_button_visibility()

            # Run detection
            print(f"\n{'='*60}")
            print(f"[DETECTION] Running watermark detection for: {filename}")
            print(f"[DETECTION] Base name: {base_name}, Suffix: {suffix}")
            print(f"[DETECTION] Original: {os.path.basename(original_path)}")
            print(f"[DETECTION] Watermarked: {os.path.basename(watermarked_path)}")
            print(f"[DETECTION] Attacked: {os.path.basename(attacked_path)}")
            print(f"{'='*60}")

            result = detection_function(original_path, watermarked_path, attacked_path)

            # Handle different return types (tuple or single value)
            if isinstance(result, tuple):
                detected = result[0]
                wpsnr_value = result[1] if len(result) > 1 else None
                print(
                    f"[DETECTION RESULT] Watermark detected: {'YES' if detected else 'NO'}"
                )
                if wpsnr_value is not None:
                    print(f"[DETECTION METRIC] WPSNR: {wpsnr_value:.2f} dB")
            else:
                detected = result
                print(
                    f"[DETECTION RESULT] Watermark detected: {'YES' if detected else 'NO'}"
                )

            print(f"{'='*60}\n")

        except Exception as e:
            print(f"[DETECTION ERROR] Failed to run detection: {e}")
            import traceback

            traceback.print_exc()

    def _update_detection_button_visibility(self):
        """Show or hide the Run Detection button based on available context."""
        try:
            show = (
                self.detection_ctx is not None
                and isinstance(self.detection_ctx, dict)
                and all(
                    k in self.detection_ctx
                    for k in (
                        "base_name",
                        "image_suffix",
                        "original_path",
                        "watermarked_path",
                        "project_root",
                        "detection_function",
                    )
                )
            )
            # Also require current image to exist
            show = show and (self.cv_image_modified is not None)

            is_mapped = self.detect_button.winfo_ismapped()
            if show and not is_mapped:
                self.detect_button.pack(fill=tk.X, padx=10, pady=(0, 10))
            elif not show and is_mapped:
                self.detect_button.pack_forget()
        except Exception:
            # Fail-safe: do nothing if widget not yet created
            pass

    def run_detection(self):
        """Run detection on the current modified image, saving it temporarily if needed."""
        if not self.detection_ctx:
            messagebox.showwarning(
                "Detection", "Detection context not available for this image."
            )
            return

        try:
            base_name = self.detection_ctx["base_name"]
            suffix = self.detection_ctx["image_suffix"]
            original_path = self.detection_ctx["original_path"]
            watermarked_path = self.detection_ctx["watermarked_path"]
            project_root = self.detection_ctx["project_root"]
            detection_function = self.detection_ctx["detection_function"]

            # Save current modified image to a temporary attacked file
            tmp_dir = os.path.join(project_root, "tmp_attacks")
            os.makedirs(tmp_dir, exist_ok=True)
            attacked_tmp_path = os.path.join(
                tmp_dir, f"{base_name}_{suffix}_attacked_tmp.bmp"
            )
            cv2.imwrite(attacked_tmp_path, self.cv_image_modified)

            print(f"\n{'='*60}")
            print("[DETECTION] Manual run on current modified image")
            print(f"[DETECTION] Original: {os.path.basename(original_path)}")
            print(f"[DETECTION] Watermarked: {os.path.basename(watermarked_path)}")
            print(f"[DETECTION] Attacked (temp): {os.path.basename(attacked_tmp_path)}")
            print(f"{'='*60}")

            result = detection_function(
                original_path, watermarked_path, attacked_tmp_path
            )

            if isinstance(result, tuple):
                detected = result[0]
                wpsnr_value = result[1] if len(result) > 1 else None
            else:
                detected = result
                wpsnr_value = None

            detected_str = "Presente" if detected else "Assente"
            wpsnr_str = f"{wpsnr_value:.2f} dB" if wpsnr_value is not None else "n/a"

            # Terminal output
            print(f"[DETECTION RESULT] Watermark: {detected_str}")
            print(f"[DETECTION METRIC] WPSNR: {wpsnr_str}")
            print(f"{'='*60}\n")

            # GUI feedback
            messagebox.showinfo(
                "Detection Result",
                f"Watermark: {detected_str}\nWPSNR: {wpsnr_str}",
            )

            # Optional: cleanup temporary file
            try:
                os.remove(attacked_tmp_path)
            except Exception:
                pass

        except Exception as e:
            messagebox.showerror("Detection Error", f"Failed to run detection:\n{e}")

    def display_image(self, cv_image):
        """Convert and set the OpenCV image on the canvas."""
        if cv_image is None:
            return

        try:
            # Handle grayscale (2D) natively; color (3D) via RGB conversion
            if cv_image.ndim == 2:
                pil_image = Image.fromarray(cv_image)
            else:
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            self.tk_image = ImageTk.PhotoImage(pil_image)  # Full-resolution image

            self.canvas.delete("all")
            self.image_on_canvas = self.canvas.create_image(
                0, 0, anchor=tk.CENTER, image=self.tk_image, tags="image"
            )

            # Reset zoom and fit to window
            self.zoom = 1.0
            self.fit_to_window()
            self.clear_selection()

        except Exception as e:
            messagebox.showerror("Display Error", f"Could not display image:\n{e}")

    def on_resize_window(self, event):
        """Fit image to window on resize, with a delay."""
        if self.last_resize_id:
            self.root.after_cancel(self.last_resize_id)
        self.last_resize_id = self.root.after(100, self.fit_to_window)

    def fit_to_window(self):
        """Scale and center the image to fit the window."""
        if self.image_on_canvas is None or self.cv_image_modified is None:
            return

        # Ensure canvas geometry is up to date
        self.canvas.update_idletasks()

        img_h, img_w = self.cv_image_modified.shape[:2]
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())

        # Compute scale to fully fit the image into the canvas area
        scale = min(canvas_width / img_w, canvas_height / img_h)
        scale = max(self.ZOOM_MIN, min(self.ZOOM_MAX, scale))

        # Center coordinates of the canvas
        cx = canvas_width / 2
        cy = canvas_height / 2

        # Directly set zoom and re-render, then center the image explicitly.
        # This avoids cumulative drift that can occur when using the
        # point-preserving mapping during an initial fit.
        self.zoom = scale
        self._render_scaled_image()
        self.canvas.coords(self.image_on_canvas, cx, cy)

        # Update scrollregion and redraw any selection overlay at new scale
        self.canvas.configure(scrollregion=self.canvas.bbox(self.image_on_canvas))
        self._redraw_selection_from_image_coords()

    # --- Selection (Left-Click) ---

    def on_press(self, event):
        """Start selection (Left-Click)."""
        if self.tk_image is None:
            return

        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if self.selection_rect:
            self.canvas.delete(self.selection_rect)

        self.selection_rect = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=2,
            dash=(5, 5),
            tags="selection",
        )

    def on_drag(self, event):
        """Update selection rectangle (Left-Click)."""
        if self.selection_rect is None:
            return

        curr_x = self.canvas.canvasx(event.x)
        curr_y = self.canvas.canvasy(event.y)

        # Allow free-form rectangle selection (no forced square)
        self.canvas.coords(
            self.selection_rect, self.start_x, self.start_y, curr_x, curr_y
        )

    def on_release(self, event):
        """Finalize selection (Left-Click)."""
        if self.selection_rect:
            coords = self.canvas.coords(self.selection_rect)
            if len(coords) < 4 or coords[0] == coords[2] or coords[1] == coords[3]:
                self.clear_selection()
            else:
                img_coords = self._get_selection_in_image_coords()
                if not img_coords:
                    self.clear_selection()  # Selection was outside image bounds
                else:
                    # Store selection in image coordinates
                    self.selection_image_coords = img_coords

    # --- CORRECTED Zoom (Keyboard) ---

    def on_key_zoom(self, event):
        """Handle zoom via + and - keys, centered on the view."""
        if self.image_on_canvas is None or self.cv_image_modified is None:
            return

        if event.keysym in ("plus", "KP_Add", "equal"):
            factor = 1.1  # Zoom in 10%
        elif event.keysym in ("minus", "KP_Subtract"):
            factor = 0.9  # Zoom out 10%
        else:
            return

        # Use the current pointer position relative to canvas for cursor-centric zoom
        try:
            px_screen = self.canvas.winfo_pointerx()
            py_screen = self.canvas.winfo_pointery()
            cx_widget = px_screen - self.canvas.winfo_rootx()
            cy_widget = py_screen - self.canvas.winfo_rooty()
            cx = self.canvas.canvasx(cx_widget)
            cy = self.canvas.canvasy(cy_widget)
        except Exception:
            # Fallback to center of view if pointer lookup fails
            cx = self.canvas.winfo_width() / 2
            cy = self.canvas.winfo_height() / 2

        self._set_zoom(self.zoom * factor, center_canvas=(cx, cy))

    # --- Pan (Right-Click) ---

    def on_pan_press(self, event):
        """Start pan (Right-Click)."""
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_motion(self, event):
        """Move the canvas (Right-Click)."""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def clear_selection(self):
        """Remove the selection rectangle."""
        self.canvas.delete("selection")
        self.selection_rect = None
        self.selection_image_coords = None

    def _redraw_selection_from_image_coords(self):
        """Redraw selection rectangle from stored image coordinates after zoom/pan."""
        if self.selection_image_coords is None or self.image_on_canvas is None:
            return

        x, y, w, h = self.selection_image_coords

        # Get image bounding box (in canvas space)
        img_bbox = self.canvas.bbox(self.image_on_canvas)
        if img_bbox is None:
            return

        img_c_x1, img_c_y1, img_c_x2, _ = img_bbox

        # Get original image dimensions
        img_orig_h, img_orig_w = self.cv_image_modified.shape[:2]
        if img_orig_w == 0 or img_orig_h == 0:
            return

        # Calculate current scale
        current_scale = (img_c_x2 - img_c_x1) / img_orig_w
        if current_scale == 0:
            return

        # Map image coords back to canvas coords
        sel_c_x1 = img_c_x1 + x * current_scale
        sel_c_y1 = img_c_y1 + y * current_scale
        sel_c_x2 = img_c_x1 + (x + w) * current_scale
        sel_c_y2 = img_c_y1 + (y + h) * current_scale

        # Redraw selection rectangle
        self.canvas.delete("selection")
        self.selection_rect = self.canvas.create_rectangle(
            sel_c_x1,
            sel_c_y1,
            sel_c_x2,
            sel_c_y2,
            outline="red",
            width=2,
            dash=(5, 5),
            tags="selection",
        )

    def _get_selection_in_image_coords(self):
        """
        Convert canvas selection rectangle coordinates to
        original image pixel coordinates, accounting for zoom and pan.
        """
        if self.selection_rect is None or self.image_on_canvas is None:
            return None

        # 1. Get selection rectangle coords (in canvas space)
        sel_c_x1, sel_c_y1, sel_c_x2, sel_c_y2 = self.canvas.coords(self.selection_rect)

        # 2. Get image bounding box (in canvas space)
        img_c_x1, img_c_y1, img_c_x2, _ = self.canvas.bbox(self.image_on_canvas)

        if img_c_x1 is None:
            return None  # Image not visible

        # 3. Get original image dimensions
        img_orig_h, img_orig_w = self.cv_image_modified.shape[:2]

        if img_orig_w == 0 or img_orig_h == 0:
            return None

        # 4. Calculate current scale
        current_scale = (img_c_x2 - img_c_x1) / img_orig_w
        if current_scale == 0:
            return None

        # 5. Get current offset (top-left corner of image on canvas)
        offset_x = img_c_x1
        offset_y = img_c_y1

        # 6. Map selection coords to pixel coords
        i_x1 = int((min(sel_c_x1, sel_c_x2) - offset_x) / current_scale)
        i_y1 = int((min(sel_c_y1, sel_c_y2) - offset_y) / current_scale)
        i_x2 = int((max(sel_c_x1, sel_c_x2) - offset_x) / current_scale)
        i_y2 = int((max(sel_c_y1, sel_c_y2) - offset_y) / current_scale)

        # 7. Clip to image boundaries
        i_x1 = max(0, i_x1)
        i_y1 = max(0, i_y1)
        i_x2 = min(img_orig_w, i_x2)
        i_y2 = min(img_orig_h, i_y2)

        if i_x1 >= i_x2 or i_y1 >= i_y2:
            return None

        # Return (x, y, w, h) for OpenCV
        return (i_x1, i_y1, i_x2 - i_x1, i_y2 - i_y1)

    def on_arrow_pan(self, event):
        """Pan the image using arrow keys."""
        if self.image_on_canvas is None:
            return

        if event.keysym == "Left":
            self.canvas.xview_scroll(-1, "units")
        elif event.keysym == "Right":
            self.canvas.xview_scroll(1, "units")
        elif event.keysym == "Up":
            self.canvas.yview_scroll(-1, "units")
        elif event.keysym == "Down":
            self.canvas.yview_scroll(1, "units")

    def apply_attack(self):
        """Apply the selected attack."""
        if self.cv_image_modified is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        roi_coords = self._get_selection_in_image_coords()

        if roi_coords is None:
            if not messagebox.askyesno(
                "Global Attack", "No region selected. Apply attack to the entire image?"
            ):
                return
            h, w = self.cv_image_modified.shape[:2]
            roi_coords = (0, 0, w, h)

        attack_name = self.attack_var.get()
        strength = self.get_strength_value()
        attack_func = attack_config.get(attack_name)

        if not attack_func:
            messagebox.showerror("Error", f"Attack function '{attack_name}' not found.")
            return

        try:
            x, y, w, h = roi_coords
            roi = self.cv_image_modified[y : y + h, x : x + w]

            attack_name = self.attack_var.get()

            # Custom handling for Resize to avoid holes due to size rounding
            if attack_name == "Resize":
                scale = param_converters["Resize"](strength)
                # Ensure minimal size of 1x1
                target_w = max(1, int(round(w * scale)))
                target_h = max(1, int(round(h * scale)))

                # Work with grayscale ROI (2D). If color, convert to gray first.
                if roi.ndim == 3 and roi.shape[2] == 3:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi

                down = cv2.resize(
                    roi_gray, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
                up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
                attacked_roi = up.astype(np.uint8)

                # If original was color (unlikely in grayscale workflow), expand back
                if roi.ndim == 3 and roi.shape[2] == 3:
                    attacked_roi = cv2.cvtColor(attacked_roi, cv2.COLOR_GRAY2BGR)

            else:
                # Default path: call the configured attack function
                attacked_roi = attack_func(roi.copy(), strength)

            self.cv_image_modified[y : y + h, x : x + w] = attacked_roi

            # Mantieni la vista corrente e ri-renderizza all'attuale zoom
            current_view = (self.canvas.xview(), self.canvas.yview())
            self._render_scaled_image()
            self.canvas.xview_moveto(current_view[0][0])
            self.canvas.yview_moveto(current_view[1][0])

            # self.clear_selection()
            # print attack and converted parameter
            print(
                f"Applied attack '{attack_name}' {param_converters[attack_name](strength)} "
                f"on region x={x}, y={y}, w={w}, h={h}"
            )

        except Exception as e:
            messagebox.showerror(
                "Attack Error", f"An error occurred while applying the attack:\n{e}"
            )

    def reset_image(self):
        """Restore the image to its original state."""
        if self.cv_image_original is None:
            return

        self.cv_image_modified = self.cv_image_original.copy()
        self.display_image(self.cv_image_modified)  # display_image also resets the view
        print("Image restored to original.")

    def save_image(self):
        """Save the modified image to a new file."""
        if self.cv_image_modified is None:
            messagebox.showwarning("No Image", "No image to save.")
            return

        if self.image_path:
            name, _ = os.path.splitext(os.path.basename(self.image_path))
            initial_file = f"crispymcmark_{name}.bmp"
        else:
            initial_file = "attacked_image.bmp"

        path = filedialog.asksaveasfilename(
            initialfile=initial_file,
            defaultextension=".bmp",
            filetypes=[("BMP", "*.bmp"), ("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return

        try:
            cv2.imwrite(path, self.cv_image_modified)
            print(f"Image saved to: {path}")
            messagebox.showinfo(
                "Save Complete", f"Image saved successfully to:\n{path}"
            )
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save image:\n{e}")

    def _render_scaled_image(self):
        """Render the current cv_image_modified at the current zoom."""
        if self.cv_image_modified is None or self.image_on_canvas is None:
            return

        img_h, img_w = self.cv_image_modified.shape[:2]
        new_w = max(1, int(round(img_w * self.zoom)))
        new_h = max(1, int(round(img_h * self.zoom)))

        # Prepare PIL image depending on grayscale/color
        if self.cv_image_modified.ndim == 2:
            pil_image = Image.fromarray(self.cv_image_modified)
        else:
            image_rgb = cv2.cvtColor(self.cv_image_modified, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

        if new_w == img_w and new_h == img_h:
            pil_scaled = pil_image
        else:
            pil_scaled = pil_image.resize((new_w, new_h), RESAMPLE)

        self.tk_image = ImageTk.PhotoImage(pil_scaled)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)
        # Aggiorna scrollregion per pan consistente
        self.canvas.configure(scrollregion=self.canvas.bbox(self.image_on_canvas))

    def _set_zoom(self, new_zoom, center_canvas=None):
        """Set zoom keeping a given canvas point fixed."""
        if self.image_on_canvas is None or self.cv_image_modified is None:
            return

        # Clamp zoom
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, new_zoom))

        # Se non è cambiato, non fare nulla
        if abs(new_zoom - self.zoom) < 1e-6:
            return

        # Punto di riferimento (centro della vista se non fornito)
        if center_canvas is None:
            cx = self.canvas.winfo_width() / 2
            cy = self.canvas.winfo_height() / 2
        else:
            cx, cy = center_canvas

        # Coord ancoraggio attuali dell'immagine
        ax0, ay0 = self.canvas.coords(self.image_on_canvas)

        # Mappa il punto canvas in coordinate immagine (u,v) correnti
        zoom0 = self.zoom
        if zoom0 <= 0:
            zoom0 = 1.0
        u = (cx - ax0) / zoom0
        v = (cy - ay0) / zoom0

        # Aggiorna zoom e immagine renderizzata
        self.zoom = new_zoom
        self._render_scaled_image()

        # Calcola il nuovo ancoraggio per mantenere (u,v) sotto (cx,cy)
        ax1 = cx - u * self.zoom
        ay1 = cy - v * self.zoom
        self.canvas.coords(self.image_on_canvas, ax1, ay1)

        # Aggiorna scrollregion dopo spostamento
        self.canvas.configure(scrollregion=self.canvas.bbox(self.image_on_canvas))

        # Redraw selection if it exists
        self._redraw_selection_from_image_coords()


# --- Application Entry Point ---
if __name__ == "__main__":
    if not IMPORTS_OK:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Import Error",
            "Could not find 'attack_functions.py'.\n\n"
            "Please ensure 'attack_functions.py' is in "
            "the same folder as this script.",
        )
        root.destroy()
    else:
        root = tk.Tk()
        app = AttackGUI(root)
        root.geometry("1100x700")
        root.mainloop()
