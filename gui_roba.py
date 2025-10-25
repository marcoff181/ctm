import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# --- Funzioni di attacco (da attack_functions.py) ---
# Queste sono necessarie per far funzionare attack_config
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d
from skimage.transform import rescale as skimage_rescale


def awgn(img, std=5.0):
    """Aggiunge rumore Gaussiano all'immagine."""
    # Lavora su float per evitare problemi di clipping con la generazione del rumore
    img_float = img.astype(np.float32)
    noise = np.random.normal(0, std, img.shape)
    attacked = img_float + noise
    return np.clip(attacked, 0, 255).astype(np.uint8)


def blur(img, sigma=3.0):
    """Applica un filtro Gaussiano (sfocatura)."""
    # gaussian_filter gestisce correttamente i canali di colore
    result = gaussian_filter(img, sigma=[sigma, sigma, 0] if img.ndim == 3 else sigma)
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpening(img, sigma=1.0, alpha=1.5):
    """Applica un filtro di nitidezza (unsharp masking)."""
    img_float = img.astype(np.float32)
    blurred = gaussian_filter(
        img_float, sigma=[sigma, sigma, 0] if img.ndim == 3 else sigma
    )
    result = img_float + alpha * (img_float - blurred)
    return np.clip(result, 0, 255).astype(np.uint8)


def median(img, kernel_size=3):
    """Applica un filtro mediano."""
    if img.ndim == 3:
        # Applica il filtro a ogni canale separatamente
        r = medfilt2d(img[:, :, 0], kernel_size).astype(np.uint8)
        g = medfilt2d(img[:, :, 1], kernel_size).astype(np.uint8)
        b = medfilt2d(img[:, :, 2], kernel_size).astype(np.uint8)
        return cv2.merge([r, g, b])
    else:
        return medfilt2d(img, kernel_size).astype(np.uint8)


def resizing(img, scale=0.9):
    """Ridimensiona l'immagine (downscaling e upscaling) per simulare perdita."""
    h, w = img.shape[:2]
    # anti_aliasing=True è importante per un ridimensionamento corretto
    downscaled = skimage_rescale(
        img, scale, anti_aliasing=True, channel_axis=2 if img.ndim == 3 else None
    )

    # Riscala all'inverso per tornare (quasi) alla dimensione originale
    # `order=3` è interpolazione bicubica, buona qualità
    upscaled = skimage_rescale(
        downscaled,
        1 / scale,
        anti_aliasing=True,
        output_shape=(h, w),
        order=3,
        channel_axis=2 if img.ndim == 3 else None,
    )

    # I valori di rescale sono in [0, 1], riconverti a [0, 255]
    return np.clip(upscaled * 255, 0, 255).astype(np.uint8)


def jpeg_compression(img, quality=70):
    """Applica la compressione JPEG."""
    temp_path = "___temp_attack.jpg"

    # Converti da BGR (cv2) a RGB (PIL) se è a colori
    if img.ndim == 3:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = Image.fromarray(img)

    img_pil.save(temp_path, "JPEG", quality=quality)

    # Ricarica l'immagine
    result_pil = Image.open(temp_path)
    result_array = np.asarray(result_pil, dtype=np.uint8)
    os.remove(temp_path)

    # Riconverti da RGB a BGR se a colori
    if result_array.ndim == 3:
        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
    else:
        return result_array


# --- Configurazione Attacchi (da attack.py) ---
#
param_converters = {
    "JPEG": lambda x: int(round((1 - x) * 95) + 5),  # Qualità da 5 a 100
    "Blur": lambda x: x * 5.0 + 0.1,  # Sigma da 0.1 a 5.1
    "AWGN": lambda x: x * 50.0,  # Dev. std da 0 a 50
    "Resize": lambda x: 1.0 - (x * 0.9 + 0.05),  # Scala da 0.95 a 0.05
    "Median": lambda x: [1, 3, 5, 7][int(round(x * 3))],  # Kernel size 1, 3, 5, 7
    "Sharp": lambda x: x * 5.0,  # Alpha da 0 a 5.0
}

#
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


# --- Classe GUI ---
class AttackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Strumento Interattivo di Attacco Immagini")

        # Variabili di stato
        self.cv_image_original = None  # Immagine originale (NumPy BGR)
        self.cv_image_modified = None  # Immagine modificata (NumPy BGR)
        self.tk_image = None  # Immagine per Tkinter (PhotoImage)
        self.selection_coords = None  # (x1, y1, x2, y2)
        self.selection_rect = None  # Riferimento al rettangolo sul canvas
        self.start_x = None
        self.start_y = None

        # --- Pannello di Controllo ---
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(
            self.control_frame, text="Carica Immagine", command=self.load_image
        ).pack(fill=tk.X, pady=5)

        ttk.Separator(self.control_frame, orient="horizontal").pack(fill=tk.X, pady=10)

        ttk.Label(self.control_frame, text="Seleziona Attacco:").pack()
        self.attack_var = tk.StringVar(value=list(attack_config.keys())[0])
        self.attack_menu = ttk.OptionMenu(
            self.control_frame, self.attack_var, None, *attack_config.keys()
        )
        self.attack_menu.pack(fill=tk.X, pady=5)

        ttk.Label(self.control_frame, text="Forza Attacco (0.0 - 1.0):").pack()
        self.strength_slider = ttk.Scale(
            self.control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL
        )
        self.strength_slider.set(0.5)
        self.strength_slider.pack(fill=tk.X, pady=5)

        ttk.Button(
            self.control_frame,
            text="Applica Attacco alla Selezione",
            command=self.apply_attack,
        ).pack(fill=tk.X, pady=10)

        ttk.Button(
            self.control_frame, text="Reset Immagine", command=self.reset_image
        ).pack(fill=tk.X, pady=5)

        ttk.Separator(self.control_frame, orient="horizontal").pack(fill=tk.X, pady=10)

        ttk.Button(
            self.control_frame, text="Salva Immagine", command=self.save_image
        ).pack(fill=tk.X, pady=5)

        # --- Canvas per l'Immagine ---
        self.canvas_frame = ttk.Frame(self.root, relief=tk.SUNKEN, borderwidth=1)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Eventi per la selezione
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self):
        """Apre un file dialog per caricare un'immagine."""
        path = filedialog.askopenfilename(
            filetypes=[
                ("Immagini", "*.bmp *.png *.jpg *.jpeg"),
                ("Tutti i file", "*.*"),
            ]
        )
        if not path:
            return

        try:
            self.cv_image_original = cv2.imread(path)
            if self.cv_image_original is None:
                raise ValueError("Impossibile leggere il file immagine.")

            self.cv_image_modified = self.cv_image_original.copy()
            print(
                f"Immagine caricata: {path} (Dimensioni: {self.cv_image_original.shape})"
            )
            self.display_image(self.cv_image_modified)
            # Centra l'immagine nel canvas
            self.center_image()
            self.root.update()
            self.center_image()  # Chiama due volte per un centraggio corretto dopo l'aggiornamento

        except Exception as e:
            messagebox.showerror(
                "Errore Caricamento", f"Impossibile caricare l'immagine:\n{e}"
            )

    def display_image(self, cv_image):
        """Converte un'immagine OpenCV (BGR) e la mostra sul canvas."""
        if cv_image is None:
            return

        # Converte da BGR (cv2) a RGB (PIL)
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Converte in formato PIL
        pil_image = Image.fromarray(image_rgb)
        # Converte in formato Tkinter
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Aggiorna il canvas
        self.canvas.delete("all")
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="image")

        # Pulisce la selezione precedente
        self.selection_rect = None
        self.selection_coords = None

    def center_image(self):
        """Centra l'immagine visibile nel canvas."""
        if self.tk_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width = self.tk_image.width()
        img_height = self.tk_image.height()

        # Calcola le coordinate per centrare
        x = (canvas_width - img_width) / 2
        y = (canvas_height - img_height) / 2

        self.canvas.coords("image", x, y)

    def on_press(self, event):
        """Inizia la selezione."""
        if self.tk_image is None:
            return

        # Converte le coordinate dell'evento (canvas) in coordinate dell'immagine
        img_x, img_y = self.get_image_coords(event.x, event.y)

        self.start_x = img_x
        self.start_y = img_y

        # Rimuovi il vecchio rettangolo se esiste
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)

        # Crea un nuovo rettangolo (invisibile finché non ci si muove)
        self.selection_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=2, dash=(5, 5)
        )

    def on_drag(self, event):
        """Aggiorna il rettangolo di selezione (mantenendolo quadrato)."""
        if self.selection_rect is None:
            return

        # Coordinate immagine
        curr_x_img, curr_y_img = self.get_image_coords(event.x, event.y)

        delta_x = curr_x_img - self.start_x
        delta_y = curr_y_img - self.start_y

        # Forza la selezione ad essere un quadrato
        side = max(abs(delta_x), abs(delta_y))

        end_x_img = self.start_x + (side if delta_x > 0 else -side)
        end_y_img = self.start_y + (side if delta_y > 0 else -side)

        # Riconverti in coordinate canvas per disegnare
        start_x_canvas, start_y_canvas = self.get_canvas_coords(
            self.start_x, self.start_y
        )
        end_x_canvas, end_y_canvas = self.get_canvas_coords(end_x_img, end_y_img)

        self.canvas.coords(
            self.selection_rect,
            start_x_canvas,
            start_y_canvas,
            end_x_canvas,
            end_y_canvas,
        )

    def on_release(self, event):
        """Finalizza la selezione."""
        if self.selection_rect is None:
            return

        # Ottieni le coordinate finali del rettangolo sul canvas
        x1_c, y1_c, x2_c, y2_c = self.canvas.coords(self.selection_rect)

        # Converti in coordinate immagine
        x1_img, y1_img = self.get_image_coords(x1_c, y1_c)
        x2_img, y2_img = self.get_image_coords(x2_c, y2_c)

        # Assicura che (x1, y1) sia l'angolo in alto a sinistra
        x1 = int(min(x1_img, x2_img))
        y1 = int(min(y1_img, y2_img))
        x2 = int(max(x1_img, x2_img))
        y2 = int(max(y1_img, y2_img))

        # Fai il "clipping" (limita) alle dimensioni dell'immagine
        h, w = self.cv_image_modified.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) > 0 and (y2 - y1) > 0:
            self.selection_coords = (x1, y1, x2, y2)
            print(
                f"Selezione finalizzata (coordinate immagine): {self.selection_coords}"
            )
        else:
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
            self.selection_coords = None

    def get_image_coords(self, canvas_x, canvas_y):
        """Converte le coordinate del canvas in coordinate dell'immagine."""
        if self.tk_image is None:
            return 0, 0
        img_origin_x, img_origin_y = self.canvas.coords("image")
        return int(canvas_x - img_origin_x), int(canvas_y - img_origin_y)

    def get_canvas_coords(self, img_x, img_y):
        """Converte le coordinate dell'immagine in coordinate del canvas."""
        if self.tk_image is None:
            return 0, 0
        img_origin_x, img_origin_y = self.canvas.coords("image")
        return int(img_x + img_origin_x), int(img_y + img_origin_y)

    def apply_attack(self):
        """Applica l'attacco scelto alla regione selezionata (o a tutta l'immagine)."""
        if self.cv_image_modified is None:
            messagebox.showwarning(
                "Nessuna Immagine",
                "Per favore, carica un'immagine prima di applicare un attacco.",
            )
            return

        if self.selection_coords is None:
            if not messagebox.askyesno(
                "Attacco Globale",
                "Nessuna regione selezionata. Vuoi applicare l'attacco all'intera immagine?",
            ):
                return

            # Coordinate dell'intera immagine
            h, w = self.cv_image_modified.shape[:2]
            roi_coords = (0, 0, w, h)
            print("Applicazione attacco all'intera immagine...")
        else:
            roi_coords = self.selection_coords
            print(f"Applicazione attacco alla regione: {roi_coords}")

        # Ottieni i parametri dell'attacco
        attack_name = self.attack_var.get()
        strength = self.strength_slider.get()  # (concetto di slider 0-1)

        # Prendi la funzione di attacco dal dizionario
        attack_func = attack_config.get(attack_name)
        if not attack_func:
            messagebox.showerror(
                "Errore", f"Funzione di attacco '{attack_name}' non trovata."
            )
            return

        try:
            # Estrai la Regione di Interesse (ROI)
            x1, y1, x2, y2 = roi_coords
            roi = self.cv_image_modified[y1:y2, x1:x2]

            # Applica l'attacco
            # Usiamo .copy() per evitare modifiche inaspettate
            attacked_roi = attack_func(roi.copy(), strength)

            # Sostituisci la ROI nell'immagine modificata
            self.cv_image_modified[y1:y2, x1:x2] = attacked_roi

            # Aggiorna la visualizzazione
            self.display_image(self.cv_image_modified)
            self.center_image()

            print(f"Attacco '{attack_name}' con forza {strength:.2f} applicato.")

        except Exception as e:
            messagebox.showerror(
                "Errore Attacco", f"Errore durante l'applicazione dell'attacco:\n{e}"
            )
            # In caso di errore, ricarica l'immagine per sicurezza
            self_display_image(self.cv_image_modified)
            self.center_image()

    def reset_image(self):
        """Ripristina l'immagine allo stato originale."""
        if self.cv_image_original is None:
            return

        self.cv_image_modified = self.cv_image_original.copy()
        self.display_image(self.cv_image_modified)
        self.center_image()
        print("Immagine ripristinata all'originale.")

    def save_image(self):
        """Salva l'immagine modificata in un nuovo file."""
        if self.cv_image_modified is None:
            messagebox.showwarning("Nessuna Immagine", "Nessuna immagine da salvare.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("BMP", "*.bmp"), ("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return

        try:
            cv2.imwrite(path, self.cv_image_modified)
            print(f"Immagine salvata in: {path}")
            messagebox.showinfo(
                "Salvataggio Completato", f"Immagine salvata con successo in:\n{path}"
            )
        except Exception as e:
            messagebox.showerror(
                "Errore Salvataggio", f"Impossibile salvare l'immagine:\n{e}"
            )


# --- Avvio dell'Applicazione ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AttackGUI(root)
    root.geometry("1000x700")  # Dimensioni iniziali della finestra
    root.mainloop()
