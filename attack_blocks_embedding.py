import os
import cv2
import numpy as np
import importlib.util
from attack import discover_detection_functions, param_converters, attack_config
from wpsnr import wpsnr

WATERMARKED_IMAGES_PATH = "./watermarked_groups_images/"
ORIGINAL_IMAGES_PATH = "./challenge_images/"
OUTPUT_PATH = "./watermark_blocks_analysis/"

ATTACK_ITERATIONS = 10
MIN_REGION_SIZE = 4  # pixel

ATTACK_TYPES = ["Blur", "AWGN", "JPEG", "Resize", "Median", "Sharp"]


def binary_search_watermark_location(
    original_path,
    watermarked_path,
    detection_func,
    watermarked_img,
    region_coords,
    depth=0,
    best_attack=None,
):
    """
    Args:
        original_path: Path all'immagine originale
        watermarked_path: Path all'immagine watermarked
        detection_func: Funzione di detection
        watermarked_img: Array numpy dell'immagine watermarked
        region_coords: Coordinate correnti (start_y, end_y, start_x, end_x)
        depth: Profondità corrente della ricorsione
        best_attack: Tipo di attacco da usare (None = prova tutti)

    Returns:
        Tupla (lista di regioni critiche, best_attack usato)
    """
    start_y, end_y, start_x, end_x = region_coords
    height = end_y - start_y
    width = end_x - start_x

    indent = "  " * depth

    # Se non abbiamo ancora determinato il miglior attacco, proviamoli tutti
    if best_attack is None and depth == 0:
        print(f"\n{indent}[Testing all attacks on full image...]")
        for attack_type in ATTACK_TYPES:
            result = analyze_region(
                original_path,
                watermarked_path,
                watermarked_img,
                region_coords,
                detection_func,
                attack_type,
            )
            if result["detected"] == 0:
                best_attack = attack_type
                print(
                    f"{indent}✓ Found working attack: {attack_type} "
                    f"(WPSNR: {result['global_wpsnr']:.2f} dB)"
                )
                break
            else:
                print(f"{indent}✗ {attack_type} failed (still detected)")

        if best_attack is None:
            print(f"{indent}No attack succeeded - watermark is resilient!")
            return [], None

    # Testa se attaccando questa regione il watermark viene ancora rilevato
    result = analyze_region(
        original_path,
        watermarked_path,
        watermarked_img,
        region_coords,
        detection_func,
        best_attack,
    )

    detected = result["detected"]
    wpsnr = result["global_wpsnr"]
    local_wpsnr = result["local_wpsnr"]

    if detected:
        # Watermark ancora rilevato -> questa regione NON contiene (tutto) il watermark
        return [], best_attack
    else:
        # Watermark NON rilevato -> questa regione contiene watermark
        print(
            f"{indent}[Depth {depth}] ({start_y}:{end_y}, {start_x}:{end_x}) [{height}x{width}] "
            f"✗ CRITICA (WPSNR: {wpsnr:.2f} dB)"
        )

        # Se la regione è abbastanza piccola, fermati
        if height <= MIN_REGION_SIZE or width <= MIN_REGION_SIZE:
            return [result], best_attack

        # Altrimenti dividi in 4 sottoquadranti e analizza ricorsivamente
        critical_regions = []
        mid_y = start_y + height // 2
        mid_x = start_x + width // 2

        # Dividi in 4 quadranti
        quadrants = [
            (start_y, mid_y, start_x, mid_x),  # Top-left
            (start_y, mid_y, mid_x, end_x),  # Top-right
            (mid_y, end_y, start_x, mid_x),  # Bottom-left
            (mid_y, end_y, mid_x, end_x),  # Bottom-right
        ]

        for i, quad in enumerate(quadrants, 1):
            sub_results, _ = binary_search_watermark_location(
                original_path,
                watermarked_path,
                detection_func,
                watermarked_img,
                quad,
                depth + 1,
                best_attack,
            )
            critical_regions.extend(sub_results)

        # Se nessuna sottoregione è critica, logga questa regione come critica
        # (caso in cui la regione è critica ma dividendola tutte le parti sono non critiche)
        if len(critical_regions) == 0:
            if depth <= 2:
                print(f"{indent}  → No critical sub-regions, marking parent as leaf")
            return [result], best_attack

        return critical_regions, best_attack


def create_region_mask(img_shape, region_coords):
    """Crea una maschera booleana per una regione specifica.

    Args:
        img_shape: Shape dell'immagine (height, width)
        region_coords: Tupla (start_y, end_y, start_x, end_x)

    Returns:
        Maschera booleana con True nella regione specificata
    """
    mask = np.zeros(img_shape, dtype=bool)
    start_y, end_y, start_x, end_x = region_coords
    mask[start_y:end_y, start_x:end_x] = True
    return mask


def analyze_region(
    original_path,
    watermarked_path,
    watermarked_img,
    region_coords,
    detection_func,
    attack_type="JPEG",
):
    """Attacca una specifica regione al massimo e verifica la detection.

    Args:
        original_path: Path all'immagine originale
        watermarked_path: Path all'immagine watermarked
        watermarked_img: Array numpy dell'immagine watermarked
        region_coords: Coordinate della regione (start_y, end_y, start_x, end_x)
        detection_func: Funzione di detection da utilizzare
        attack_type: Tipo di attacco da usare

    Returns:
        dict con i risultati dell'analisi (include anche 'attacked_img')
    """
    start_y, end_y, start_x, end_x = region_coords

    # Crea maschera per questa regione
    mask = create_region_mask(watermarked_img.shape, region_coords)

    # Attacco con intensità massima ripetuto più volte
    attack = attack_config[attack_type]
    alpha_max = 1.0  # Massima intensità (quality = 0)

    # Applica l'attacco multiplo sulla stessa regione per distruggerla completamente
    attacked_img = watermarked_img.copy()
    for _ in range(ATTACK_ITERATIONS):
        full_attacked = attack(attacked_img.copy(), alpha_max)
        attacked_img = np.where(mask, full_attacked, attacked_img)

    # Salva temporaneamente per detection
    tmp_path = f"./tmp_attacks/region_test_{start_y}_{start_x}.bmp"
    os.makedirs("./tmp_attacks", exist_ok=True)
    cv2.imwrite(tmp_path, attacked_img)

    detected, wpsnr_val = detection_func(original_path, watermarked_path, tmp_path)

    # Calcola WPSNR locale della regione attaccata
    original = cv2.imread(original_path, 0)
    region_original = original[start_y:end_y, start_x:end_x]
    region_attacked = attacked_img[start_y:end_y, start_x:end_x]
    local_wpsnr = wpsnr(region_original, region_attacked)

    # Cleanup
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    # Convert parameter to string to handle both scalar and list values
    param_value = param_converters[attack_type](alpha_max)
    if isinstance(param_value, list):
        param_str = str(param_value)
    else:
        param_str = param_value

    return {
        "region": region_coords,
        "detected": detected,
        "global_wpsnr": wpsnr_val,
        "local_wpsnr": local_wpsnr,
        "quality": param_str,
        "attack_type": attack_type,
        "attack_iterations": ATTACK_ITERATIONS,
        "critical": detected == 1,  # Se ancora rilevato, la regione NON è critica
        "attacked_img": attacked_img,  # Salva l'immagine attaccata
    }


def visualize_results(results, output_path, image_name):
    """Crea una visualizzazione dei risultati con heatmap.

    Args:
        results: Lista di risultati dall'analisi
        output_path: Path dove salvare la visualizzazione
        image_name: Nome dell'immagine analizzata
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Determina dimensioni della griglia
    max_y = max(r["region"][1] for r in results)
    max_x = max(r["region"][3] for r in results)
    grid_size = results[0]["region"][1] - results[0]["region"][0]

    grid_rows = max_y // grid_size
    grid_cols = max_x // grid_size

    # Crea matrici per le visualizzazioni
    detection_grid = np.zeros((grid_rows, grid_cols))
    wpsnr_grid = np.zeros((grid_rows, grid_cols))

    for result in results:
        y, _, x, _ = result["region"]
        row = y // grid_size
        col = x // grid_size
        detection_grid[row, col] = 1 if result["detected"] else 0
        wpsnr_grid[row, col] = result["global_wpsnr"]

    # Crea figura con 2 subplot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Detection map
    im1 = ax1.imshow(
        detection_grid, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest"
    )
    ax1.set_title(
        f"Watermark Detection Map\n{image_name}", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Grid Column")
    ax1.set_ylabel("Grid Row")

    # Aggiungi griglia
    for i in range(grid_rows + 1):
        ax1.axhline(i - 0.5, color="black", linewidth=0.5)
    for j in range(grid_cols + 1):
        ax1.axvline(j - 0.5, color="black", linewidth=0.5)

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Detected (1) / Not Detected (0)", rotation=270, labelpad=20)

    # Plot 2: WPSNR heatmap
    im2 = ax2.imshow(wpsnr_grid, cmap="coolwarm", interpolation="nearest")
    ax2.set_title(
        f"WPSNR After Region Attack\n{image_name}", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Grid Column")
    ax2.set_ylabel("Grid Row")

    # Aggiungi griglia
    for i in range(grid_rows + 1):
        ax2.axhline(i - 0.5, color="black", linewidth=0.5)
    for j in range(grid_cols + 1):
        ax2.axvline(j - 0.5, color="black", linewidth=0.5)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("WPSNR (dB)", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n[INFO] Visualization saved to: {output_path}")
    plt.close()


def analyze_critical_regions(results):
    """Analizza i risultati per identificare le regioni critiche del watermark.

    Args:
        results: Lista di risultati dall'analisi

    Returns:
        dict con statistiche e regioni critiche
    """
    total_regions = len(results)
    detected_regions = sum(1 for r in results if r["detected"])
    not_detected_regions = total_regions - detected_regions

    # Le regioni critiche sono quelle dove il watermark NON viene più rilevato
    # quando vengono attaccate
    critical_regions = [r for r in results if not r["detected"]]
    non_critical_regions = [r for r in results if r["detected"]]

    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total regions analyzed: {total_regions}")
    print(
        f"Still detected after attack: {detected_regions} "
        f"({detected_regions/total_regions*100:.1f}%)"
    )
    print(
        f"NOT detected after attack: {not_detected_regions} "
        f"({not_detected_regions/total_regions*100:.1f}%)"
    )
    print(f"\n{'='*60}")
    print("CRITICAL REGIONS (contain watermark blocks):")
    print(f"{'='*60}")

    if critical_regions:
        for i, r in enumerate(critical_regions, 1):
            y1, y2, x1, x2 = r["region"]
            print(
                f"{i}. Region ({y1}:{y2}, {x1}:{x2}) | "
                f"WPSNR: {r['global_wpsnr']:.2f} dB | "
                f"Local WPSNR: {r['local_wpsnr']:.2f} dB"
            )
    else:
        print("No critical regions found (watermark sempre rilevato)")

    print(f"\n{'='*60}\n")

    return {
        "total_regions": total_regions,
        "detected_count": detected_regions,
        "not_detected_count": not_detected_regions,
        "critical_regions": critical_regions,
        "non_critical_regions": non_critical_regions,
    }


def main_binary_search():
    """Main function usando binary search spaziale."""

    # Setup
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs("./tmp_attacks", exist_ok=True)

    # Scopri funzioni di detection
    detection_functions = discover_detection_functions()

    if not detection_functions:
        print("[ERROR] No detection modules found.")
        return

    # Analizza tutte le immagini watermarked
    img_list = sorted(os.listdir(WATERMARKED_IMAGES_PATH))

    # Statistiche globali
    total_images = 0
    successful_attacks = 0
    saved_images = 0
    attacks_summary = {}  # {image_name: {attack, regions, wpsnr, saved}}

    for filename in img_list:
        if not filename.endswith(".bmp"):
            continue

        watermarked_path = os.path.join(WATERMARKED_IMAGES_PATH, filename)
        name_without_ext = os.path.splitext(filename)[0]

        if "_" not in name_without_ext:
            continue

        group_name, image_name = name_without_ext.split("_", 1)

        if group_name not in detection_functions:
            continue

        original_path = os.path.join(ORIGINAL_IMAGES_PATH, f"{image_name}.bmp")

        if not os.path.exists(original_path):
            continue

        total_images += 1
        print(f"\n{'='*60}")
        print(f"[{total_images}] Analyzing: {filename}")
        print(f"{'='*60}")

        # Load detection function from file path
        detection_file_path = detection_functions[group_name]
        try:
            module_name = f"detection_modules.{group_name}"
            spec = importlib.util.spec_from_file_location(
                module_name, detection_file_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create spec for {detection_file_path}")
            detection_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(detection_module)
            detection_func = detection_module.detection
        except Exception as e:
            print(f"[ERROR] Could not load detection function for {group_name}: {e}")
            continue

        watermarked_img = cv2.imread(watermarked_path, 0)
        height, width = watermarked_img.shape

        # Esegui binary search dall'intera immagine
        critical_regions, best_attack = binary_search_watermark_location(
            original_path,
            watermarked_path,
            detection_func,
            watermarked_img,
            (0, height, 0, width),
            depth=0,
            best_attack=None,
        )

        # Se nessuna regione critica trovata, salta
        if not critical_regions or best_attack is None:
            print(f"\n✗ No critical regions found - skipping PNG/CSV generation")
            attacks_summary[filename] = {
                "attack": "None",
                "regions": 0,
                "wpsnr": 0,
                "saved": False,
            }
            continue

        successful_attacks += 1

        # Analizza risultati (output limitato)
        print(f"\n{'─'*60}")
        print(f"✓ Found {len(critical_regions)} critical region(s) using {best_attack}")
        print(f"{'─'*60}")

        # Mostra solo le prime 3 e l'ultima regione (se ce ne sono di più)
        num_to_show = min(3, len(critical_regions))
        for i in range(num_to_show):
            r = critical_regions[i]
            y1, y2, x1, x2 = r["region"]
            h, w = y2 - y1, x2 - x1
            print(f"{i+1}. [{h}x{w}] WPSNR: {r['global_wpsnr']:.2f} dB")

        if len(critical_regions) > 3:
            print(f"... ({len(critical_regions) - 3} more regions)")
            r = critical_regions[-1]
            y1, y2, x1, x2 = r["region"]
            h, w = y2 - y1, x2 - x1
            print(
                f"{len(critical_regions)}. [{h}x{w}] WPSNR: {r['global_wpsnr']:.2f} dB (last)"
            )

        # Visualizza su griglia per confronto
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(watermarked_img, cmap="gray")
        ax.set_title(f"Critical Regions - {filename}", fontsize=14, fontweight="bold")

        # Disegna le regioni critiche
        for r in critical_regions:
            y1, y2, x1, x2 = r["region"]
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")

        output_viz_path = os.path.join(
            OUTPUT_PATH, f"{name_without_ext}_binary_search.png"
        )
        plt.tight_layout()
        plt.savefig(output_viz_path, dpi=200, bbox_inches="tight")
        print(f"[PNG] {output_viz_path}")
        plt.close()

        # Salva risultati in CSV
        import csv

        csv_path = os.path.join(OUTPUT_PATH, f"{name_without_ext}_binary_search.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "region_y1",
                    "region_y2",
                    "region_x1",
                    "region_x2",
                    "width",
                    "height",
                    "detected",
                    "global_wpsnr",
                    "local_wpsnr",
                    "attack_type",
                    "attack_iterations",
                    "critical",
                ],
            )
            writer.writeheader()
            for r in critical_regions:
                y1, y2, x1, x2 = r["region"]
                writer.writerow(
                    {
                        "region_y1": y1,
                        "region_y2": y2,
                        "region_x1": x1,
                        "region_x2": x2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "detected": r["detected"],
                        "global_wpsnr": r["global_wpsnr"],
                        "local_wpsnr": r["local_wpsnr"],
                        "attack_type": r.get("attack_type", best_attack),
                        "attack_iterations": r["attack_iterations"],
                        "critical": r["critical"],
                    }
                )

        print(f"[CSV] {csv_path}")

        # Salva l'immagine con l'ultima regione critica attaccata
        # Solo se: watermark NON rilevato (detected=0) E WPSNR > 35 dB
        last_critical = critical_regions[-1]
        image_saved = False

        if (
            "attacked_img" in last_critical
            and last_critical["detected"] == 0
            and last_critical["global_wpsnr"] > 35
        ):
            attacked_output_path = os.path.join(
                OUTPUT_PATH, f"{name_without_ext}_last_critical_attacked.bmp"
            )
            cv2.imwrite(attacked_output_path, last_critical["attacked_img"])

            y1, y2, x1, x2 = last_critical["region"]
            print(
                f"[BMP] {attacked_output_path} (WPSNR: {last_critical['global_wpsnr']:.2f} dB)"
            )
            image_saved = True
            saved_images += 1
        else:
            reason = []
            if last_critical["detected"] != 0:
                reason.append("still detected")
            if last_critical["global_wpsnr"] <= 35:
                reason.append(f"WPSNR={last_critical['global_wpsnr']:.2f}≤35")
            print(f"[BMP] Not saved ({', '.join(reason)})")

        # Aggiorna statistiche
        attacks_summary[filename] = {
            "attack": best_attack,
            "regions": len(critical_regions),
            "wpsnr": last_critical["global_wpsnr"],
            "saved": image_saved,
        }

    # RECAP FINALE
    print(f"\n\n{'='*70}")
    print(f"{'RECAP - BINARY SEARCH ANALYSIS':^70}")
    print(f"{'='*70}")
    print(f"Total images processed: {total_images}")
    print(f"Successful attacks: {successful_attacks}/{total_images}")
    print(
        f"Images saved: {saved_images}/{successful_attacks if successful_attacks > 0 else 0}"
    )
    print(f"{'='*70}\n")

    if attacks_summary:
        print("Detailed Results:")
        print(f"{'Image':<40} {'Attack':<15} {'Regions':>8} {'WPSNR':>8} {'Saved':>8}")
        print(f"{'-'*40} {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
        for img_name, data in attacks_summary.items():
            saved_icon = "✓" if data["saved"] else "✗"
            print(
                f"{img_name:<40} {data['attack']:<15} {data['regions']:>8} "
                f"{data['wpsnr']:>7.2f}  {saved_icon:>7}"
            )

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main_binary_search()
