### imports
import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import math
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

figure_dir = ""

###################################################################################
########################### TEIL 1: DEINE FFT ANALYSE #############################
###################################################################################

scenarios = [
    {
        "title": "Hourly (1h) - Kurzfristig",
        "interval": "1h",
        "is_short_term": True,
        "hours_per_unit": 1
    },
    {
        "title": "Daily (1d) - Mittelfristig",
        "interval": "1d",
        "is_short_term": False,
        "hours_per_unit": 24
    },
    {
        "title": "Weekly (1wk) - Langfristig (4-Year-Cycle)",
        "interval": "1wk",
        "is_short_term": False,
        "hours_per_unit": 24 * 7
    }
]

# Layout: 3 Zeilen (Preis, FFT, Projektion), 3 Spalten
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
plt.subplots_adjust(hspace=0.6, wspace=0.2)

for i, scen in enumerate(scenarios):
    # --- 1. DATEN LADEN ---
    if scen["is_short_term"]:
        start_date = datetime.now() - timedelta(days=729)
        s_date_str = start_date.strftime('%Y-%m-%d')
    else:
        s_date_str = "2015-01-01"

    print(f"[{scen['interval']}] Lade Daten ab {s_date_str}...")

    try:
        data = yf.download("BTC-USD", start=s_date_str, interval=scen["interval"], progress=False)
        if data.empty or len(data) < 10: continue

        data = data.dropna()
        price = data['Close'].values.astype(np.float32).flatten()
        time_axis = np.arange(len(price), dtype=int)  # 0 bis N

    except Exception as e:
        print(f"Fehler: {e}")
        continue

    # --- 2. TREND & FFT BASIS ---
    coeffs = np.polyfit(time_axis, price, 1)
    trend = np.polyval(coeffs, time_axis)
    price_detrended = price - trend

    N = len(price_detrended)
    fft_vals = np.fft.rfft(price_detrended)
    fft_freqs = np.fft.rfftfreq(N, d=1.0)

    # Amplitude & Perioden
    amplitude = np.abs(fft_vals) * 2 / N
    with np.errstate(divide='ignore'):
        periods_units = 1 / fft_freqs
        factor_to_days = scen["hours_per_unit"] / 24.0
        periods_days = periods_units * factor_to_days

    # Plot Arrays
    plot_periods = periods_days[1:]
    plot_amplitude = amplitude[1:]

    # --- 3. DOMINANTEN ZYKLUS FINDEN ---
    if scen["interval"] == "1h":
        mask = (plot_periods > 0.5) & (plot_periods < 100)
    else:
        mask = (plot_periods > 7) & (plot_periods < 2000)

    cycle_found = False
    if np.any(mask):
        cycle_found = True
        local_max_idx = np.argmax(plot_amplitude[mask])
        masked_indices = np.where(mask)[0]
        real_idx = masked_indices[local_max_idx] + 1
        top_period_days = periods_days[real_idx - 1]

        # REKONSTRUKTION
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[real_idx] = fft_vals[real_idx]
        cycle_wave = np.fft.irfft(fft_filtered, n=N)
        cycle_on_chart = trend + cycle_wave

        # PROJEKTION
        A = amplitude[real_idx - 1]
        freq = real_idx / N
        phase = np.angle(fft_vals[real_idx])
        current_t = N - 1
        cycle_len_units = 1 / freq
        t_zoom_relative = np.linspace(-1.5 * cycle_len_units, 1.5 * cycle_len_units, 500)
        t_zoom_absolute = current_t + t_zoom_relative
        zoom_wave = A * np.cos(2 * np.pi * freq * t_zoom_absolute + phase)
        value_today = A * np.cos(2 * np.pi * freq * current_t + phase)

    # --- PLOTTING ---
    ax_price = axs[0, i]
    ax_price.plot(time_axis, price, color='black', alpha=0.2, label='Preis')
    if cycle_found:
        ax_price.plot(time_axis, cycle_on_chart, color='cyan', linewidth=2, label=f'Zyklus ~{top_period_days:.0f}d')
        ax_price.scatter([time_axis[-1]], [cycle_on_chart[-1]], color='blue', s=80, zorder=5, edgecolors='white')
    ax_price.set_title(f"{scen['title']}")
    ax_price.grid(True, alpha=0.3)
    if i == 0: ax_price.legend(loc='upper left', fontsize='small')

    ax_fft = axs[1, i]
    ax_fft.plot(plot_periods[mask], plot_amplitude[mask], color='purple')
    ax_fft.set_title(f"Frequenz-Spektrum")
    ax_fft.set_xlabel("Periodendauer (Tage)")
    ax_fft.grid(True, alpha=0.3)

    ax_proj = axs[2, i]
    if cycle_found:
        t_zoom_days = t_zoom_relative * factor_to_days
        ax_proj.plot(t_zoom_days, zoom_wave, color='cyan', linewidth=2)
        ax_proj.scatter([0], [value_today], color='blue', s=150, zorder=10, edgecolors='white', label='HEUTE')
        ax_proj.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax_proj.set_title(f"Dominanter Zyklus ({top_period_days:.1f} Tage)")
        ax_proj.set_xlabel("Tage relativ zu Heute")
        ax_proj.set_ylabel("Amplitude")
        ax_proj.axvspan(t_zoom_days[0], 0, color='grey', alpha=0.1)
        ax_proj.axvspan(0, t_zoom_days[-1], color='green', alpha=0.05)
        ax_proj.grid(True, alpha=0.3)
    else:
        ax_proj.text(0.5, 0.5, "Kein klarer Zyklus", ha='center')

if figure_dir:
    plt.savefig(os.path.join(figure_dir, "btc_fft_projection.png"))

print("FFT Analyse fertig. Fenster schließen für Polynom-Analyse.")
plt.show()

###################################################################################
########################### TEIL 2: POLYNOM ANALYSE (DYNAMISCH + PROJEKTION) ######
###################################################################################

print("\n-------------------------------------------------------")
print("Starte Polynom-Fitting (Least Squares) mit Zukunfts-Projektion...")

# [CONFIG]
test_degrees = range(2, 10)  # 2 bis 9
future_days = 600  # Wie weit in die Zukunft zeichnen?

start_date_poly = "2015-01-01"
data_poly = yf.download("BTC-USD", start=start_date_poly, interval="1d", progress=False)

if not data_poly.empty:
    data_poly = data_poly.dropna()
    price_poly = data_poly['Close'].values.astype(np.float32).flatten()

    # 1. Normale Zeitachse (Vergangenheit bis Heute)
    time_poly = np.arange(len(price_poly), dtype=int)

    # 2. Erweiterte Zeitachse (Vergangenheit + 600 Tage Zukunft)
    total_len = len(price_poly) + future_days
    time_poly_extended = np.arange(total_len, dtype=int)

    # Layout berechnen
    num_plots = len(test_degrees)
    cols = 2
    rows = math.ceil(num_plots / cols)
    fig_height = rows * 5

    fig_poly, axs_poly = plt.subplots(rows, cols, figsize=(15, fig_height))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    if num_plots > 1:
        axs_poly = axs_poly.flatten()
    else:
        axs_poly = [axs_poly]

    print(f"Erstelle {num_plots} Plots (Zukunftsprojektion: {future_days} Tage)...")

    for idx, deg in enumerate(test_degrees):
        ax = axs_poly[idx]

        # A. Fitting NUR auf den echten Daten (Vergangenheit)
        coeffs = np.polyfit(time_poly, price_poly, deg)

        # B. Kurve berechnen für die GESAMTE Zeit (inkl. Zukunft)
        poly_curve_extended = np.polyval(coeffs, time_poly_extended)

        # C. Fehler (RMSE) nur auf den echten Daten berechnen
        # Dafür schneiden wir die extended Kurve ab
        residuals = price_poly - poly_curve_extended[:len(price_poly)]
        rmse = np.sqrt(np.mean(residuals ** 2))

        # D. Plotten
        # Echte Daten (Schwarz)
        ax.plot(time_poly, price_poly, color='black', alpha=0.3, label='BTC Preis (Historie)')

        # Polynom Kurve (Rot) - Geht bis in die Zukunft
        ax.plot(time_poly_extended, poly_curve_extended, color='red', linewidth=2, label=f'Polynom {deg}. Grad')

        # Visuelle Trennung "Heute"
        ax.axvline(time_poly[-1], color='blue', linestyle='--', alpha=0.5, label='Heute')

        # Zukunftsbereich farbig hinterlegen
        ax.axvspan(time_poly[-1], time_poly_extended[-1], color='green', alpha=0.05, label='Projektion (600d)')

        # Beschriftung
        ax.set_title(f"Polynom {deg}. Grad | RMSE: {rmse:.2f} USD")
        ax.set_ylabel("USD")
        ax.set_xlabel("Tage seit 2015 (inkl. Zukunft)")
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)

        # Y-Achsen Limitierung: Wenn Kurven extrem explodieren, Plot lesbar halten
        # Wir setzen das Limit auf min/max Preis + 50% Puffer, damit wir Ausreißer sehen aber der Chart nicht unlesbar wird
        y_min = min(price_poly) * 0.5
        y_max = max(price_poly) * 3
        # Optional: Automatische Skalierung lassen, falls man die Explosion sehen will
        # ax.set_ylim(y_min, y_max)

        print(f"-> Polynom {deg}. Grad fertig.")

    # Leere Plots verstecken
    if len(axs_poly) > num_plots:
        for i in range(num_plots, len(axs_poly)):
            axs_poly[i].axis('off')

    if figure_dir:
        plt.savefig(os.path.join(figure_dir, "btc_poly_fitting_projection.png"))

    plt.suptitle(f"BTC Trend & Projektion (+{future_days} Tage)", fontsize=16, y=0.98)
    plt.show()

else:
    print("Fehler: Keine Daten für Polynom-Analyse.")