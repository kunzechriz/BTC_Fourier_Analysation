### imports
import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

figure_dir = ""

###################################################################################
########################### KONFIGURATION #########################################
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
# Mehr Platz zwischen den Zeilen für die Beschriftungen
plt.subplots_adjust(hspace=0.6, wspace=0.2)

###################################################################################
########################### SCHLEIFE DURCH ALLE SZENARIEN #########################
###################################################################################

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

    # Plot Arrays (Index 0 weg)
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
        # Index finden
        local_max_idx = np.argmax(plot_amplitude[mask])
        masked_indices = np.where(mask)[0]
        real_idx = masked_indices[local_max_idx] + 1

        top_period_days = periods_days[real_idx - 1]

        # --- REKONSTRUKTION FÜR PLOT 1 (Vergangenheit) ---
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[real_idx] = fft_vals[real_idx]
        cycle_wave = np.fft.irfft(fft_filtered, n=N)
        cycle_on_chart = trend + cycle_wave  # Overlay für Preis-Chart

        # --- BERECHNUNG FÜR PLOT 3 (Projektion / Zoom) ---
        # Wir berechnen die reine Sinuswelle mathematisch, um sie in die Zukunft zu verlängern
        # Formel: y = A * cos(2*pi*f*t + phase)
        # f = real_idx / N (Frequenz in 'Cycles per Unit')
        A = amplitude[real_idx - 1]
        freq = real_idx / N
        phase = np.angle(fft_vals[real_idx])

        # Wir erstellen eine Zeitachse um "Heute" herum (Minus 1.5 Perioden bis Plus 1.5 Perioden)
        # "Heute" ist bei t = N-1
        current_t = N - 1

        # Länge der Projektion in Zeitschritten
        # Wir wollen ca. 1.5 Zyklen in die Zukunft und Vergangenheit sehen
        cycle_len_units = 1 / freq

        # Neue t-Achse relativ zu Heute (0 = Heute)
        t_zoom_relative = np.linspace(-1.5 * cycle_len_units, 1.5 * cycle_len_units, 500)

        # Umrechnen in absolute Zeit-Indizes für die Formel
        t_zoom_absolute = current_t + t_zoom_relative

        # Die Welle berechnen
        zoom_wave = A * np.cos(2 * np.pi * freq * t_zoom_absolute + phase)

        # Der Wert HEUTE (bei relative time 0)
        value_today = A * np.cos(2 * np.pi * freq * current_t + phase)

    # =========================================================================
    # PLOTTING
    # =========================================================================

    # --- ZEILE 1: PREIS + OVERLAY ---
    ax_price = axs[0, i]
    ax_price.plot(time_axis, price, color='black', alpha=0.2, label='Preis')
    # ax_price.plot(time_axis, trend, color='red', linestyle='--', alpha=0.3) # Trend ausblenden für Klarheit

    if cycle_found:
        ax_price.plot(time_axis, cycle_on_chart, color='cyan', linewidth=2, label=f'Zyklus ~{top_period_days:.0f}d')
        # Blauer Punkt am Ende
        ax_price.scatter([time_axis[-1]], [cycle_on_chart[-1]], color='blue', s=80, zorder=5, edgecolors='white')

    ax_price.set_title(f"{scen['title']}")
    ax_price.grid(True, alpha=0.3)
    if i == 0: ax_price.legend(loc='upper left', fontsize='small')

    # --- ZEILE 2: FFT SPEKTRUM (Ohne vertikale Linie) ---
    ax_fft = axs[1, i]
    ax_fft.plot(plot_periods[mask], plot_amplitude[mask], color='purple')
    ax_fft.set_title(f"Frequenz-Spektrum")
    ax_fft.set_xlabel("Periodendauer (Tage)")
    # HIER WURDE DIE VERTIKALE LINIE ENTFERNT, WIE GEWÜNSCHT
    ax_fft.grid(True, alpha=0.3)

    # --- ZEILE 3: ZYKLUS PROJEKTION (Mitte = Heute) ---
    ax_proj = axs[2, i]

    if cycle_found:
        # X-Achse in Tagen umrechnen (relative Zeit)
        t_zoom_days = t_zoom_relative * factor_to_days

        # Die Welle plotten
        ax_proj.plot(t_zoom_days, zoom_wave, color='cyan', linewidth=2)

        # Den "Heute" Punkt setzen (bei x=0)
        ax_proj.scatter([0], [value_today], color='blue', s=150, zorder=10, edgecolors='white', label='HEUTE')

        # Vertikale Linie für "Heute"
        ax_proj.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        # Beschriftung
        ax_proj.set_title(f"Dominanter Zyklus ({top_period_days:.1f} Tage)")
        ax_proj.set_xlabel("Tage relativ zu Heute")
        ax_proj.set_ylabel("Amplitude")

        # Hintergrund markieren (Vergangenheit vs Zukunft)
        ax_proj.axvspan(t_zoom_days[0], 0, color='grey', alpha=0.1, label='Vergangenheit')
        ax_proj.axvspan(0, t_zoom_days[-1], color='green', alpha=0.05, label='Zukunft')

        # Grid
        ax_proj.grid(True, alpha=0.3)

        if i == 0:  # Legende nur einmal
            ax_proj.legend(loc='lower left', fontsize='small')
    else:
        ax_proj.text(0.5, 0.5, "Kein klarer Zyklus", ha='center')

###################################################################################
########################### SAVE & SHOW ###########################################
###################################################################################

if figure_dir:
    plt.savefig(os.path.join(figure_dir, "btc_fft_projection.png"))

print("\nAnalyse abgeschlossen. Plot wird angezeigt.")
plt.show()
plt.close()