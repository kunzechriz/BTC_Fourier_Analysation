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

# Wir definieren die 3 Szenarien in einer Liste, um den Code sauber zu halten
scenarios = [
    {
        "title": "Hourly (1h) - Max 2 Jahre",
        "interval": "1h",
        "is_short_term": True,  # Flag für 730 Tage Limit
        "hours_per_unit": 1  # Umrechnung in Tage: 1h = 1/24 Tag
    },
    {
        "title": "Daily (1d) - Seit 2015",
        "interval": "1d",
        "is_short_term": False,
        "hours_per_unit": 24  # 1 Tag = 24h
    },
    {
        "title": "Weekly (1wk) - Seit 2015",
        "interval": "1wk",
        "is_short_term": False,
        "hours_per_unit": 24 * 7  # 1 Woche = 168h
    }
]

# Layout: 2 Zeilen (Oben Preis, Unten FFT), 3 Spalten (1h, 1d, 1wk)
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

###################################################################################
########################### SCHLEIFE DURCH ALLE SZENARIEN #########################
###################################################################################

for i, scen in enumerate(scenarios):

    # --- 1. DATUM BESTIMMEN ---
    if scen["is_short_term"]:
        # 1h Daten gehen maximal 730 Tage zurück
        start_date = datetime.now() - timedelta(days=729)
        s_date_str = start_date.strftime('%Y-%m-%d')
    else:
        # Langzeitdaten ab 2015
        s_date_str = "2015-01-01"

    print(f"[{scen['interval']}] Lade Daten ab {s_date_str}...")

    # --- 2. DATEN LADEN ---
    try:
        data = yf.download("BTC-USD", start=s_date_str, interval=scen["interval"], progress=False)

        # Leere Daten abfangen
        if data.empty or len(data) < 10:
            print(f"WARNUNG: Keine Daten für {scen['interval']}")
            continue

        # Bereinigen (NaN entfernen)
        data = data.dropna()
        price = data['Close'].values.astype(np.float32).flatten()

        # Zeitachse (einfach durchnummeriert 0..N)
        time_axis = np.arange(len(price), dtype=int)

    except Exception as e:
        print(f"Fehler bei {scen['interval']}: {e}")
        continue

    # --- 3. PLOT ZEILE 1: PREIS & TREND ---
    # Linearer Trend
    coeffs = np.polyfit(time_axis, price, 1)
    trend = np.polyval(coeffs, time_axis)
    price_detrended = price - trend

    ax_price = axs[0, i]
    ax_price.plot(time_axis, price, label='Price')
    ax_price.plot(time_axis, trend, label='Trend', linestyle='--', color='red', alpha=0.7)
    ax_price.set_title(f"{scen['title']}\n({len(price)} Punkte)")
    ax_price.set_ylabel("USD")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend()

    # --- 4. FFT BERECHNUNG ---
    N = len(price_detrended)
    fft_vals = np.fft.rfft(price_detrended)
    fft_freqs = np.fft.rfftfreq(N, d=1.0)  # d=1 Unit (h, d oder wk)

    # Amplitude
    amplitude = np.abs(fft_vals) * 2 / N

    # Umrechnung der Frequenz in "Tage" für alle Plots (damit vergleichbar)
    # Formel: Period_in_Tagen = (1 / Freq) * (Stunden_pro_Einheit / 24)
    with np.errstate(divide='ignore'):
        periods_units = 1 / fft_freqs
        # Umrechnung Faktor:
        # 1h -> mal 1/24
        # 1d -> mal 1
        # 1wk -> mal 7
        factor_to_days = scen["hours_per_unit"] / 24.0
        periods_days = periods_units * factor_to_days

    # Index 0 (Unendlich/DC) entfernen
    plot_periods = periods_days[1:]
    plot_amplitude = amplitude[1:]

    # --- 5. PLOT ZEILE 2: FFT SPEKTRUM ---
    ax_fft = axs[1, i]

    # Filter für sinnvolle Anzeige (Mask):
    # Wir filtern extrem kurze Rauscheffekte (< 2 Tage bei Daily/Weekly)
    # Und extrem lange Perioden (> Gesamtlänge der Daten)
    if scen["interval"] == "1h":
        mask = (plot_periods > 0.5) & (plot_periods < 100)  # Fokus auf 0.5 bis 100 Tage
    else:
        mask = (plot_periods > 7) & (plot_periods < 2000)  # Fokus auf Woche bis 5 Jahre

    ax_fft.plot(plot_periods[mask], plot_amplitude[mask], color='purple')
    ax_fft.set_title(f"FFT Spektrum ({scen['interval']})")
    ax_fft.set_xlabel("Periodendauer (Tage)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.grid(True, alpha=0.3, which="both")

    # Peak Detection (Stärkster Zyklus)
    if np.any(mask):
        max_idx = np.argmax(plot_amplitude[mask])
        top_period = plot_periods[mask][max_idx]
        top_amp = plot_amplitude[mask][max_idx]

        ax_fft.axvline(top_period, color='red', linestyle='--', alpha=0.8)
        ax_fft.text(top_period, top_amp, f' {top_period:.1f}d',
                    color='red', fontweight='bold', verticalalignment='bottom')

###################################################################################
########################### SAVE & SHOW ###########################################
###################################################################################

if figure_dir:
    plt.savefig(os.path.join(figure_dir, "btc_multi_timeframe_fft.png"))

print("\nPlot erstellt.")
plt.show()
plt.close()