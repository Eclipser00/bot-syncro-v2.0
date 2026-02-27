import winsound
import time

def music():
    # Notas
    C4 = 262
    D4 = 294
    E4 = 330
    F4 = 349
    G4 = 392
    A4 = 440
    B4 = 494
    C5 = 523
    D5 = 587
    E5 = 659

    # Duraciones
    quarter = 400
    half = 800
    eighth = 200

    # Melodía inventada, estilo "anime opening"
    melody = [
        (E5, quarter), (D5, eighth), (C5, eighth), (A4, quarter),
        (C5, quarter), (E5, half),
        (D5, quarter), (C5, eighth), (B4, eighth), (G4, quarter),
        (A4, half),
        (C5, quarter), (D5, quarter), (E5, quarter), (C5, half),
    ]

    # Reproducir
    for freq, dur in melody:
        winsound.Beep(freq, dur)
        time.sleep(0.05)