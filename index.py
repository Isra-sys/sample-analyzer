import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Mapeo de notas en inglés a notación latina
note_mapping = {
    "C": "Do", "C♯": "Do♯", "D♭": "Re♭", "D": "Re", "D♯": "Re♯", "E♭": "Mi♭",
    "E": "Mi", "F": "Fa", "F♯": "Fa♯", "G♭": "Sol♭", "G": "Sol", "G♯": "Sol♯", "A♭": "La♭",
    "A": "La", "A♯": "La♯", "B♭": "Si♭", "B": "Si"
}

def note_to_latin(note):
    """Convierte una nota en inglés a notación latina."""
    if note:
        if len(note) > 1 and note[1] in ['♯', '♭']:  # Nota con alteración
            base_note = note[:2]  # Extraer la nota con alteración
            octave = note[2:]  # Extraer la octava
        else:  # Nota sin alteración
            base_note = note[0]  # Nota base
            octave = note[1:]  # Octava
        return f"{note_mapping.get(base_note, base_note)}{octave}"
    return note

# Cargar un archivo de audio
audio_file = 'gregory.wav'  # Cambia la ruta al archivo
y, sr = librosa.load(audio_file)

# Extraer las frecuencias y la armonía
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Extraer la frecuencia fundamental
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
print(f'BPM estimado: {tempo}')

# Obtenemos la frecuencia fundamental
f0, voiced_flag, voiced_probs = librosa.pyin(y_harmonic, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('B8'))

# Interpolar valores NaN para una gráfica más limpia
times = librosa.times_like(f0)
f0_interp = np.interp(times, times[~np.isnan(f0)], f0[~np.isnan(f0)])

# Convertimos la frecuencia a notas musicales en notación latina
notes = [note_to_latin(librosa.hz_to_note(f)) for f in f0_interp]

# Mostrar el gráfico de la frecuencia fundamental
plt.figure(figsize=(6, 6))
plt.plot(times, f0_interp, label='Frecuencia Fundamental (f0)', color='b')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Frecuencia Fundamental Estimada')

# Agregar las notas en el gráfico con mayor tamaño y desplazadas hacia abajo
for i in range(0, len(times), max(1, len(times) // 35)):  # Mostrar cada 25 puntos aprox.
    plt.text(times[i], f0_interp[i] - 20, notes[i], fontsize=12, ha='right', color='red')

plt.legend()
plt.grid()
plt.show()
