"""
Pedalera de guitarra en tiempo real.

Cadena: Noise Gate -> Drive -> Gabinete -> Chorus -> Delay -> Reverb -> Auriculares -> Master

Pensado para una Behringer UMC204HD con su driver ASIO, pero funciona con
cualquier placa de audio que aparezca en Windows.
"""
import os

os.environ.setdefault("SD_ENABLE_ASIO", "1")  # tiene que ir antes de importar sounddevice

import json
import math
import re
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd
from scipy import signal

AJUSTES = Path(__file__).with_name("ajustes.json")

# Por defecto Python le deja el control a la interfaz hasta 5 ms seguidos, lo mismo que dura
# un bloque de audio: el hilo de audio llegaba tarde y el sonido salía entrecortado.
sys.setswitchinterval(0.0005)

DEFAULTS = {
    "gate.on": True, "gate.umbral": -55.0, "gate.release": 120.0,
    "drive.on": True, "drive.modo": "Overdrive", "drive.gain": 18.0,
    "drive.tono": 55.0, "drive.volumen": 0.0,
    "gab.on": True, "gab.brillo": 50.0,
    "chorus.on": False, "chorus.velocidad": 0.8, "chorus.profundidad": 50.0, "chorus.mezcla": 50.0,
    "delay.on": False, "delay.tiempo": 380.0, "delay.repeticiones": 35.0, "delay.mezcla": 30.0,
    "rev.on": True, "rev.tamano": 50.0, "rev.brillo": 50.0, "rev.mezcla": 20.0,
    "salida.modo": "Mono", "salida.amplitud": 50.0,
    "master.entrada": 0.0, "master.volumen": -6.0,
}

# Cada sonido parte de DEFAULTS (menos Auriculares y Master, que no se tocan)
PRESETS = {
    "Limpio": {"drive.on": False, "gab.brillo": 60.0, "rev.mezcla": 22.0, "rev.tamano": 40.0},
    "Limpio + Chorus": {"drive.on": False, "gab.brillo": 65.0, "chorus.on": True, "rev.mezcla": 25.0},
    "Crunch": {"drive.gain": 16.0, "drive.tono": 55.0, "rev.mezcla": 15.0},
    "Rock": {"drive.modo": "Distorsión", "drive.gain": 22.0, "drive.tono": 55.0, "gab.brillo": 45.0,
             "rev.mezcla": 15.0},
    "Metal": {"gate.umbral": -50.0, "gate.release": 80.0, "drive.modo": "Distorsión", "drive.gain": 36.0,
              "drive.tono": 60.0, "drive.volumen": -2.0, "gab.brillo": 40.0, "rev.mezcla": 10.0},
    "Solo con eco": {"drive.gain": 28.0, "drive.tono": 65.0, "delay.on": True, "delay.tiempo": 420.0,
                     "delay.mezcla": 25.0, "rev.mezcla": 25.0},
    "Fuzz retro": {"drive.modo": "Fuzz", "drive.gain": 20.0, "drive.tono": 45.0, "drive.volumen": -3.0},
    "Ambiente": {"drive.on": False, "chorus.on": True, "chorus.velocidad": 0.5, "chorus.mezcla": 40.0,
                 "delay.on": True, "delay.tiempo": 500.0, "delay.repeticiones": 50.0, "delay.mezcla": 35.0,
                 "rev.tamano": 75.0, "rev.brillo": 60.0, "rev.mezcla": 45.0},
}


def db(v):
    return 10.0 ** (v / 20.0)


# ----------------------------------------------------------------- efectos

class NoiseGate:
    def __init__(self, sr):
        self.sr = sr
        self.reset()

    def reset(self):
        self.gain = 1.0
        self.env = 0.0
        self.abierto = True

    def process(self, x, p):
        n = len(x)
        rms = math.sqrt(float(np.mean(x * x))) + 1e-12
        # envolvente con caída de ~50 ms para que no tartamudee en notas graves
        self.env = max(rms, self.env * math.exp(-n / (0.05 * self.sr)))
        nivel = 20 * math.log10(self.env + 1e-12)
        umbral = p["gate.umbral"]
        # histéresis: abre en el umbral y cierra 6 dB por debajo
        if self.abierto and nivel < umbral - 6:
            self.abierto = False
        elif not self.abierto and nivel > umbral:
            self.abierto = True
        objetivo = 1.0 if self.abierto else 0.0
        tau = 0.001 if objetivo > self.gain else p["gate.release"] / 1000.0
        nuevo = objetivo + (self.gain - objetivo) * math.exp(-n / (tau * self.sr))
        rampa = np.linspace(self.gain, nuevo, n)
        self.gain = nuevo
        return x * rampa


class Drive:
    MODOS = ("Overdrive", "Distorsión", "Fuzz")

    def __init__(self, sr):
        self.sr = sr
        self.pre_hp = signal.butter(1, 100, "highpass", fs=sr, output="sos")
        self.post_hp = signal.butter(1, 30, "highpass", fs=sr, output="sos")
        self.tono = None
        self.reset()

    def reset(self):
        self.pre_zi = np.zeros((1, 2))
        self.post_zi = np.zeros((1, 2))
        self.lp_zi = np.zeros((1, 2))

    def _filtro_tono(self, tono):
        if tono != self.tono:
            fc = 700 * (12000 / 700) ** (tono / 100)  # 0..100 -> 700 Hz..12 kHz
            self.lp = signal.butter(2, min(fc, 0.45 * self.sr), "lowpass", fs=self.sr, output="sos")
            self.tono = tono
        return self.lp

    def process(self, x, p):
        x, self.pre_zi = signal.sosfilt(self.pre_hp, x, zi=self.pre_zi)
        y = x * db(p["drive.gain"])
        modo = p["drive.modo"]
        if modo == "Distorsión":
            # recorte más duro y asimétrico (el DC que genera lo saca el post_hp)
            y = np.where(y >= 0, np.tanh(3 * y), 0.8 * np.tanh(2 * y))
        elif modo == "Fuzz":
            y = np.sign(y) * (1 - np.exp(-np.abs(10 * y)))
        else:
            y = np.tanh(y)
        y, self.post_zi = signal.sosfilt(self.post_hp, y, zi=self.post_zi)
        y, self.lp_zi = signal.sosfilt(self._filtro_tono(p["drive.tono"]), y, zi=self.lp_zi)
        return y * 0.3 * db(p["drive.volumen"])


class Gabinete:
    """Simula el parlante: sin esto la distorsión suena a abeja."""

    def __init__(self, sr):
        self.sr = sr
        self.brillo = None
        self.reset()

    def reset(self):
        self.zi = np.zeros((3, 2))

    def _filtros(self, brillo):
        if brillo != self.brillo:
            fc = 2500 * (7000 / 2500) ** (brillo / 100)
            self.sos = np.vstack([
                signal.butter(2, 90, "highpass", fs=self.sr, output="sos"),
                signal.butter(4, fc, "lowpass", fs=self.sr, output="sos"),
            ])
            self.brillo = brillo
        return self.sos

    def process(self, x, p):
        y, self.zi = signal.sosfilt(self._filtros(p["gab.brillo"]), x, zi=self.zi)
        return y


class Chorus:
    def __init__(self, sr):
        self.sr = sr
        self.reset()

    def reset(self):
        self.buf = np.zeros(int(0.05 * self.sr))
        self.w = 0
        self.fase = 0.0

    def process(self, x, p):
        n, N = len(x), len(self.buf)
        k = np.arange(n)
        self.buf[(self.w + k) % N] = x
        paso = 2 * math.pi * p["chorus.velocidad"] / self.sr
        lfo = np.sin(self.fase + paso * k)
        self.fase = (self.fase + paso * n) % (2 * math.pi)
        prof = p["chorus.profundidad"] / 100
        retardo = (0.012 + 0.008 * prof * lfo) * self.sr  # 4..20 ms
        lectura = self.w + k - retardo
        i0 = np.floor(lectura).astype(int)
        frac = lectura - i0
        a = self.buf[i0 % N]
        b = self.buf[(i0 + 1) % N]
        mojado = a + (b - a) * frac
        self.w = (self.w + n) % N
        mezcla = p["chorus.mezcla"] / 100
        return (x + mezcla * mojado) / (1 + 0.5 * mezcla)


class Delay:
    def __init__(self, sr, maximo_s=2.0):
        self.sr = sr
        self.N = int(maximo_s * sr) + 1
        d = math.exp(-2 * math.pi * 4000 / sr)  # las repeticiones se oscurecen un poco
        self.b, self.a = [1 - d], [1, -d]
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.N)
        self.w = 0
        self.zi = np.zeros(1)

    def process(self, x, p):
        D = int(min(max(p["delay.tiempo"] * self.sr / 1000, 1), self.N - 1))
        fb = p["delay.repeticiones"] / 100
        mezcla = p["delay.mezcla"] / 100
        out = np.empty_like(x)
        pos = 0
        # trozos de como mucho D muestras: así lo que se lee ya fue escrito
        while pos < len(x):
            m = min(D, len(x) - pos)
            k = np.arange(m)
            eco = self.buf[(self.w - D + k) % self.N]
            eco, self.zi = signal.lfilter(self.b, self.a, eco, zi=self.zi)
            seco = x[pos:pos + m]
            self.buf[(self.w + k) % self.N] = seco + fb * eco
            out[pos:pos + m] = seco + mezcla * eco
            self.w = (self.w + m) % self.N
            pos += m
        return out


class _Comb:
    def __init__(self, n):
        self.buf = np.zeros(n)
        self.i = 0
        self.zi = np.zeros(1)

    def process(self, x, fb, damp):
        n = len(self.buf)
        out = np.empty_like(x)
        pos = 0
        while pos < len(x):
            m = min(n - self.i, len(x) - pos)
            sal = self.buf[self.i:self.i + m].copy()
            filt, self.zi = signal.lfilter([1 - damp], [1, -damp], sal, zi=self.zi)
            self.buf[self.i:self.i + m] = x[pos:pos + m] + fb * filt
            out[pos:pos + m] = sal
            self.i = (self.i + m) % n
            pos += m
        return out


class _Allpass:
    def __init__(self, n):
        self.buf = np.zeros(n)
        self.i = 0

    def process(self, x):
        n = len(self.buf)
        out = np.empty_like(x)
        pos = 0
        while pos < len(x):
            m = min(n - self.i, len(x) - pos)
            sal = self.buf[self.i:self.i + m].copy()
            self.buf[self.i:self.i + m] = x[pos:pos + m] + 0.5 * sal
            out[pos:pos + m] = sal - x[pos:pos + m]
            self.i = (self.i + m) % n
            pos += m
        return out


class Reverb:
    """Reverb tipo Freeverb: 8 filtros comb en paralelo + 4 allpass en serie."""

    COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
    ALLPASS = (556, 441, 341, 225)

    def __init__(self, sr):
        self.k = sr / 44100
        self.reset()

    def reset(self):
        self.combs = [_Comb(int(d * self.k)) for d in self.COMBS]
        self.allpass = [_Allpass(int(d * self.k)) for d in self.ALLPASS]

    def process(self, x, p):
        fb = 0.7 + 0.28 * p["rev.tamano"] / 100
        damp = 0.4 * (1 - p["rev.brillo"] / 100)
        entrada = x * 0.015
        mojado = sum(c.process(entrada, fb, damp) for c in self.combs)
        for a in self.allpass:
            mojado = a.process(mojado)
        return x + 3 * (p["rev.mezcla"] / 100) * mojado


class Salida:
    """Mono: la misma señal en los dos oídos. Estéreo: ensancha sin perder nada al sumar a mono."""

    MODOS = ("Mono", "Estéreo")

    def __init__(self, sr):
        self.N = int(0.03 * sr)
        self.D = int(0.015 * sr)
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.N)
        self.w = 0

    def process(self, x, p):
        k = np.arange(len(x))
        self.buf[(self.w + k) % self.N] = x
        retrasado = self.buf[(self.w + k - self.D) % self.N]
        self.w = (self.w + len(x)) % self.N
        if p["salida.modo"] == "Mono":
            return np.column_stack((x, x))
        lado = 0.7 * (p["salida.amplitud"] / 100) * retrasado
        return np.column_stack((x + lado, x - lado))


class Pedalera:
    def __init__(self, sr):
        self.efectos = [
            ("gate", NoiseGate(sr)),
            ("drive", Drive(sr)),
            ("gab", Gabinete(sr)),
            ("chorus", Chorus(sr)),
            ("delay", Delay(sr)),
            ("rev", Reverb(sr)),
        ]
        self.activos = {k: False for k, _ in self.efectos}
        self.salida = Salida(sr)

    def process(self, x, p):
        """Recibe la guitarra (mono) y devuelve un bloque de 2 columnas: izquierda y derecha."""
        x = x * db(p["master.entrada"])
        for k, ef in self.efectos:
            on = p[k + ".on"]
            if on and not self.activos[k]:
                ef.reset()  # al prender un pedal no queremos restos viejos del buffer
            self.activos[k] = on
            if on:
                x = ef.process(x, p)
        estereo = self.salida.process(x, p)
        return np.clip(estereo * db(p["master.volumen"]), -1.0, 1.0)


# ---------------------------------------------------------------- interfaz

# (clave, nombre, descripción, color, controles); un control con tupla de opciones son botones de elección
PEDALES = [
    ("gate", "NOISE GATE", "Corta el zumbido\ncuando no tocás", "#5f6670", [
        ("umbral", "Sensibilidad (dB)", -90, -20, 1),
        ("release", "Cierre (ms)", 20, 500, 10),
    ]),
    ("drive", "DRIVE", "Ensucia el sonido:\nde crunch a metal", "#d9822b", [
        ("modo", "Modo", Drive.MODOS),
        ("gain", "Ganancia (dB)", 0, 45, 0.5),
        ("tono", "Tono", 0, 100, 1),
        ("volumen", "Volumen (dB)", -20, 12, 0.5),
    ]),
    ("gab", "GABINETE", "Simula el parlante del ampli.\nDejalo prendido con Drive", "#8a5a3c", [
        ("brillo", "Brillo", 0, 100, 1),
    ]),
    ("chorus", "CHORUS", "Duplica y ondula\nel sonido", "#2f7fc1", [
        ("velocidad", "Velocidad (Hz)", 0.1, 5, 0.1),
        ("profundidad", "Profundidad", 0, 100, 1),
        ("mezcla", "Cantidad", 0, 100, 1),
    ]),
    ("delay", "DELAY", "Eco que repite\nlo que tocás", "#2e9e6a", [
        ("tiempo", "Tiempo entre ecos (ms)", 50, 1500, 10),
        ("repeticiones", "Repeticiones", 0, 90, 1),
        ("mezcla", "Cantidad", 0, 100, 1),
    ]),
    ("rev", "REVERB", "Ambiente de sala\no de iglesia", "#7b52b9", [
        ("tamano", "Tamaño de la sala", 0, 100, 1),
        ("brillo", "Brillo", 0, 100, 1),
        ("mezcla", "Cantidad", 0, 100, 1),
    ]),
    ("salida", "AURICULARES", "Mono = la guitarra suena\nigual en los dos oídos", "#1f6f78", [
        ("modo", "Modo", Salida.MODOS),
        ("amplitud", "Amplitud (solo Estéreo)", 0, 100, 1),
    ]),
    ("master", "MASTER", "Volumen general\ndel programa", "#3a3d42", [
        ("entrada", "Entrada (dB)", -12, 24, 0.5),
        ("volumen", "Volumen (dB)", -40, 6, 0.5),
    ]),
]

# retraso pedido al driver (s) y tamaño de bloque; con la UMC por ASIO dan ≈10 / 15 / 27 ms
RETRASOS = {"Mínimo": (0.002, 128), "Bajo": (0.005, 256), "Seguro": (0.010, 512)}

# Programas que suelen quedarse con la placa de audio (comienzo del nombre del .exe, en minúsculas)
PROGRAMAS_AUDIO = {
    "fl64": "FL Studio", "fl.exe": "FL Studio", "ableton": "Ableton Live", "reaper": "REAPER",
    "cubase": "Cubase", "studio one": "Studio One", "bitwig": "Bitwig Studio", "audacity": "Audacity",
    "guitar rig": "Guitar Rig", "amplitube": "AmpliTube", "bias fx": "BIAS FX", "voicemeeter": "Voicemeeter",
    "obs64": "OBS Studio", "discord": "Discord", "zoom": "Zoom", "ms-teams": "Microsoft Teams",
    "teams": "Microsoft Teams", "skype": "Skype", "spotify": "Spotify",
}

# Palabras que aparecen en nombres de dispositivos de cualquier marca y no sirven para reconocer una placa
GENERICAS = {"ASIO", "DRIVER", "AUDIO", "USB", "MICRÓFONO", "MICROFONO", "MICROPHONE", "ALTAVOCES",
             "SPEAKERS", "AURICULARES", "HEADPHONES", "LINE", "DEVICE", "HIGH", "DEFINITION", "INPUT", "OUTPUT", "OUT"}


PREFERIDAS_IN = ("UMC ASIO", "IN 1-2 (BEHRINGER", "UMC")
PREFERIDAS_OUT = ("UMC ASIO", "OUT 1-2 (BEHRINGER", "UMC")


def elegir(lista, preferido, palabras):
    """El preferido si está; si no, el primero que contenga alguna de las palabras, en orden."""
    if preferido in lista:
        return preferido
    for palabra in palabras:
        for nombre in lista:
            if palabra in nombre.upper():
                return nombre
    return lista[0] if lista else ""


def marcas(nombre):
    """Palabras que identifican a una placa, p. ej. 'UMC ASIO Driver' -> {'UMC'}."""
    return {w for w in re.findall(r"\w{3,}", nombre.upper()) if w not in GENERICAS and not w.isdigit()}


def programas_de_audio_abiertos():
    script = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
              "Get-CimInstance Win32_Process | Select-Object ProcessId, Name, CommandLine | ConvertTo-Json")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        procesos = json.loads(r.stdout.decode("utf-8", errors="replace"))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []
    encontrados = []
    for proc in procesos:
        nombre = (proc.get("Name") or "").lower()
        linea = (proc.get("CommandLine") or "").lower()
        if "pedalera.py" in linea and nombre.startswith("python") and proc.get("ProcessId") != os.getpid():
            encontrados.append("Otra ventana de la Pedalera")
        for inicio, programa in PROGRAMAS_AUDIO.items():
            if nombre.startswith(inicio):
                encontrados.append(programa)
    return sorted(set(encontrados))

FONDO = "#1b1c1f"
TEXTO = "#e8e8e8"
GRIS = "#9aa0a6"


class App:
    def __init__(self, root):
        self.root = root
        self.p = dict(DEFAULTS)
        self.vars = {}
        self.dispositivos = {}
        self.stream = None
        self.pedalera = None
        self.canal = 0
        self.canal_auto = True
        self.pico_in = self.pico_out = 0.0
        self.barra_in = self.barra_out = -60.0
        self.cortes = 0
        self.cortes_previos = 0
        self.ticks = 0
        self.aviso = ""
        self.error = None

        guardado = self._cargar()
        root.title("Pedalera de guitarra")
        root.configure(bg=FONDO)
        root.protocol("WM_DELETE_WINDOW", self._cerrar)
        self._armar_cabecera()
        self._armar_config(guardado)
        self._armar_medidores()
        self._armar_pedales()
        self._refrescar()

    # ---------- ajustes

    def _cargar(self):
        try:
            datos = json.loads(AJUSTES.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        for k, v in datos.get("params", {}).items():
            if k not in DEFAULTS:
                continue
            base = DEFAULTS[k]
            if type(v) is type(base) or (type(base) is float and type(v) is int):
                self.p[k] = v
        return datos.get("audio", {})

    def _guardar(self):
        audio = {
            "api": self.var_api.get(), "entrada": self.var_in.get(), "salida": self.var_out.get(),
            "canal": self.var_canal.get(), "retraso": self.var_retraso.get(), "sr": self.var_sr.get(),
        }
        try:
            AJUSTES.write_text(json.dumps({"params": self.p, "audio": audio}, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        except OSError:
            pass

    # ---------- cabecera, sonidos listos y ayuda

    def _armar_cabecera(self):
        cab = tk.Frame(self.root, bg=FONDO)
        cab.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(cab, text="🎸 Pedalera", font=("Segoe UI", 18, "bold"), bg=FONDO, fg=TEXTO).pack(side="left")
        self.boton = tk.Button(cab, text="▶  ENCENDER", command=self._alternar, bg="#2e9e6a", fg="white",
                               activebackground="#257f55", activeforeground="white", relief="flat",
                               padx=22, pady=6, font=("Segoe UI", 13, "bold"), cursor="hand2")
        self.boton.pack(side="right")
        tk.Button(cab, text="⚙  Configuración de audio", command=self._mostrar_config, bg="#2f3237", fg=TEXTO,
                  activebackground="#3a3e44", activeforeground=TEXTO, relief="flat", padx=10, pady=6,
                  cursor="hand2").pack(side="right", padx=10)

        fila = tk.Frame(self.root, bg=FONDO)
        fila.pack(fill="x", padx=16, pady=4)
        tk.Label(fila, text="Sonidos listos:", bg=FONDO, fg=GRIS).pack(side="left", padx=(0, 6))
        for nombre in PRESETS:
            tk.Button(fila, text=nombre, command=lambda n=nombre: self._aplicar_preset(n), bg="#2f3237",
                      fg=TEXTO, activebackground="#454a52", activeforeground=TEXTO, relief="flat",
                      padx=10, pady=3, cursor="hand2").pack(side="left", padx=3)

        self.ayuda = tk.Label(
            self.root, justify="left", anchor="w", bg="#24303a", fg="#cfe3f5", padx=12, pady=8,
            text="1)  Apretá ENCENDER y tocá: la barra IN se tiene que mover. Si se pone roja, bajá la perilla "
                 "GAIN de la placa.\n"
                 "2)  ¿Escuchás la guitarra en un solo oído o sin efectos? Girá la perilla MIX de la UMC204HD "
                 "del todo hacia PLAYBACK.\n"
                 "3)  Elegí un sonido listo arriba y después ajustá las perillas a gusto. Los pedales se prenden "
                 "y apagan con su botón ON/OFF.")
        self.ayuda.pack(fill="x", padx=16, pady=6)

    # ---------- configuración de audio (escondida)

    def _armar_config(self, guardado):
        self.config = tk.Frame(self.root, bg="#232529", padx=10, pady=8)
        self.apis = [h["name"] for h in sd.query_hostapis()]
        self.var_api = tk.StringVar()
        self.var_in = tk.StringVar()
        self.var_out = tk.StringVar()
        self.var_canal = tk.StringVar(value=guardado.get("canal", "Auto"))
        retraso = guardado.get("retraso")
        self.var_retraso = tk.StringVar(value=retraso if retraso in RETRASOS else "Bajo")
        self.var_sr = tk.StringVar(value=guardado.get("sr", "Auto"))

        def combo(fila, col, texto, var, ancho, valores=()):
            tk.Label(self.config, text=texto, bg="#232529", fg=TEXTO).grid(row=fila, column=col, sticky="e",
                                                                           padx=(10, 4), pady=3)
            c = ttk.Combobox(self.config, textvariable=var, width=ancho, values=valores, state="readonly")
            c.grid(row=fila, column=col + 1, sticky="w", pady=3)
            return c

        self.cb_api = combo(0, 0, "Driver", self.var_api, 22, self.apis)
        self.cb_in = combo(0, 2, "Entrada", self.var_in, 34)
        self.cb_out = combo(0, 4, "Salida", self.var_out, 34)
        self.cb_canal = combo(1, 0, "Canal de la guitarra", self.var_canal, 8)
        combo(1, 2, "Retraso", self.var_retraso, 8, tuple(RETRASOS))
        combo(1, 4, "Frecuencia (Hz)", self.var_sr, 8, ("Auto", "44100", "48000", "96000"))
        tk.Label(self.config, bg="#232529", fg=GRIS, justify="left", anchor="w",
                 text="Recomendado: Driver ASIO + UMC ASIO Driver. Canal en Auto detecta solo dónde está "
                      "enchufada la guitarra. Frecuencia en Auto usa la que ya tiene configurada la placa.\n"
                      "Retraso: si hay cortes, el programa lo sube solo. Estos cambios se aplican al apagar y "
                      "volver a encender.").grid(
            row=2, column=0, columnspan=6, sticky="w", padx=10, pady=(6, 0))

        self.cb_api.bind("<<ComboboxSelected>>", lambda e: self._listar_dispositivos())
        self.cb_in.bind("<<ComboboxSelected>>", lambda e: self._listar_canales())

        api = guardado.get("api")
        if api not in self.apis:
            api = next((a for a in self.apis if "ASIO" in a), None) or \
                  next((a for a in self.apis if "WASAPI" in a), self.apis[0])
        self.var_api.set(api)
        self._listar_dispositivos(guardado.get("entrada"), guardado.get("salida"))

    def _mostrar_config(self):
        if self.config.winfo_ismapped():
            self.config.pack_forget()
        else:
            self.config.pack(fill="x", padx=16, pady=(0, 6), after=self.ayuda)

    def _dispositivos_de(self, api, entrada=None, salida=None):
        """Para un driver: {nombre: índice}, entradas, salidas y la entrada/salida más probables."""
        indice = self.apis.index(api)
        disp, ins, outs = {}, [], []
        for i, d in enumerate(sd.query_devices()):
            if d["hostapi"] != indice:
                continue
            disp[d["name"]] = i
            if d["max_input_channels"] > 0:
                ins.append(d["name"])
            if d["max_output_channels"] > 0:
                outs.append(d["name"])

        return disp, ins, outs, elegir(ins, entrada, PREFERIDAS_IN), elegir(outs, salida, PREFERIDAS_OUT)

    def _listar_dispositivos(self, entrada=None, salida=None):
        self.dispositivos, ins, outs, ent, sal = self._dispositivos_de(self.var_api.get(), entrada, salida)
        self.cb_in["values"] = ins
        self.cb_out["values"] = outs
        self.var_in.set(ent)
        self.var_out.set(sal)
        self._listar_canales()

    def _listar_canales(self):
        nombre = self.var_in.get()
        n = sd.query_devices(self.dispositivos[nombre])["max_input_channels"] if nombre else 1
        valores = ["Auto"] + [str(i + 1) for i in range(n)]
        self.cb_canal["values"] = valores
        if self.var_canal.get() not in valores:
            self.var_canal.set("Auto")

    # ---------- medidores

    def _armar_medidores(self):
        marco = tk.Frame(self.root, bg=FONDO)
        marco.pack(fill="x", padx=20, pady=4)
        self.lienzos = []
        for texto in ("IN", "OUT"):
            fila = tk.Frame(marco, bg=FONDO)
            fila.pack(fill="x", pady=1)
            tk.Label(fila, text=texto, width=4, bg=FONDO, fg=TEXTO, anchor="w").pack(side="left")
            c = tk.Canvas(fila, height=10, bg="#2a2c30", highlightthickness=0)
            c.pack(side="left", fill="x", expand=True)
            self.lienzos.append(c)
        self.estado = tk.Label(marco, text="Apagado", bg=FONDO, fg=GRIS, anchor="w", justify="left")
        self.estado.pack(fill="x", pady=(4, 0))

    def _dibujar_barra(self, lienzo, nivel_db):
        ancho = lienzo.winfo_width()
        lienzo.delete("all")
        frac = min(max((nivel_db + 60) / 60, 0), 1)
        color = "#e04848" if nivel_db > -1 else "#e0b848" if nivel_db > -12 else "#4cc27a"
        lienzo.create_rectangle(0, 0, ancho * frac, 10, fill=color, width=0)

    # ---------- pedales

    def _armar_pedales(self):
        tablero = tk.Frame(self.root, bg=FONDO)
        tablero.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        self.botones_on = {}
        for n, (clave, nombre, descripcion, color, controles) in enumerate(PEDALES):
            pedal = tk.Frame(tablero, bg=color, padx=10, pady=8)
            pedal.grid(row=n // 4, column=n % 4, padx=6, pady=6, sticky="nsew")
            tk.Label(pedal, text=nombre, bg=color, fg="white", font=("Segoe UI", 12, "bold")).pack()
            tk.Label(pedal, text=descripcion, bg=color, fg="#f0f0f0", font=("Segoe UI", 8)).pack(pady=(0, 6))

            for control in controles:
                k = f"{clave}.{control[0]}"
                if isinstance(control[2], tuple):
                    var = tk.StringVar(value=self.p[k])
                    self.vars[k] = var
                    opciones = tk.Frame(pedal, bg=color)
                    opciones.pack(pady=(0, 6))
                    for opcion in control[2]:
                        tk.Radiobutton(opciones, text=opcion, value=opcion, variable=var, indicatoron=False,
                                       command=lambda k=k, var=var: self.p.__setitem__(k, var.get()),
                                       bg="#2a2a2a", fg="white", selectcolor="#111111",
                                       activebackground="#444444", activeforeground="white",
                                       relief="flat", padx=8, pady=3, cursor="hand2").pack(side="left", padx=1)
                    continue
                _, etiqueta, desde, hasta, paso = control
                var = tk.DoubleVar(value=self.p[k])
                self.vars[k] = var
                tk.Scale(pedal, label=etiqueta, from_=desde, to=hasta, resolution=paso, orient="horizontal",
                         variable=var, length=200, bg=color, fg="white", troughcolor="#1f1f1f",
                         highlightthickness=0, activebackground="white", sliderrelief="flat",
                         command=lambda v, k=k: self.p.__setitem__(k, float(v))).pack()

            if clave + ".on" in self.p:
                b = tk.Button(pedal, command=lambda c=clave: self._pisar(c), relief="flat", width=10,
                              font=("Segoe UI", 9, "bold"), cursor="hand2")
                b.pack(side="bottom", pady=(8, 0))
                self.botones_on[clave] = b
                self._pintar_boton(clave)

        for col in range(4):
            tablero.columnconfigure(col, weight=1)

    def _pisar(self, clave):
        self.p[clave + ".on"] = not self.p[clave + ".on"]
        self._pintar_boton(clave)

    def _pintar_boton(self, clave):
        on = self.p[clave + ".on"]
        fondo = "#ff4d4d" if on else "#2a2a2a"
        self.botones_on[clave].configure(text="● ON" if on else "○ OFF", bg=fondo, fg="white",
                                         activebackground=fondo, activeforeground="white")

    def _aplicar_preset(self, nombre):
        for k, v in DEFAULTS.items():
            if not k.startswith(("salida.", "master.")):
                self.p[k] = v
        self.p.update(PRESETS[nombre])
        for k, var in self.vars.items():
            var.set(self.p[k])
        for clave in self.botones_on:
            self._pintar_boton(clave)

    # ---------- audio

    def _callback(self, indata, outdata, frames, tiempo, status):
        if status.input_overflow or status.output_underflow:
            self.cortes += 1
        try:
            entrada = indata.astype(np.float64)
            if self.canal_auto and entrada.shape[1] > 1:
                # si el otro canal tiene claramente más señal, la guitarra está ahí
                picos = np.max(np.abs(entrada), axis=0)
                otro = 1 - self.canal
                if picos[otro] > 0.01 and picos[otro] > 4 * picos[self.canal]:
                    self.canal = otro
            x = entrada[:, self.canal]
            y = self.pedalera.process(x, self.p)
            outdata[:] = y[:, :outdata.shape[1]]
            self.pico_in = max(self.pico_in, float(np.max(np.abs(x))))
            self.pico_out = max(self.pico_out, float(np.max(np.abs(y))))
        except Exception as e:  # nunca dejar que el callback muera: silencio y avisar
            outdata.fill(0)
            self.error = repr(e)

    def _alternar(self):
        if self.stream:
            self._apagar()
        else:
            self.aviso = ""
            self._encender()

    def _intentos(self):
        """Lo elegido primero; si falla, otras frecuencias y la misma placa en los otros drivers."""
        elegida = self.var_sr.get()
        buscadas = marcas(self.var_in.get()) | marcas(self.var_out.get())
        apis = [self.var_api.get()] + [a for pref in ("ASIO", "WASAPI", "DirectSound", "MME")
                                       for a in self.apis if pref in a]
        vistos = set()
        for n, api in enumerate(apis):
            if n == 0:
                disp, ent, sal = self.dispositivos, self.var_in.get(), self.var_out.get()
            else:
                disp, ins, outs, _, _ = self._dispositivos_de(api)
                # en otro driver solo sirve la misma placa, no el micrófono de la laptop
                ins = [d for d in ins if marcas(d) & buscadas]
                outs = [d for d in outs if marcas(d) & buscadas]
                if not ins or not outs:
                    continue
                ent, sal = elegir(ins, None, PREFERIDAS_IN), elegir(outs, None, PREFERIDAS_OUT)
            if ent not in disp or sal not in disp:
                continue
            info_in, info_out = sd.query_devices(disp[ent]), sd.query_devices(disp[sal])
            frecuencias = [int(elegida)] if elegida != "Auto" and n == 0 else []
            frecuencias += [int(info_in["default_samplerate"]), int(info_out["default_samplerate"]), 48000, 44100]
            for sr in frecuencias:
                if (api, ent, sal, sr) not in vistos:
                    vistos.add((api, ent, sal, sr))
                    yield api, disp, ent, sal, sr

    def _abrir(self, api, entrada, salida, sr):
        info_in, info_out = sd.query_devices(entrada), sd.query_devices(salida)
        max_in = info_in["max_input_channels"]
        if self.canal_auto:
            self.canal = 0
            n_in = min(2, max_in)
        else:
            self.canal = min(int(self.var_canal.get()), max_in) - 1
            n_in = self.canal + 1
        n_out = min(2, info_out["max_output_channels"])
        latencia, bloque = RETRASOS[self.var_retraso.get()]
        extra = None
        if "ASIO" not in api:
            # sin ASIO, Windows no llega con retrasos tan chicos y el audio sale entrecortado
            latencia = max(latencia, info_in["default_low_input_latency"],
                           info_out["default_low_output_latency"], 0.02)
        if "WASAPI" in api:
            ajustes = sd.WasapiSettings(auto_convert=True)  # Windows convierte si la frecuencia no coincide
            extra = (ajustes, ajustes)
        self.pedalera = Pedalera(sr)
        stream = None
        try:
            stream = sd.Stream(device=(entrada, salida), samplerate=sr, blocksize=bloque,
                               channels=(n_in, n_out), dtype="float32", latency=latencia,
                               extra_settings=extra, callback=self._callback)
            stream.start()
        except Exception:
            if stream:
                stream.close()
            raise
        return stream

    def _encender(self):
        if self.var_in.get() not in self.dispositivos or self.var_out.get() not in self.dispositivos:
            messagebox.showerror("Pedalera", "Abrí ⚙ Configuración de audio y elegí una entrada y una salida.")
            return
        self.canal_auto = self.var_canal.get() == "Auto"
        self.cortes = 0
        self.cortes_previos = 0
        self.ticks = 0
        self.error = None
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        errores = []
        try:
            for api, disp, ent, sal, sr in self._intentos():
                try:
                    self.stream = self._abrir(api, disp[ent], disp[sal], sr)
                    break
                except Exception as e:
                    errores.append(f"{api} · {ent} · {sr} Hz: {e}")
        finally:
            self.root.configure(cursor="")
        if not self.stream:
            self._avisar_falla(errores)
            return
        if (api, ent, sal) != (self.var_api.get(), self.var_in.get(), self.var_out.get()):
            self.var_api.set(api)
            self._listar_dispositivos(ent, sal)
            self.aviso = f"Con la configuración elegida no se pudo abrir la placa; funcionó con el driver {api}."
        self.boton.configure(text="■  APAGAR", bg="#c94444", activebackground="#a33636")
        for c in (self.cb_api, self.cb_in, self.cb_out, self.cb_canal):
            c.configure(state="disabled")

    def _avisar_falla(self, errores):
        abiertos = programas_de_audio_abiertos()
        if abiertos:
            texto = ("Estos programas están abiertos y pueden estar usando la placa:\n\n    • "
                     + "\n    • ".join(abiertos)
                     + "\n\nCerralos (fijate también en los íconos al lado del reloj) y volvé a apretar ENCENDER.")
        else:
            texto = ("No encontré ningún programa de audio abierto.\n\n"
                     "Desenchufá la placa del USB, esperá unos segundos, volvé a enchufarla y abrí "
                     "de nuevo la Pedalera.")
        if errores:
            texto += f"\n\nDetalle técnico: {errores[0]}"
        messagebox.showerror("No se pudo abrir la placa de audio", texto)

    def _apagar(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.boton.configure(text="▶  ENCENDER", bg="#2e9e6a", activebackground="#257f55")
        for c in (self.cb_api, self.cb_in, self.cb_out, self.cb_canal):
            c.configure(state="readonly")

    def _refrescar(self):
        for lienzo, attr, barra in ((self.lienzos[0], "pico_in", "barra_in"),
                                    (self.lienzos[1], "pico_out", "barra_out")):
            pico = getattr(self, attr)
            setattr(self, attr, 0.0)
            nivel = 20 * math.log10(pico + 1e-9)
            actual = max(nivel, getattr(self, barra) - 2.5)  # cae suave como un vúmetro
            setattr(self, barra, actual)
            self._dibujar_barra(lienzo, actual)

        if self.stream:
            self._vigilar_cortes()
        if self.stream:
            ent, sal = self.stream.latency
            texto = (f"Encendido  ·  {self.stream.samplerate:.0f} Hz  ·  guitarra en canal {self.canal + 1}"
                     f"{' (auto)' if self.canal_auto else ''}  ·  "
                     f"retraso ≈ {(ent + sal) * 1000:.1f} ms  ·  CPU {self.stream.cpu_load * 100:.0f}%  ·  "
                     f"cortes: {self.cortes}")
            if self.aviso:
                texto += f"\n{self.aviso}"
            if self.error:
                texto += f"\nERROR: {self.error}"
            self.estado.configure(text=texto, fg="#e0b848" if self.aviso or self.error else GRIS)
        else:
            self.estado.configure(text="Apagado  ·  apretá ENCENDER para empezar", fg=GRIS)
        self.root.after(50, self._refrescar)

    def _vigilar_cortes(self):
        """Cada segundo: si el audio se corta seguido, sube el retraso un escalón y vuelve a encender."""
        self.ticks += 1
        if self.ticks % 20:
            return
        nuevos = self.cortes - self.cortes_previos
        self.cortes_previos = self.cortes
        if self.ticks == 20 or nuevos < 3:  # el primer segundo siempre trae algún corte al arrancar
            return
        niveles = list(RETRASOS)
        actual = niveles.index(self.var_retraso.get())
        if actual == len(niveles) - 1:
            self.aviso = ("Sigue habiendo cortes. Instalá el driver ASIO de la placa, enchufá la laptop "
                          "al cargador y conectá la placa directo a la PC (sin hub USB).")
            return
        self._apagar()
        self.var_retraso.set(niveles[actual + 1])
        self._encender()
        self.aviso = f"Había cortes de audio: subí el retraso a «{niveles[actual + 1]}» automáticamente."

    def _cerrar(self):
        self._apagar()
        self._guardar()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
